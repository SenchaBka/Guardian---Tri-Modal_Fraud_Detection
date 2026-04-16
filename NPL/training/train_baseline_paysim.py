"""Train a baseline TF-IDF + Logistic Regression fraud classifier for PaySim."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .evaluate import evaluate_model
from .utils import (
    LABEL_COLUMN,
    TEXT_COLUMN,
    SplitConfig,
    clean_dataset,
    find_best_threshold,
    format_metrics_block,
    load_dataset,
    save_dataframe,
    save_metrics,
    save_model,
    save_text_report,
    stratified_split,
    stratified_split_dataframe,
)


DEFAULT_INPUT = Path("NPL/data/interim/paysim/paysim_nlp_interim.csv")
DEFAULT_MODEL_DIR = Path("models/nlp/baseline")
DEFAULT_REPORTS_DIR = Path("reports/nlp")
DEFAULT_SPLITS_DIR = Path("NPL/data/splits/paysim")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PaySim NLP baseline model")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Interim PaySim CSV")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Artifact output directory")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR, help="Reports output directory")
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR, help="Optional split export directory")
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional number of rows to read from the CSV for quicker experiments",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional stratified sample size after cleaning, useful for local smoke tests",
    )
    parser.add_argument("--train-size", type=float, default=0.70, help="Train split ratio")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--max-features", type=int, default=50000, help="TF-IDF max_features")
    parser.add_argument("--ngram-max", type=int, default=2, help="Maximum n-gram size")
    return parser.parse_args()


def build_summary(
    dataset_name: str,
    df_size: int,
    fraud_rate: float,
    threshold: float,
    val_metrics: dict,
    test_metrics: dict,
) -> str:
    return "\n".join(
        [
            f"Dataset: {dataset_name}",
            f"Rows: {df_size}",
            f"Fraud rate: {fraud_rate:.4f}",
            f"Best validation threshold: {threshold:.4f}",
            "",
            format_metrics_block("Validation", val_metrics),
            "",
            format_metrics_block("Test", test_metrics),
        ]
    )


def log_step(message: str) -> None:
    print(f"[baseline] {message}", flush=True)


def maybe_take_stratified_sample(df, sample_size: int | None, random_state: int):
    if sample_size is None or sample_size >= len(df):
        return df

    fractions = sample_size / len(df)
    sampled = (
        df.groupby(LABEL_COLUMN, group_keys=False)
        .apply(
            lambda part: part.sample(
                n=max(1, round(len(part) * fractions)),
                random_state=random_state,
            )
        )
        .reset_index(drop=True)
    )

    if len(sampled) > sample_size:
        sampled = (
            sampled.groupby(LABEL_COLUMN, group_keys=False)
            .apply(
                lambda part: part.sample(
                    n=max(1, round(len(part) * sample_size / len(sampled))),
                    random_state=random_state,
                )
            )
            .reset_index(drop=True)
        )

    return sampled.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    log_step(f"Reading dataset from {args.input}")
    if args.nrows is not None:
        log_step(f"Limiting CSV read to first {args.nrows} rows")
    df = load_dataset(args.input, nrows=args.nrows)

    log_step(f"Raw dataset shape: {df.shape}")
    log_step("Cleaning dataset")
    df = clean_dataset(df)
    log_step(f"Clean dataset shape: {df.shape}")

    if args.sample_size is not None:
        log_step(f"Taking stratified sample of {args.sample_size} rows")
        df = maybe_take_stratified_sample(df, args.sample_size, args.random_state)
        log_step(f"Sampled dataset shape: {df.shape}")

    X = df[TEXT_COLUMN]
    y = df[LABEL_COLUMN]
    log_step(f"Fraud rate after cleaning/sampling: {y.mean():.4f}")

    split_cfg = SplitConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    log_step("Creating stratified train/validation/test split")
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y, cfg=split_cfg)
    log_step(f"Split sizes train/val/test: {len(X_train)}/{len(X_val)}/{len(X_test)}")

    log_step("Fitting TF-IDF vectorizer")
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        lowercase=True,
        strip_accents="unicode",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    log_step(
        f"Vectorization done. Shapes train/val/test: "
        f"{X_train_vec.shape}/{X_val_vec.shape}/{X_test_vec.shape}"
    )

    log_step("Training LogisticRegression baseline")
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=args.random_state,
    )
    model.fit(X_train_vec, y_train)
    log_step("Model training finished")

    log_step("Searching best threshold on validation split")
    val_prob = model.predict_proba(X_val_vec)[:, 1]
    best_threshold, best_val_f1 = find_best_threshold(y_val, val_prob)
    log_step(f"Best validation threshold: {best_threshold:.4f} | best F1: {best_val_f1:.4f}")

    log_step("Evaluating validation and test splits")
    val_metrics = evaluate_model(model, X_val_vec, y_val, threshold=best_threshold)
    test_metrics = evaluate_model(model, X_test_vec, y_test, threshold=best_threshold)
    val_metrics["best_validation_f1"] = float(best_val_f1)
    test_metrics["best_validation_f1"] = float(best_val_f1)

    metadata = {
        "dataset": "paysim",
        "input_path": str(args.input),
        "nrows": args.nrows,
        "sample_size": args.sample_size,
        "split_config": {
            "train_size": args.train_size,
            "val_size": args.val_size,
            "test_size": args.test_size,
            "random_state": args.random_state,
        },
        "vectorizer": {
            "max_features": args.max_features,
            "ngram_range": [1, args.ngram_max],
        },
        "model": {
            "type": "LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 2000,
        },
        "threshold": best_threshold,
        "dataset_rows": int(len(df)),
        "fraud_rate": float(y.mean()),
    }

    log_step(f"Saving artifacts to {args.model_dir}")
    save_model(vectorizer, args.model_dir / "tfidf_vectorizer.joblib")
    save_model(model, args.model_dir / "logreg_model.joblib")
    (args.model_dir / "metadata.json").parent.mkdir(parents=True, exist_ok=True)
    (args.model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log_step(f"Saving reports to {args.reports_dir}")
    save_metrics(
        {
            "dataset": "paysim",
            "threshold": best_threshold,
            "validation": val_metrics,
            "test": test_metrics,
            "metadata": metadata,
        },
        args.reports_dir / "paysim_baseline_metrics.json",
    )

    summary = build_summary(
        dataset_name="PaySim NLP baseline",
        df_size=len(df),
        fraud_rate=float(y.mean()),
        threshold=best_threshold,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    save_text_report(summary, args.reports_dir / "paysim_baseline_summary.txt")

    if args.splits_dir:
        log_step(f"Saving split CSV files to {args.splits_dir}")
        train_split, val_split, test_split = stratified_split_dataframe(df, cfg=split_cfg)
        save_dataframe(train_split, args.splits_dir / "train.csv")
        save_dataframe(val_split, args.splits_dir / "validation.csv")
        save_dataframe(test_split, args.splits_dir / "test.csv")

    elapsed = time.perf_counter() - t0
    log_step(f"Finished in {elapsed:.2f} seconds")
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {y.mean():.4f}")
    print(f"Train/val/test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    print(f"Best threshold (validation): {best_threshold:.4f}")
    print("")
    print(summary)


if __name__ == "__main__":
    main()
