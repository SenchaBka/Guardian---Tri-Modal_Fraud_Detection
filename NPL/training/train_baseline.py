"""Train a baseline TF-IDF + Logistic Regression fraud classifier for any aligned NLP dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from .evaluate import evaluate_model, get_positive_scores
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


DEFAULT_INPUT = Path("NPL/data/processed/paysim/paysim_sample_100k.csv")
DEFAULT_MODEL_DIR = Path("models/nlp/baseline")
DEFAULT_REPORTS_DIR = Path("reports/nlp")
DEFAULT_SPLITS_DIR = Path("NPL/data/splits/baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TF-IDF classical NLP baseline model")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Aligned NLP CSV")
    parser.add_argument("--dataset-name", type=str, default=None, help="Optional display name for reports")
    parser.add_argument("--run-name", type=str, default=None, help="Optional artifact prefix (defaults to input stem)")
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
    parser.add_argument("--min-df", type=int, default=1, help="Ignore terms appearing in fewer than this many docs")
    parser.add_argument(
        "--max-df",
        type=float,
        default=1.0,
        help="Ignore terms appearing in more than this document fraction",
    )
    parser.add_argument(
        "--sublinear-tf",
        action="store_true",
        help="Apply sublinear term frequency scaling in TF-IDF",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="logreg",
        choices=["logreg", "linear_svm"],
        help="Classical model to train on top of TF-IDF features",
    )
    parser.add_argument("--svm-c", type=float, default=1.0, help="Regularization strength for LinearSVC")
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

    sampled, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=df[LABEL_COLUMN],
        random_state=random_state,
    )
    return sampled.reset_index(drop=True)


def infer_dataset_name(df, input_path: Path, explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name
    if "dataset_source" in df.columns:
        sources = sorted(df["dataset_source"].dropna().astype(str).str.lower().unique().tolist())
        if len(sources) == 1:
            return sources[0]
        if len(sources) > 1:
            return "_".join(sources)
    return input_path.stem


def get_model_filename(model_type: str) -> str:
    if model_type == "linear_svm":
        return "linear_svm_model.joblib"
    return "logreg_model.joblib"


def build_model(model_type: str, random_state: int, svm_c: float):
    if model_type == "linear_svm":
        return LinearSVC(
            C=svm_c,
            class_weight="balanced",
            random_state=random_state,
            max_iter=5000,
        )
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
    )


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    log_step(f"Reading dataset from {args.input}")
    if args.nrows is not None:
        log_step(f"Limiting CSV read to first {args.nrows} rows")
    df = load_dataset(args.input, nrows=args.nrows)

    dataset_name = infer_dataset_name(df, args.input, args.dataset_name)
    run_name = args.run_name or f"{dataset_name.replace(' ', '_').lower()}_{args.model_type}"

    log_step(f"Dataset name: {dataset_name}")
    log_step(f"Run name: {run_name}")
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
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=args.sublinear_tf,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    log_step(
        f"Vectorization done. Shapes train/val/test: "
        f"{X_train_vec.shape}/{X_val_vec.shape}/{X_test_vec.shape}"
    )

    log_step(f"Training {args.model_type} baseline")
    model = build_model(args.model_type, args.random_state, args.svm_c)
    model.fit(X_train_vec, y_train)
    log_step("Model training finished")

    log_step("Searching best threshold on validation split")
    val_prob = get_positive_scores(model, X_val_vec)
    best_threshold, best_val_f1 = find_best_threshold(y_val, val_prob)
    log_step(f"Best validation threshold: {best_threshold:.4f} | best F1: {best_val_f1:.4f}")

    log_step("Evaluating validation and test splits")
    val_metrics = evaluate_model(model, X_val_vec, y_val, threshold=best_threshold)
    test_metrics = evaluate_model(model, X_test_vec, y_test, threshold=best_threshold)
    val_metrics["best_validation_f1"] = float(best_val_f1)
    test_metrics["best_validation_f1"] = float(best_val_f1)

    metadata = {
        "dataset": dataset_name,
        "run_name": run_name,
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
            "min_df": args.min_df,
            "max_df": args.max_df,
            "sublinear_tf": args.sublinear_tf,
        },
        "model": {
            "type": args.model_type,
            "class_weight": "balanced",
        },
        "threshold": best_threshold,
        "dataset_rows": int(len(df)),
        "fraud_rate": float(y.mean()),
    }
    if args.model_type == "logreg":
        metadata["model"]["max_iter"] = 2000
    if args.model_type == "linear_svm":
        metadata["model"]["max_iter"] = 5000
        metadata["model"]["C"] = args.svm_c

    artifact_dir = args.model_dir / run_name
    report_dir = args.reports_dir
    split_dir = args.splits_dir / run_name if args.splits_dir else None

    log_step(f"Saving artifacts to {artifact_dir}")
    save_model(vectorizer, artifact_dir / "tfidf_vectorizer.joblib")
    save_model(model, artifact_dir / get_model_filename(args.model_type))
    (artifact_dir / "metadata.json").parent.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log_step(f"Saving reports to {report_dir}")
    save_metrics(
        {
            "dataset": dataset_name,
            "threshold": best_threshold,
            "validation": val_metrics,
            "test": test_metrics,
            "metadata": metadata,
        },
        report_dir / f"{run_name}_baseline_metrics.json",
    )

    summary = build_summary(
        dataset_name=f"{dataset_name} NLP baseline ({args.model_type})",
        df_size=len(df),
        fraud_rate=float(y.mean()),
        threshold=best_threshold,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    save_text_report(summary, report_dir / f"{run_name}_baseline_summary.txt")

    if split_dir:
        log_step(f"Saving split CSV files to {split_dir}")
        train_split, val_split, test_split = stratified_split_dataframe(df, cfg=split_cfg)
        save_dataframe(train_split, split_dir / "train.csv")
        save_dataframe(val_split, split_dir / "validation.csv")
        save_dataframe(test_split, split_dir / "test.csv")

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
