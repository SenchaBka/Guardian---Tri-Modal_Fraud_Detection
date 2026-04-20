"""Fine-tune FinBERT for binary fraud classification on any aligned NLP dataset."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .utils import (
    LABEL_COLUMN,
    TEXT_COLUMN,
    SplitConfig,
    clean_dataset,
    compute_metrics,
    find_best_threshold,
    format_metrics_block,
    load_dataset,
    save_dataframe,
    save_metrics,
    save_text_report,
    stratified_split_dataframe,
)


DEFAULT_INPUT = Path("NPL/data/processed/paysim/paysim_sample_100k.csv")
DEFAULT_MODEL_DIR = Path("models/nlp/finbert")
DEFAULT_REPORTS_DIR = Path("reports/nlp")
DEFAULT_SPLITS_DIR = Path("NPL/data/splits/finbert")

# Small search grid designed to be practical on a Mac laptop.
# Dataset size and epochs are inherited from CLI args so local experiments stay flexible.
SAFE_SEARCH_GRID = [
    {"learning_rate": 2e-5, "max_length": 128},
    {"learning_rate": 2e-5, "max_length": 256},
    {"learning_rate": 3e-5, "max_length": 128},
    {"learning_rate": 3e-5, "max_length": 256},
]

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT on an aligned NLP fraud dataset")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Aligned NLP CSV")
    parser.add_argument("--dataset-name", type=str, default=None, help="Optional display name for reports")
    parser.add_argument("--run-name", type=str, default=None, help="Optional artifact prefix (defaults to dataset)")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Checkpoint parent directory")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR, help="Reports output directory")
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR, help="Optional split export directory")
    parser.add_argument("--model-name", type=str, default="ProsusAI/finbert", help="Base Hugging Face model")
    parser.add_argument("--nrows", type=int, default=None, help="Optional number of rows to read")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional stratified sample size after cleaning for local experiments",
    )
    parser.add_argument("--train-size", type=float, default=0.70, help="Train split ratio")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max length")
    parser.add_argument("--num-train-epochs", type=float, default=2.0, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--train-batch-size", type=int, default=8, help="Per-device train batch size")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Per-device eval batch size")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run a small, Mac-friendly hyperparameter search instead of a single training run",
    )
    return parser.parse_args()


def log_step(message: str) -> None:
    logger.info(message)
    print(f"[finbert] {message}", flush=True)


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


def sanitize_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


class TokenizedTextDataset(Dataset):
    """Torch dataset wrapper for tokenized text classification inputs."""

    def __init__(self, encodings: dict[str, list[int]], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross entropy for imbalanced fraud labels."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def softmax_binary_fraud_probs(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(logits)
    probs = exps / exps.sum(axis=1, keepdims=True)
    return probs[:, 1]


def trainer_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    fraud_probs = softmax_binary_fraud_probs(logits)
    y_pred = (fraud_probs >= 0.5).astype(int)
    metrics = compute_metrics(labels, y_pred, fraud_probs)
    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
    }


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
            f"Dataset: {dataset_name} FinBERT fine-tune",
            f"Rows: {df_size}",
            f"Fraud rate: {fraud_rate:.4f}",
            f"Best validation threshold: {threshold:.4f}",
            "",
            format_metrics_block("Validation", val_metrics),
            "",
            format_metrics_block("Test", test_metrics),
        ]
    )


def build_training_arguments(output_dir: str, args: argparse.Namespace, use_fp16: bool) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments.__init__)
    supported = set(signature.parameters.keys())

    kwargs = {
        "output_dir": output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "load_best_model_at_end": True,
        "metric_for_best_model": "pr_auc",
        "greater_is_better": True,
        "save_total_limit": 2,
        "report_to": "none",
        "fp16": use_fp16,
        "dataloader_num_workers": 0,
        "seed": args.random_state,
    }

    if "overwrite_output_dir" in supported:
        kwargs["overwrite_output_dir"] = True
    if "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in supported:
        kwargs["eval_strategy"] = "epoch"
    if "save_strategy" in supported:
        kwargs["save_strategy"] = "epoch"
    if "logging_strategy" in supported:
        kwargs["logging_strategy"] = "steps"
    if "logging_steps" in supported:
        kwargs["logging_steps"] = 50

    filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported}
    return TrainingArguments(**filtered_kwargs)


def clone_args(args: argparse.Namespace, **overrides) -> argparse.Namespace:
    data = vars(args).copy()
    data.update(overrides)
    return argparse.Namespace(**data)


def make_run_name(args: argparse.Namespace, dataset_name: str) -> str:
    lr_str = f"{args.learning_rate:.0e}".replace("-0", "-")
    epoch_str = str(args.num_train_epochs).replace(".", "p")
    sample_str = "full" if args.sample_size is None else str(args.sample_size)
    nrows_str = "full" if args.nrows is None else str(args.nrows)
    dataset_slug = sanitize_name(dataset_name)
    return (
        f"{dataset_slug}_finbert_lr{lr_str}_len{args.max_length}_ep{epoch_str}"
        f"_n{nrows_str}_s{sample_str}"
    )


def rank_result(result: dict) -> tuple:
    test = result["test"]
    return (
        float(test["pr_auc"]),
        float(test["recall"]),
        float(test["f1"]),
        float(test["precision"]),
    )


def run_experiment(args: argparse.Namespace) -> dict:
    t0 = time.perf_counter()

    log_step(f"Reading dataset from {args.input}")
    if args.nrows is not None:
        log_step(f"Limiting CSV read to first {args.nrows} rows")
    df = load_dataset(args.input, nrows=args.nrows)

    dataset_name = infer_dataset_name(df, args.input, args.dataset_name)
    run_name = args.run_name or make_run_name(args, dataset_name)
    artifact_dir = args.model_dir / run_name
    report_dir = args.reports_dir / run_name
    split_dir = args.splits_dir / run_name if args.splits_dir else None

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

    fraud_rate = float(df[LABEL_COLUMN].mean())
    log_step(f"Fraud rate after cleaning/sampling: {fraud_rate:.4f}")

    split_cfg = SplitConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    log_step("Creating stratified train/validation/test split")
    train_df, val_df, test_df = stratified_split_dataframe(df, cfg=split_cfg)
    log_step(f"Split sizes train/val/test: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    if split_dir:
        log_step(f"Saving split CSV files to {split_dir}")
        save_dataframe(train_df, split_dir / "train.csv")
        save_dataframe(val_df, split_dir / "validation.csv")
        save_dataframe(test_df, split_dir / "test.csv")

    log_step(f"Loading tokenizer and base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    log_step("Tokenizing train/validation/test datasets")
    train_encodings = tokenizer(
        train_df[TEXT_COLUMN].tolist(),
        truncation=True,
        max_length=args.max_length,
    )
    val_encodings = tokenizer(
        val_df[TEXT_COLUMN].tolist(),
        truncation=True,
        max_length=args.max_length,
    )
    test_encodings = tokenizer(
        test_df[TEXT_COLUMN].tolist(),
        truncation=True,
        max_length=args.max_length,
    )

    train_dataset = TokenizedTextDataset(train_encodings, train_df[LABEL_COLUMN].tolist())
    val_dataset = TokenizedTextDataset(val_encodings, val_df[LABEL_COLUMN].tolist())
    test_dataset = TokenizedTextDataset(test_encodings, test_df[LABEL_COLUMN].tolist())

    class_counts = train_df[LABEL_COLUMN].value_counts().sort_index()
    total = int(class_counts.sum())
    class_weights = torch.tensor(
        [total / (2.0 * class_counts.get(i, 1)) for i in range(2)],
        dtype=torch.float32,
    )
    log_step(f"Using class weights: {class_weights.tolist()}")

    use_fp16 = torch.cuda.is_available()
    training_args = build_training_arguments(
        output_dir=str(artifact_dir),
        args=args,
        use_fp16=use_fp16,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=trainer_metrics,
        class_weights=class_weights,
    )

    log_step("Starting FinBERT fine-tuning")
    trainer.train()
    log_step("Training finished")

    log_step("Running validation predictions for threshold search")
    val_predictions = trainer.predict(val_dataset)
    val_probs = softmax_binary_fraud_probs(val_predictions.predictions)
    best_threshold, best_val_f1 = find_best_threshold(val_df[LABEL_COLUMN], val_probs)
    val_pred_labels = (val_probs >= best_threshold).astype(int)
    val_metrics = compute_metrics(val_df[LABEL_COLUMN], val_pred_labels, val_probs)
    val_metrics["threshold"] = float(best_threshold)
    val_metrics["best_validation_f1"] = float(best_val_f1)
    log_step(f"Best validation threshold: {best_threshold:.4f} | best F1: {best_val_f1:.4f}")

    log_step("Running final test evaluation")
    test_predictions = trainer.predict(test_dataset)
    test_probs = softmax_binary_fraud_probs(test_predictions.predictions)
    test_pred_labels = (test_probs >= best_threshold).astype(int)
    test_metrics = compute_metrics(test_df[LABEL_COLUMN], test_pred_labels, test_probs)
    test_metrics["threshold"] = float(best_threshold)
    test_metrics["best_validation_f1"] = float(best_val_f1)

    log_step(f"Saving best checkpoint and tokenizer to {artifact_dir}")
    trainer.save_model(str(artifact_dir))
    tokenizer.save_pretrained(str(artifact_dir))

    metadata = {
        "dataset": dataset_name,
        "run_name": run_name,
        "input_path": str(args.input),
        "model_name": args.model_name,
        "nrows": args.nrows,
        "sample_size": args.sample_size,
        "dataset_rows": int(len(df)),
        "fraud_rate": fraud_rate,
        "split_config": {
            "train_size": args.train_size,
            "val_size": args.val_size,
            "test_size": args.test_size,
            "random_state": args.random_state,
        },
        "training": {
            "max_length": args.max_length,
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "warmup_ratio": args.warmup_ratio,
            "fp16": use_fp16,
        },
        "threshold": best_threshold,
        "class_weights": class_weights.tolist(),
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report_payload = {
        "dataset": dataset_name,
        "run_name": run_name,
        "model_version": artifact_dir.name,
        "threshold": best_threshold,
        "validation": val_metrics,
        "test": test_metrics,
        "metadata": metadata,
    }

    log_step(f"Saving reports to {report_dir}")
    save_metrics(report_payload, report_dir / "metrics.json")

    summary = build_summary(
        dataset_name=dataset_name,
        df_size=len(df),
        fraud_rate=fraud_rate,
        threshold=best_threshold,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    save_text_report(summary, report_dir / "summary.txt")

    elapsed = time.perf_counter() - t0
    log_step(f"Finished in {elapsed:.2f} seconds")
    print(summary)

    return {
        "run_name": run_name,
        "dataset_name": dataset_name,
        "elapsed_seconds": elapsed,
        "threshold": best_threshold,
        "validation": val_metrics,
        "test": test_metrics,
        "metadata": metadata,
        "summary": summary,
    }


def run_small_grid_search(args: argparse.Namespace) -> None:
    base_reports_dir = args.reports_dir / "grid_search"
    base_reports_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset_name or args.input.stem
    results: list[dict] = []

    print("\nStarting small FinBERT search grid...")
    print(f"Dataset: {dataset_name}")
    print(f"Total runs: {len(SAFE_SEARCH_GRID)}")

    for idx, overrides in enumerate(SAFE_SEARCH_GRID, start=1):
        run_args = clone_args(args, **overrides)
        grid_run_name = make_run_name(run_args, dataset_name)
        run_args.run_name = grid_run_name

        log_step(f"[grid {idx}/{len(SAFE_SEARCH_GRID)}] Starting {grid_run_name}")
        result = run_experiment(run_args)
        result["grid_overrides"] = overrides
        results.append(result)

    ranked = sorted(results, key=rank_result, reverse=True)
    best = ranked[0]

    leaderboard_lines = [
        f"{dataset_name} FinBERT Search Leaderboard",
        "",
        "Ranking priority: PR-AUC > Recall > F1 > Precision",
        "",
        "Runs:",
    ]

    for i, item in enumerate(ranked, start=1):
        test = item["test"]
        leaderboard_lines.extend(
            [
                f"{i}. {item['run_name']}",
                f"   pr_auc={float(test['pr_auc']):.4f}",
                f"   recall={float(test['recall']):.4f}",
                f"   f1={float(test['f1']):.4f}",
                f"   precision={float(test['precision']):.4f}",
                f"   roc_auc={float(test['roc_auc']):.4f}",
                f"   threshold={float(item['threshold']):.4f}",
                f"   elapsed_seconds={float(item['elapsed_seconds']):.2f}",
                "",
            ]
        )

    leaderboard_lines.extend(
        [
            "Best run selected:",
            f"  {best['run_name']}",
            f"  PR-AUC={float(best['test']['pr_auc']):.4f}",
            f"  Recall={float(best['test']['recall']):.4f}",
            f"  F1={float(best['test']['f1']):.4f}",
            f"  Precision={float(best['test']['precision']):.4f}",
            f"  Threshold={float(best['threshold']):.4f}",
        ]
    )

    leaderboard_text = "\n".join(leaderboard_lines)
    print("\n" + leaderboard_text)

    save_text_report(leaderboard_text, base_reports_dir / f"{sanitize_name(dataset_name)}_leaderboard.txt")
    save_metrics(
        {
            "dataset": dataset_name,
            "best_run": best["run_name"],
            "runs": ranked,
        },
        base_reports_dir / f"{sanitize_name(dataset_name)}_leaderboard.json",
    )


def main() -> None:
    args = parse_args()

    if args.grid_search:
        run_small_grid_search(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
