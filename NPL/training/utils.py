"""Shared utilities for offline NLP training and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
TRANSACTION_ID_COLUMN = "transaction_id"
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class SplitConfig:
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = DEFAULT_RANDOM_STATE

    def validate(self) -> None:
        total = self.train_size + self.val_size + self.test_size
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"train/val/test must sum to 1.0, got {total:.6f}"
            )
        for name, value in (
            ("train_size", self.train_size),
            ("val_size", self.val_size),
            ("test_size", self.test_size),
        ):
            if value <= 0 or value >= 1:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")


def load_dataset(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read an interim CSV dataset."""

    return pd.read_csv(Path(path), nrows=nrows)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the minimum columns required for supervised text classification."""

    required_columns = {TEXT_COLUMN, LABEL_COLUMN}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    clean_df = df.copy()
    clean_df = clean_df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    clean_df[TEXT_COLUMN] = clean_df[TEXT_COLUMN].astype(str).str.strip()
    clean_df = clean_df[clean_df[TEXT_COLUMN] != ""]
    clean_df[LABEL_COLUMN] = clean_df[LABEL_COLUMN].astype(int)

    if TRANSACTION_ID_COLUMN in clean_df.columns:
        clean_df[TRANSACTION_ID_COLUMN] = clean_df[TRANSACTION_ID_COLUMN].astype(str)

    clean_df = clean_df.reset_index(drop=True)

    if clean_df[LABEL_COLUMN].nunique() < 2:
        raise ValueError("Training requires at least two label classes")

    return clean_df


def stratified_split(
    X: pd.Series,
    y: pd.Series,
    cfg: SplitConfig | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create train/validation/test splits with label stratification."""

    cfg = cfg or SplitConfig()
    cfg.validate()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(cfg.val_size + cfg.test_size),
        random_state=cfg.random_state,
        stratify=y,
    )

    val_ratio_within_temp = cfg.val_size / (cfg.val_size + cfg.test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1.0 - val_ratio_within_temp),
        random_state=cfg.random_state,
        stratify=y_temp,
    )

    return (
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_val.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def stratified_split_dataframe(
    df: pd.DataFrame,
    label_column: str = LABEL_COLUMN,
    cfg: SplitConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a full dataframe into train/validation/test partitions."""

    cfg = cfg or SplitConfig()
    cfg.validate()

    train_df, temp_df = train_test_split(
        df,
        test_size=(cfg.val_size + cfg.test_size),
        random_state=cfg.random_state,
        stratify=df[label_column],
    )

    val_ratio_within_temp = cfg.val_size / (cfg.val_size + cfg.test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio_within_temp),
        random_state=cfg.random_state,
        stratify=temp_df[label_column],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def find_best_threshold(y_true: pd.Series, y_prob: list[float] | pd.Series) -> tuple[float, float]:
    """Find the probability threshold that maximizes F1 on validation data."""

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5, 0.0

    best_threshold = 0.5
    best_f1 = -1.0
    for precision, recall, threshold in zip(precisions[:-1], recalls[:-1], thresholds):
        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(f1)

    return best_threshold, best_f1


def compute_metrics(
    y_true: pd.Series,
    y_pred: list[int] | pd.Series,
    y_prob: list[float] | pd.Series,
) -> dict[str, Any]:
    """Compute the main fraud metrics for a binary classifier."""

    y_true_series = pd.Series(y_true)
    y_pred_series = pd.Series(y_pred)
    y_prob_series = pd.Series(y_prob)

    metrics = {
        "precision": float(precision_score(y_true_series, y_pred_series, zero_division=0)),
        "recall": float(recall_score(y_true_series, y_pred_series, zero_division=0)),
        "f1": float(f1_score(y_true_series, y_pred_series, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_series, y_prob_series)),
        "pr_auc": float(average_precision_score(y_true_series, y_prob_series)),
        "support": int(len(y_true_series)),
        "fraud_rate": float(y_true_series.mean()),
        "confusion_matrix": confusion_matrix(y_true_series, y_pred_series).tolist(),
    }
    return metrics


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    """Save metrics as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_text_report(text: str, path: str | Path) -> None:
    """Save a human-readable summary."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def save_model(model: Any, path: str | Path) -> None:
    """Persist a Python model artifact with joblib."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Persist a dataframe to CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def format_metrics_block(title: str, metrics: dict[str, Any]) -> str:
    """Format a compact text block for reports."""

    return "\n".join(
        [
            title,
            f"  precision: {metrics['precision']:.4f}",
            f"  recall:    {metrics['recall']:.4f}",
            f"  f1:        {metrics['f1']:.4f}",
            f"  roc_auc:   {metrics['roc_auc']:.4f}",
            f"  pr_auc:    {metrics['pr_auc']:.4f}",
            f"  support:   {metrics['support']}",
            f"  fraud_rate:{metrics['fraud_rate']:.4f}",
            f"  confusion: {metrics['confusion_matrix']}",
        ]
    )
