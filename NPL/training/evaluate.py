"""Reusable evaluation helpers for Guardian NLP models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .utils import (
    LABEL_COLUMN,
    TEXT_COLUMN,
    compute_metrics,
    load_dataset,
)


def evaluate_model(
    model: Any,
    X_test: pd.Series,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float | int | list[list[int]]]:
    """Evaluate a classifier that exposes predict_proba."""

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["threshold"] = float(threshold)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved baseline NLP model")
    parser.add_argument("--model", type=Path, required=True, help="Path to saved model artifact")
    parser.add_argument(
        "--vectorizer",
        type=Path,
        default=None,
        help="Optional vectorizer artifact for text-based sklearn models",
    )
    parser.add_argument("--test-csv", type=Path, required=True, help="Path to test CSV")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model)
    df = load_dataset(args.test_csv)
    X_test = df[TEXT_COLUMN].astype(str)
    y_test = df[LABEL_COLUMN].astype(int)
    if args.vectorizer is not None:
        vectorizer = joblib.load(args.vectorizer)
        X_test = vectorizer.transform(X_test)
    metrics = evaluate_model(model, X_test, y_test, threshold=args.threshold)

    print("Evaluation metrics")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
