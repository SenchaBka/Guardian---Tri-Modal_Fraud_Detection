"""Create a stratified sample from any aligned NLP dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .sample_paysim import stratified_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a stratified sample from an aligned NLP CSV")
    parser.add_argument("--input-path", type=Path, required=True, help="Path to the aligned NLP CSV")
    parser.add_argument("--output-path", type=Path, required=True, help="Where to save the sampled CSV")
    parser.add_argument("--sample-size", type=int, default=100000, help="Target sample size")
    parser.add_argument("--label-col", type=str, default="label", help="Label column name")
    parser.add_argument("--random-state", type=int, default=42, help="Sampling seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[sample-dataset] Reading dataset from {args.input_path}")
    df = pd.read_csv(args.input_path)
    print(f"[sample-dataset] Original dataset shape: {df.shape}")

    if args.label_col not in df.columns:
        raise ValueError(
            f"Label column '{args.label_col}' not found. Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["text", args.label_col]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    print(f"[sample-dataset] After cleaning: {df.shape}")

    fraud_rate_before = df[args.label_col].mean()
    print(f"[sample-dataset] Fraud rate (before): {fraud_rate_before:.6f}")

    print(f"[sample-dataset] Creating stratified sample of size {args.sample_size}")
    df_sample = stratified_sample(
        df,
        label_col=args.label_col,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )

    print(f"[sample-dataset] Sample shape: {df_sample.shape}")
    fraud_rate_after = df_sample[args.label_col].mean()
    print(f"[sample-dataset] Fraud rate (after): {fraud_rate_after:.6f}")

    if "dataset_source" in df_sample.columns:
        source_counts = df_sample["dataset_source"].value_counts().to_dict()
        print(f"[sample-dataset] Dataset source mix: {source_counts}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(args.output_path, index=False)
    print(f"[sample-dataset] Saved sampled dataset to {args.output_path}")


if __name__ == "__main__":
    main()
