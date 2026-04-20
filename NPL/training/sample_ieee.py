"""Create a stratified IEEE-CIS sample for faster NLP experiments.

Usage:
python3 -m NPL.training.sample_ieee \
    --input-path NPL/data/interim/ieee/ieee_nlp_interim.csv \
    --output-path NPL/data/processed/ieee/ieee_sample_100k.csv \
    --sample-size 100000
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from .sample_paysim import stratified_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a stratified 100k IEEE NLP sample")
    parser.add_argument(
        "--input-path",
        type=str,
        default="NPL/data/interim/ieee/ieee_nlp_interim.csv",
        help="Path to the IEEE interim CSV",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="NPL/data/processed/ieee/ieee_sample_100k.csv",
        help="Path to save the sampled IEEE CSV",
    )
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    print(f"[sample-ieee] Reading dataset from {args.input_path}")
    df = pd.read_csv(args.input_path)
    print(f"[sample-ieee] Original dataset shape: {df.shape}")

    if args.label_col not in df.columns:
        raise ValueError(
            f"Label column '{args.label_col}' not found. Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["text", args.label_col]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    if "dataset_source" not in df.columns:
        df["dataset_source"] = "ieee"
    print(f"[sample-ieee] After cleaning: {df.shape}")

    fraud_rate_before = df[args.label_col].mean()
    print(f"[sample-ieee] Fraud rate (before): {fraud_rate_before:.6f}")

    print(f"[sample-ieee] Creating stratified sample of size {args.sample_size}")
    df_sample = stratified_sample(
        df,
        label_col=args.label_col,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )

    print(f"[sample-ieee] Sample shape: {df_sample.shape}")
    fraud_rate_after = df_sample[args.label_col].mean()
    print(f"[sample-ieee] Fraud rate (after): {fraud_rate_after:.6f}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    print(f"[sample-ieee] Saving sampled dataset to {args.output_path}")
    df_sample.to_csv(args.output_path, index=False)
    print("[sample-ieee] Done ✔")


if __name__ == "__main__":
    main()
