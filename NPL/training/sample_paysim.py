"""
# Sample PaySim Dataset (Stratified & Reproducible)

This script:
- Reads the full PaySim CSV (or large subset)
- Creates a stratified sample (keeps fraud ratio)
- Saves a clean, fixed dataset for training experiments

Usage:
python3 -m NPL.training.sample_paysim \
    --input-path NPL/data/interim/paysim/paysim_nlp_interim.csv \
    --output-path NPL/data/processed/paysim/paysim_sample_100k.csv \
    --sample-size 100000
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_sample(df, label_col, sample_size, random_state=42):
    """
    Perform stratified sampling while preserving class distribution
    """
    if sample_size >= len(df):
        print("[sample] Requested sample >= dataset size. Returning full dataset.")
        return df

    # Split to keep proportions
    df_sample, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=df[label_col],
        random_state=random_state,
    )

    return df_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument("--label-col", type=str, default="isFraud")

    args = parser.parse_args()

    print(f"[sample] Reading dataset from {args.input_path}")
    df = pd.read_csv(args.input_path)

    print(f"[sample] Original dataset shape: {df.shape}")

    # Normalize column names (strip spaces, lower case)
    original_cols = list(df.columns)
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    # Resolve label column with case-insensitive matching and common aliases
    label_col = args.label_col
    if label_col not in df.columns:
        candidates = ["isFraud", "is_fraud", "fraud", "label", "target"]
        # try exact lower-case match
        if label_col.lower() in lower_map:
            label_col = lower_map[label_col.lower()]
        else:
            # try common aliases
            found = None
            for cand in candidates:
                if cand in df.columns:
                    found = cand
                    break
                if cand.lower() in lower_map:
                    found = lower_map[cand.lower()]
                    break
            if found is None:
                print("[sample] Available columns:", original_cols)
                raise ValueError(f"Label column '{args.label_col}' not found in dataset")
            else:
                print(f"[sample] Using detected label column: '{found}'")
                label_col = found
    else:
        label_col = args.label_col

    # Drop NA just in case
    df = df.dropna()
    print(f"[sample] After cleaning: {df.shape}")

    # Fraud rate before sampling
    fraud_rate_before = df[label_col].mean()
    print(f"[sample] Fraud rate (before): {fraud_rate_before:.6f}")

    print(f"[sample] Creating stratified sample of size {args.sample_size}")
    df_sample = stratified_sample(
        df,
        label_col=label_col,
        sample_size=args.sample_size,
    )

    print(f"[sample] Sample shape: {df_sample.shape}")

    # Fraud rate after sampling
    fraud_rate_after = df_sample[label_col].mean()
    print(f"[sample] Fraud rate (after): {fraud_rate_after:.6f}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print(f"[sample] Saving sampled dataset to {args.output_path}")
    df_sample.to_csv(args.output_path, index=False)

    print("[sample] Done ✔")


if __name__ == "__main__":
    main()
