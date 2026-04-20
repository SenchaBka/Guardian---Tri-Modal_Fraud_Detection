"""Merge aligned interim NLP datasets into a single training CSV.

Usage:
python3 -m NPL.training.merge_interim_datasets \
    --paysim-path NPL/data/interim/paysim/paysim_nlp_interim.csv \
    --ieee-path NPL/data/interim/ieee/ieee_nlp_interim.csv \
    --output-path NPL/data/interim/combined/nlp_combined_interim.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
    "transaction_id",
    "text",
    "label",
    "amount",
    "transaction_type",
    "dataset_source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge PaySim and IEEE interim NLP datasets into one shared training CSV"
    )
    parser.add_argument(
        "--paysim-path",
        type=Path,
        default=Path("NPL/data/interim/paysim/paysim_nlp_interim.csv"),
        help="Path to the PaySim interim CSV",
    )
    parser.add_argument(
        "--ieee-path",
        type=Path,
        default=Path("NPL/data/interim/ieee/ieee_nlp_interim.csv"),
        help="Path to the IEEE interim CSV",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("NPL/data/interim/combined/nlp_combined_interim.csv"),
        help="Path for the merged output CSV",
    )
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="Drop duplicate rows after merging",
    )
    return parser.parse_args()


def validate_schema(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    extra = [col for col in df.columns if col not in EXPECTED_COLUMNS]

    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")
    if extra:
        raise ValueError(f"{dataset_name} has unexpected columns: {extra}")

    df = df[EXPECTED_COLUMNS].copy()
    df["transaction_id"] = df["transaction_id"].astype(str)
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["transaction_type"] = df["transaction_type"].astype(str)
    df["dataset_source"] = df["dataset_source"].astype(str).str.lower()

    if df["text"].eq("").any():
        raise ValueError(f"{dataset_name} contains empty text rows")
    if df["label"].nunique() < 2:
        raise ValueError(f"{dataset_name} must contain at least two label classes")

    return df


def summarize(df: pd.DataFrame, name: str) -> None:
    fraud_rate = float(df["label"].mean()) if len(df) else 0.0
    print(f"[merge] {name}: rows={len(df)} | fraud_rate={fraud_rate:.6f}")
    print(f"[merge] {name}: dataset_source={sorted(df['dataset_source'].unique().tolist())}")


def main() -> None:
    args = parse_args()

    print(f"[merge] Reading PaySim interim CSV from {args.paysim_path}")
    paysim_df = pd.read_csv(args.paysim_path)
    paysim_df = validate_schema(paysim_df, "PaySim")
    summarize(paysim_df, "PaySim")

    print(f"[merge] Reading IEEE interim CSV from {args.ieee_path}")
    ieee_df = pd.read_csv(args.ieee_path)
    ieee_df = validate_schema(ieee_df, "IEEE")
    summarize(ieee_df, "IEEE")

    combined = pd.concat([paysim_df, ieee_df], ignore_index=True)

    if args.drop_duplicates:
        before = len(combined)
        combined = combined.drop_duplicates().reset_index(drop=True)
        print(f"[merge] Dropped {before - len(combined)} duplicate rows")

    summarize(combined, "Combined")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_path, index=False)
    print(f"[merge] Saved merged dataset to {args.output_path}")


if __name__ == "__main__":
    main()
