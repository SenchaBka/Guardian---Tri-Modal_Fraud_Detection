"""Merge the 100k PaySim and IEEE sampled NLP datasets.

Usage:
python3 -m NPL.training.merge_sampled_datasets \
    --paysim-path NPL/data/processed/paysim/paysim_sample_100k.csv \
    --ieee-path NPL/data/processed/ieee/ieee_sample_100k.csv \
    --output-path NPL/data/processed/combined/nlp_combined_200k.csv \
    --drop-duplicates
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .merge_interim_datasets import summarize, validate_schema


def load_sample_csv(path: Path, dataset_source: str, dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "dataset_source" not in df.columns:
        df["dataset_source"] = dataset_source
    return validate_schema(df, dataset_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the 100k PaySim and IEEE sampled NLP datasets"
    )
    parser.add_argument(
        "--paysim-path",
        type=Path,
        default=Path("NPL/data/processed/paysim/paysim_sample_100k.csv"),
        help="Path to the sampled PaySim CSV",
    )
    parser.add_argument(
        "--ieee-path",
        type=Path,
        default=Path("NPL/data/processed/ieee/ieee_sample_100k.csv"),
        help="Path to the sampled IEEE CSV",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("NPL/data/processed/combined/nlp_combined_200k.csv"),
        help="Path for the merged sampled output CSV",
    )
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="Drop duplicate rows after merging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[merge-sampled] Reading PaySim sample from {args.paysim_path}")
    paysim_df = load_sample_csv(args.paysim_path, "paysim", "PaySim sample")
    summarize(paysim_df, "PaySim sample")

    print(f"[merge-sampled] Reading IEEE sample from {args.ieee_path}")
    ieee_df = load_sample_csv(args.ieee_path, "ieee", "IEEE sample")
    summarize(ieee_df, "IEEE sample")

    combined = pd.concat([paysim_df, ieee_df], ignore_index=True)

    if args.drop_duplicates:
        before = len(combined)
        combined = combined.drop_duplicates().reset_index(drop=True)
        print(f"[merge-sampled] Dropped {before - len(combined)} duplicate rows")

    summarize(combined, "Combined sampled")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_path, index=False)
    print(f"[merge-sampled] Saved merged sampled dataset to {args.output_path}")


if __name__ == "__main__":
    main()
