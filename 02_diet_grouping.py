#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

FOODS = (
    "Beer", "Berries", "Butter", "Citrus fruits", "Coffee", "Eggs",
    "Fish and other seafood", "Fried and fast foods", "Fruit juices",
    "Green leafy vegetables", "High-fat dairy products", "Legumes", "Liquor",
    "Low-fat dairy products", "Margarine", "Nuts", "Olive oil", "Organ meats",
    "Other fruits", "Other vegetables", "Pizza", "Potatoes", "Poultry",
    "Processed meats", "Red meats", "Red/Orange vegetables",
    "Snacks and pastries", "Sweetened beverages", "Tea", "Water", "Wine",
    "Whole grains", "Refined grains", "Other oil",
)


def compact_labels(labels: pd.Series) -> pd.Series:
    present = sorted(
        [x for x in labels.dropna().astype(str).unique().tolist() if x.startswith("F")],
        key=lambda x: int(x[1:]) if x[1:].isdigit() else 999,
    )
    mapping = {old: f"F{i + 1}" for i, old in enumerate(present)}
    return labels.map(mapping).astype("string")


def assign_by_thresholds(values: pd.Series, thresholds: list[float]) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").fillna(0.0)
    labels = pd.Series(pd.NA, index=x.index, dtype="string")
    thresholds = sorted(float(t) for t in thresholds if np.isfinite(t))
    prev = -np.inf
    group_num = 1
    for threshold in thresholds:
        mask = (x > prev) & (x <= threshold)
        if mask.any():
            labels.loc[mask] = f"F{group_num}"
            group_num += 1
        prev = threshold
    mask = x > prev
    if mask.any():
        labels.loc[mask] = f"F{group_num}"
    return compact_labels(labels)


def quantile_thresholds(values: pd.Series, probs: list[float]) -> list[float]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    x = x[np.isfinite(x)]
    if x.empty:
        return []
    raw = [float(x.quantile(p, interpolation="linear")) for p in probs]
    out: list[float] = []
    for val in raw:
        if not out or abs(val - out[-1]) > 1e-12:
            out.append(val)
    return out


def zero_plus_positive_quartile(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").fillna(0.0)
    labels = pd.Series(pd.NA, index=x.index, dtype="string")
    zero_mask = x <= 0
    if zero_mask.any():
        labels.loc[zero_mask] = "F1"

    positive = x[x > 0]
    if positive.empty:
        return compact_labels(labels)

    thresholds = quantile_thresholds(positive, [0.25, 0.50, 0.75])
    pos_labels = assign_by_thresholds(positive, thresholds)
    offset = 1 if zero_mask.any() else 0
    for label in sorted(pos_labels.dropna().unique(), key=lambda y: int(str(y)[1:])):
        target = f"F{int(str(label)[1:]) + offset}"
        labels.loc[pos_labels.index[pos_labels == label]] = target
    return compact_labels(labels)


def all_quintile_zero_merged(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").fillna(0.0)
    thresholds = quantile_thresholds(x, [0.20, 0.40, 0.60, 0.80])
    labels = assign_by_thresholds(x, thresholds)
    zero_mask = x <= 0
    if zero_mask.any():
        labels.loc[zero_mask] = "F1"
    return compact_labels(labels)


def build_groups(diet: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_df = pd.DataFrame({"eid": diet["eid"]})
    summary_rows: list[dict] = []

    assigner = {
        "zero_plus_positive_quartile": zero_plus_positive_quartile,
        "all_quintile_zero_merged": all_quintile_zero_merged,
    }[policy]

    for food in [c for c in diet.columns if c != "eid"]:
        values = pd.to_numeric(diet[food], errors="coerce").fillna(0.0)
        groups = assigner(values)
        group_col = f"{food}__group"
        group_df[group_col] = groups

        for label in sorted(groups.dropna().unique(), key=lambda x: int(str(x)[1:])):
            vals = values.loc[groups == label]
            summary_rows.append(
                {
                    "Food": food,
                    "Grade": label,
                    "N": int((groups == label).sum()),
                    "Median": float(vals.median()) if len(vals) else np.nan,
                    "Min": float(vals.min()) if len(vals) else np.nan,
                    "Max": float(vals.max()) if len(vals) else np.nan,
                    "Zero_N": int((vals <= 0).sum()),
                    "Policy": policy,
                }
            )

    return group_df, pd.DataFrame(summary_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build zero-merged food groups from the training subset")
    parser.add_argument("--input_csv", default="input/train_A_random.csv")
    parser.add_argument("--output_dir", default="input")
    parser.add_argument(
        "--policy",
        default="all_quintile_zero_merged",
        choices=["zero_plus_positive_quartile", "all_quintile_zero_merged"],
    )
    args = parser.parse_args()

    diet = pd.read_csv(args.input_csv, low_memory=False)
    if "eid" not in diet.columns:
        raise ValueError("Diet matrix must contain eid")
    missing = [food for food in FOODS if food not in diet.columns]
    if missing:
        raise ValueError(f"missing food columns: {missing}")
    diet = diet.loc[:, ["eid", *FOODS]]

    group_df, summary_df = build_groups(diet, args.policy)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    diet.to_csv(out / "03_Diet_Matrix_34Foods.csv", index=False)
    group_df.to_csv(out / "Diet_Groupings_ZeroMerged.csv", index=False)
    summary_df.to_csv(out / "Diet_Groupings_ZeroMerged_Summary.csv", index=False)
    (out / "grouping_policy.txt").write_text(args.policy + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
