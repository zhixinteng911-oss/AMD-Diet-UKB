#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lifelines import CoxPHFitter

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

LOCKED_34_FOODS = (
    "Beer", "Berries", "Butter", "Citrus fruits", "Coffee", "Eggs",
    "Fish and other seafood", "Fried and fast foods", "Fruit juices",
    "Green leafy vegetables", "High-fat dairy products", "Legumes", "Liquor",
    "Low-fat dairy products", "Margarine", "Nuts", "Olive oil", "Organ meats",
    "Other fruits", "Other vegetables", "Pizza", "Potatoes", "Poultry",
    "Processed meats", "Red meats", "Red/Orange vegetables",
    "Snacks and pastries", "Sweetened beverages", "Tea", "Water", "Wine",
    "Whole grains", "Refined grains", "Other oil",
)


def validate_locked_34_food_columns(
    columns,
    *,
    context="diet matrix",
    allow_eid=True,
    require_order=False,
):
    """Check that a column collection contains exactly the locked 34 foods.

    Raises ValueError when locked foods are missing or unexpected columns are
    present; returns the locked list when validation passes.
    """
    cols = [c for c in columns if c != "eid"] if allow_eid else list(columns)
    locked = list(LOCKED_34_FOODS)
    missing = [food for food in locked if food not in cols]
    unexpected = [col for col in cols if col not in locked]
    if missing or unexpected:
        raise ValueError(
            f"[{context}] column mismatch: missing={missing}, unexpected={unexpected}"
        )
    if require_order:
        present = [col for col in cols if col in locked]
        if present != locked:
            raise ValueError(f"[{context}] locked food order not preserved")
    return locked

try:
    from scipy.stats import chi2
except Exception:
    chi2 = None


DEFAULT_TRAIN = os.environ.get("AMD_TRAIN_CSV", "input/train_A_random.csv")
DEFAULT_GROUPINGS = os.environ.get("AMD_GROUPINGS_CSV", "input/Diet_Groupings_ZeroMerged.csv")
DEFAULT_REFERENCE = os.environ.get(
    "AMD_GROUP_REFERENCE_CSV",
    str(SCRIPT_DIR / "reference" / "01_Grouping_Summary_All34_reference.csv"),
)
DEFAULT_OUTPUT_DIR = os.environ.get("AMD_RR_OUTPUT_DIR", "output/rr_cox_m2")

COX_PENALIZER = 0.0
MIN_GROUP_SIZE = 30
MIN_EVENTS = 10

M1_VARS = [
    "age_z",
    "sex_binary",
    "Energy_Total_z",
]

M2_VARS_ADD = [
    "tdi_normal_z",
    "EduYears",
    "smoker_current",
    "smoker_former",
    "PA_METs",
    "alcohol_frequent",
    "AMD_PRS_z",
]


def fg_num(label: str) -> int:
    match = re.search(r"(\d+)", str(label))
    return int(match.group(1)) if match else 999


def fmt_num(x: float) -> str:
    if pd.isna(x):
        return "NA"
    x = float(x)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


def fmt_hr(hr: float, lo: float, hi: float) -> str:
    if any(pd.isna(v) for v in [hr, lo, hi]):
        return "NA"
    return f"{float(hr):.2f} ({float(lo):.2f}-{float(hi):.2f})"


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return ""
    if float(p) < 0.001:
        return "<0.001"
    return f"{float(p):.3f}"


def load_food_order(reference_csv: str, fallback_cols: List[str]) -> List[str]:
    if os.path.exists(reference_csv):
        ref = pd.read_csv(reference_csv)
        order = [food for food in ref["Food"].tolist() if food in fallback_cols]
        if order:
            return order
    return fallback_cols


def build_anchor_scores_and_cov(
    summary_df: pd.DataFrame,
    variance_df: pd.DataFrame,
    unique_groups: List[str],
    anchor_ref: str,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    scores: Dict[str, float] = {anchor_ref: 0.0}
    cov: Dict[str, Dict[str, float]] = {}

    for group in unique_groups:
        term = f"G_{group}"
        if group == anchor_ref:
            continue
        scores[group] = float(summary_df.loc[term, "coef"]) if term in summary_df.index else float("nan")

    for g1 in unique_groups:
        cov[g1] = {}
        term1 = f"G_{g1}"
        for g2 in unique_groups:
            term2 = f"G_{g2}"
            if g1 == anchor_ref or g2 == anchor_ref:
                cov[g1][g2] = 0.0
            elif term1 in variance_df.index and term2 in variance_df.columns:
                cov[g1][g2] = float(variance_df.loc[term1, term2])
            else:
                cov[g1][g2] = float("nan")
    return scores, cov


def contrast_from_anchor(
    scores: Dict[str, float],
    cov: Dict[str, Dict[str, float]],
    target_group: str,
    ref_group: str,
) -> Tuple[float, float, float, float]:
    delta = float(scores[target_group] - scores[ref_group])
    var_t = float(cov[target_group][target_group])
    var_r = float(cov[ref_group][ref_group])
    cov_tr = float(cov[target_group][ref_group])
    var_delta = var_t + var_r - 2.0 * cov_tr

    if np.isnan(delta) or np.isnan(var_delta) or var_delta < 0:
        return np.nan, np.nan, np.nan, np.nan

    se = math.sqrt(max(var_delta, 0.0))
    hr = math.exp(delta)
    ci_l = math.exp(delta - 1.96 * se)
    ci_u = math.exp(delta + 1.96 * se)
    p = 1.0 if se == 0 else math.erfc(abs(delta / se) / math.sqrt(2.0))
    return hr, ci_l, ci_u, p


def select_nadir(scores: Dict[str, float], unique_groups: List[str], fallback: str) -> str:
    valid = {g: scores.get(g) for g in unique_groups if not np.isnan(scores.get(g, np.nan))}
    return min(valid, key=valid.get) if valid else fallback


def standardize_event_column(df: pd.DataFrame, event_col: str) -> pd.DataFrame:
    if event_col not in df.columns:
        raise KeyError(f"Requested event_col={event_col!r} is missing from the merged analytic table.")
    out = df.copy()
    out["event_amd"] = pd.to_numeric(out[event_col], errors="coerce")
    return out


def drop_covariates(covariates: List[str], drop_covariates_arg: str) -> List[str]:
    drop_set = {item.strip() for item in drop_covariates_arg.split(",") if item.strip()}
    if not drop_set:
        return covariates
    return [cov for cov in covariates if cov not in drop_set]


def assign_consumer_quantile_groups(series: pd.Series, max_groups: int = 4) -> pd.Series:
    """Assign F1..Fk dose groups among non-zero consumers only."""
    x = pd.to_numeric(series, errors="coerce")
    labels = pd.Series(pd.NA, index=series.index, dtype="object")
    positive = x[np.isfinite(x) & (x > 0)]
    if positive.empty:
        return labels

    unique_vals = np.sort(positive.unique())
    if len(unique_vals) <= 1:
        labels.loc[positive.index] = "F1"
        return labels

    q = max(1, min(int(max_groups), int(len(positive)), int(len(unique_vals))))
    try:
        codes = pd.qcut(positive, q=q, labels=False, duplicates="drop")
    except ValueError:
        try:
            codes = pd.cut(positive, bins=q, labels=False, include_lowest=True, duplicates="drop")
        except ValueError:
            labels.loc[positive.index] = "F1"
            return labels

    codes = pd.Series(codes, index=positive.index)
    valid_codes = sorted(pd.unique(codes.dropna()))
    if not valid_codes:
        labels.loc[positive.index] = "F1"
        return labels

    code_map = {code: f"F{i + 1}" for i, code in enumerate(valid_codes)}
    labels.loc[positive.index] = codes.map(code_map).astype("object")
    return labels


def overall_group_p(fit_df_reduced: pd.DataFrame, fit_df_full: pd.DataFrame) -> float:
    if chi2 is None:
        return np.nan
    try:
        reduced = CoxPHFitter(penalizer=COX_PENALIZER)
        full = CoxPHFitter(penalizer=COX_PENALIZER)
        reduced.fit(fit_df_reduced, duration_col="duration", event_col="event_amd")
        full.fit(fit_df_full, duration_col="duration", event_col="event_amd")
        lr = 2.0 * float(full.log_likelihood_ - reduced.log_likelihood_)
        df = max(int(fit_df_full.shape[1] - fit_df_reduced.shape[1]), 1)
        return float(chi2.sf(lr, df))
    except Exception:
        return np.nan


def is_binary_like(series: pd.Series) -> bool:
    vals = pd.to_numeric(series.dropna(), errors="coerce")
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return False
    uniq = set(np.unique(vals).tolist())
    return uniq.issubset({0.0, 1.0})


def simple_impute_covariates(df: pd.DataFrame, covars: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    rows: list[dict] = []

    for col in covars:
        if col not in out.columns:
            continue
        missing_before = int(out[col].isna().sum())
        strategy = "none"
        fill_value = np.nan

        if missing_before > 0:
            if is_binary_like(out[col]):
                mode = out[col].mode(dropna=True)
                fill_value = float(mode.iloc[0]) if not mode.empty else 0.0
                strategy = "mode_binary"
            else:
                fill_value = float(pd.to_numeric(out[col], errors="coerce").median())
                strategy = "median_numeric"
            out[col] = out[col].fillna(fill_value)

        rows.append(
            {
                "Variable": col,
                "Missing_Before": missing_before,
                "Missing_After": int(out[col].isna().sum()),
                "Impute_Strategy": strategy,
                "Impute_Value": fill_value,
            }
        )

    return out, pd.DataFrame(rows)


def process_food(
    df: pd.DataFrame, food: str, final_m2: List[str], grouping_policy: str
) -> Tuple[List[dict], List[dict], dict | None, dict | None]:
    group_col = f"{food}__group"
    if food not in df.columns:
        return [], [], None, None
    if grouping_policy == "with0" and group_col not in df.columns:
        return [], [], None, None

    needed = ["eid", "duration", "event_amd", food] + final_m2
    if grouping_policy == "with0":
        needed.append(group_col)
    sub = df[needed].copy()

    if grouping_policy == "with0":
        sub[group_col] = sub[group_col].astype("string")
        sub[group_col] = sub[group_col].mask(sub[group_col].isin(["nan", "None", "<NA>"]), pd.NA)
        sub = sub.dropna(subset=["duration", "event_amd", food, group_col] + final_m2).copy()
        sub["FoodGroup"] = sub[group_col].astype(str)
        grouping_mode = "with0_aligned_external_groups"
        excluded_zero_n = 0
    elif grouping_policy == "consumers_quantile":
        sub[food] = pd.to_numeric(sub[food], errors="coerce")
        zero_or_nonconsumer = sub[food].isna() | ~np.isfinite(sub[food]) | (sub[food] <= 0)
        excluded_zero_n = int(zero_or_nonconsumer.sum())
        sub = sub.loc[~zero_or_nonconsumer].dropna(subset=["duration", "event_amd", food] + final_m2).copy()
        if len(sub) < MIN_GROUP_SIZE or int(sub["event_amd"].sum()) < MIN_EVENTS:
            return [], [], None, None
        sub["FoodGroup"] = assign_consumer_quantile_groups(sub[food])
        sub = sub.dropna(subset=["FoodGroup"]).copy()
        grouping_mode = "consumers_only_nonzero_quantiles"
    else:
        raise ValueError(f"Unsupported grouping_policy={grouping_policy!r}")

    if len(sub) < MIN_GROUP_SIZE or int(sub["event_amd"].sum()) < MIN_EVENTS:
        return [], [], None, None

    unique_groups = sorted(sub["FoodGroup"].dropna().unique().tolist(), key=fg_num)
    if len(unique_groups) <= 1:
        return [], [], None, None

    group_stats: Dict[str, dict] = {}
    for group in unique_groups:
        g_df = sub[sub["FoodGroup"] == group]
        group_stats[group] = {
            "n": int(len(g_df)),
            "events": int(g_df["event_amd"].sum()),
            "median": float(g_df[food].median()),
            "min": float(g_df[food].min()),
            "max": float(g_df[food].max()),
        }

    base_group = "F1" if "F1" in unique_groups else unique_groups[0]
    dummies_base = pd.get_dummies(sub["FoodGroup"], prefix="G")
    ref_col = f"G_{base_group}"
    dummy_cols_base = [c for c in dummies_base.columns if c != ref_col and not c.endswith("_nan")]
    if not dummy_cols_base:
        return [], [], None, None

    fit_df_base_full = pd.concat([sub[["duration", "event_amd"] + final_m2], dummies_base[dummy_cols_base]], axis=1).dropna()
    fit_df_base_reduced = sub[["duration", "event_amd"] + final_m2].dropna()
    if len(fit_df_base_full) < MIN_GROUP_SIZE or int(fit_df_base_full["event_amd"].sum()) < MIN_EVENTS:
        return [], [], None, None

    base_cph = CoxPHFitter(penalizer=COX_PENALIZER)
    base_cph.fit(fit_df_base_full, duration_col="duration", event_col="event_amd")
    scores, cov = build_anchor_scores_and_cov(base_cph.summary, base_cph.variance_matrix_, unique_groups, base_group)
    nadir = select_nadir(scores, unique_groups, base_group)
    overall_p = overall_group_p(fit_df_base_reduced, fit_df_base_full)

    rr_rows: List[dict] = []
    for ref_group in unique_groups:
        dummies = pd.get_dummies(sub["FoodGroup"], prefix="G")
        ref_col = f"G_{ref_group}"
        dummy_cols = [c for c in dummies.columns if c != ref_col and not c.endswith("_nan")]
        if not dummy_cols:
            continue

        fit_df = pd.concat([sub[["duration", "event_amd"] + final_m2], dummies[dummy_cols]], axis=1).dropna()
        if len(fit_df) < MIN_GROUP_SIZE or int(fit_df["event_amd"].sum()) < MIN_EVENTS:
            continue

        cph = CoxPHFitter(penalizer=COX_PENALIZER)
        cph.fit(fit_df, duration_col="duration", event_col="event_amd")
        summary = cph.summary

        for target_group in unique_groups:
            if ref_group == target_group:
                hr, lo, hi, p = 1.0, 1.0, 1.0, 1.0
            else:
                term = f"G_{target_group}"
                if term in summary.index:
                    row = summary.loc[term]
                    hr = float(row["exp(coef)"])
                    lo = float(row["exp(coef) lower 95%"])
                    hi = float(row["exp(coef) upper 95%"])
                    p = float(row["p"])
                else:
                    hr, lo, hi, p = np.nan, np.nan, np.nan, np.nan

            st = group_stats[target_group]
            rr_rows.append(
                {
                    "Food": food,
                    "Genetic_Group": "All",
                    "Reference_Food": ref_group,
                    "Target_Food": target_group,
                    "M2_HR": hr,
                    "M2_CI_Lower": lo,
                    "M2_CI_Upper": hi,
                    "M2_P": p,
                    "Model_N": int(len(fit_df)),
                    "Model_Events": int(fit_df["event_amd"].sum()),
                    "Group_N": st["n"],
                    "Group_Events": st["events"],
                    "Events_Total": f"{st['events']:,}/{st['n']:,}",
                    "Median_Intake": st["median"],
                    "Lower_Bound": st["min"],
                    "Upper_Bound": st["max"],
                }
            )

    table_rows: List[dict] = []
    min_hr_vs_nadir = np.inf
    bad_count = 0
    for group in unique_groups:
        st = group_stats[group]
        if group == nadir:
            hr, lo, hi, p = np.nan, np.nan, np.nan, np.nan
            hr_disp = "Ref."
            p_disp = "/"
            is_ref = True
        else:
            hr, lo, hi, p = contrast_from_anchor(scores, cov, target_group=group, ref_group=nadir)
            hr_disp = fmt_hr(hr, lo, hi)
            p_disp = fmt_p(p)
            is_ref = False
            if pd.notna(hr):
                min_hr_vs_nadir = min(min_hr_vs_nadir, float(hr))
                if float(hr) < 0.999999:
                    bad_count += 1

        table_rows.append(
            {
                "Food": food,
                "Genetic_Group": "All",
                "Grade": fg_num(group),
                "Original_FoodGroup": group,
                "Nadir_Group": nadir,
                "Anchor_Group": base_group,
                "N": st["n"],
                "Events": st["events"],
                "Events_Total": f"{st['events']:,}/{st['n']:,}",
                "Median_Intake": st["median"],
                "Lower_Bound": st["min"],
                "Upper_Bound": st["max"],
                "Range": f"{fmt_num(st['median'])} ({fmt_num(st['min'])}~{fmt_num(st['max'])})",
                "Incidence_Rate": (st["events"] / st["n"]) if st["n"] else np.nan,
                "M2_HR": hr_disp,
                "M2_P": p_disp,
                "M2_HR_num": hr,
                "M2_CI_Lower_num": lo,
                "M2_CI_Upper_num": hi,
                "M2_P_num": p,
                "Overall_P_M2": overall_p,
                "Overall_P_M2_Disp": fmt_p(overall_p),
                "Grouping_Policy": grouping_policy,
                "Grouping_Mode": grouping_mode,
                "Excluded_Zero_N_M2": excluded_zero_n,
                "Is_Reference": is_ref,
            }
        )

    summary_row = {
        "Food": food,
        "Genetic_Group": "All",
        "Nadir_Group": nadir,
        "Anchor_Group": base_group,
        "Overall_P_M2": overall_p,
        "N_Total": int(len(sub)),
        "Events_Total": int(sub["event_amd"].sum()),
        "Grouping_Policy": grouping_policy,
        "Grouping_Mode": grouping_mode,
        "Excluded_Zero_N_M2": excluded_zero_n,
    }
    selfcheck_row = {
        "Food": food,
        "Nadir_Group": nadir,
        "Bad_HR_Count": int(bad_count),
        "Min_HR_Against_Nadir": float(min_hr_vs_nadir if min_hr_vs_nadir != np.inf else 1.0),
        "Is_Perfect_Reference": int(bad_count == 0),
    }
    return rr_rows, table_rows, summary_row, selfcheck_row


def main() -> int:
    parser = argparse.ArgumentParser(description="Pairwise round-robin Cox analysis on the training subset (M2)")
    parser.add_argument("--train_csv", default=DEFAULT_TRAIN)
    parser.add_argument("--groupings_csv", default=DEFAULT_GROUPINGS)
    parser.add_argument("--reference_csv", default=DEFAULT_REFERENCE)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--grouping_policy", choices=["with0", "consumers_quantile"], default="with0")
    parser.add_argument("--covariate_policy", choices=["complete_case", "simple_impute"], default="complete_case")
    parser.add_argument("--event_col", default="event_amd")
    parser.add_argument("--drop_covariates", default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train = pd.read_csv(args.train_csv, low_memory=False)
    if "duration" not in train.columns:
        if "follow_up_years" in train.columns:
            train["duration"] = train["follow_up_years"]
        elif {"age_at_exit", "age_at_index"} <= set(train.columns):
            train["duration"] = train["age_at_exit"] - train["age_at_index"]

    cohort = train[["eid", args.event_col, "duration"]].copy()
    cov_cols = [c for c in M1_VARS + M2_VARS_ADD if c in train.columns]
    covars = train[["eid", *cov_cols]]
    diet = train.loc[:, ["eid", *LOCKED_34_FOODS]]
    validate_locked_34_food_columns(
        diet.columns,
        context="RR Cox diet matrix",
        allow_eid=True,
        require_order=False,
    )
    groups = pd.read_csv(args.groupings_csv, low_memory=False)
    energy = train[["eid", "Energy_Total_z"]] if "Energy_Total_z" in train.columns else train[["eid"]]

    df = cohort.merge(covars, on="eid", how="left", suffixes=("", "_cov"))
    df = df.merge(energy, on="eid", how="left")
    df = df.merge(diet, on="eid", how="left")
    df = df.merge(groups, on="eid", how="left")
    df = standardize_event_column(df, args.event_col)

    final_m1 = [c for c in M1_VARS if c in df.columns]
    final_m2 = final_m1 + [c for c in M2_VARS_ADD if c in df.columns]
    final_m2 = drop_covariates(final_m2, args.drop_covariates)

    if args.covariate_policy == "simple_impute":
        df, impute_summary = simple_impute_covariates(df, final_m2)
        impute_summary.to_csv(os.path.join(args.output_dir, "Covariate_Imputation_Summary.csv"), index=False)

    food_cols = list(LOCKED_34_FOODS)
    food_order = load_food_order(args.reference_csv, food_cols)

    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(process_food)(df, food, final_m2, args.grouping_policy) for food in food_order
    )

    rr_all: List[dict] = []
    table_all: List[dict] = []
    summary_all: List[dict] = []
    selfcheck_all: List[dict] = []
    for rr_rows, table_rows, summary_row, selfcheck_row in results:
        rr_all.extend(rr_rows)
        table_all.extend(table_rows)
        if summary_row is not None:
            summary_all.append(summary_row)
        if selfcheck_row is not None:
            selfcheck_all.append(selfcheck_row)

    rr_df = pd.DataFrame(rr_all)
    table_df = pd.DataFrame(table_all)
    summary_df = pd.DataFrame(summary_all)
    selfcheck_df = pd.DataFrame(selfcheck_all)

    food_rank = {food: i for i, food in enumerate(food_order)}
    if not rr_df.empty:
        rr_df["Food_Order"] = rr_df["Food"].map(food_rank)
        rr_df = rr_df.sort_values(["Food_Order", "Reference_Food", "Target_Food"]).drop(columns=["Food_Order"])
    if not table_df.empty:
        table_df["Food_Order"] = table_df["Food"].map(food_rank)
        table_df = table_df.sort_values(["Food_Order", "Grade"]).drop(columns=["Food_Order"])
    if not summary_df.empty:
        summary_df["Food_Order"] = summary_df["Food"].map(food_rank)
        summary_df = summary_df.sort_values(["Food_Order"]).drop(columns=["Food_Order"])
    if not selfcheck_df.empty:
        selfcheck_df["Food_Order"] = selfcheck_df["Food"].map(food_rank)
        selfcheck_df = selfcheck_df.sort_values(["Food_Order"]).drop(columns=["Food_Order"])

    rr_df.to_csv(os.path.join(args.output_dir, "RoundRobin_M2_All.csv"), index=False)
    table_df.to_csv(os.path.join(args.output_dir, "Diet_Table3_M2_All.csv"), index=False)
    table_df.to_csv(os.path.join(args.output_dir, "Diet_Table3_Fig2Input_All.csv"), index=False)
    summary_df.to_csv(os.path.join(args.output_dir, "FoodLevel_Summary_M2_All.csv"), index=False)
    selfcheck_df.to_csv(os.path.join(args.output_dir, "FoodLevel_SelfCheck_M2_All.csv"), index=False)

    print(f"foods={table_df['Food'].nunique() if not table_df.empty else 0}")
    print(f"rows={len(table_df)}")
    if not selfcheck_df.empty:
        print(f"perfect={int(selfcheck_df['Is_Perfect_Reference'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
