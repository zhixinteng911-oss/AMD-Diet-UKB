#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from lightgbm import LGBMClassifier
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

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


FOOD_ORDER = list(LOCKED_34_FOODS)

M2_COVS = [
    "Energy_Total_z",
    "age_z",
    "sex_binary",
    "tdi_normal_z",
    "EduYears",
    "smoker_current",
    "smoker_former",
    "PA_METs",
    "alcohol_frequent",
    "bmi_log_z",
    "crp_log_z",
    "hdl_z",
    "hypertension_prevalent",
    "diabetes_prevalent",
    "ckd_status",
    "hyperlipidemia_status",
    "AMD_PRS_z",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def detect_foods(columns: list[str]) -> list[str]:
    return [food for food in FOOD_ORDER if food in columns]


def resolve_candidate_foods(available_foods: list[str], candidate_foods_file: str) -> list[str]:
    if not candidate_foods_file:
        return available_foods
    requested = []
    for line in Path(candidate_foods_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            requested.append(line)
    requested = list(dict.fromkeys(requested))
    missing = [food for food in requested if food not in set(available_foods)]
    if missing:
        raise KeyError(f"Requested candidate foods missing from wide table: {missing}")
    return [food for food in available_foods if food in set(requested)]


def drop_covariates(covariates: list[str], drop_covariates_arg: str) -> list[str]:
    drop_set = {item.strip() for item in drop_covariates_arg.split(",") if item.strip()}
    if not drop_set:
        return covariates
    return [cov for cov in covariates if cov not in drop_set]


def summarize_tie_breaker(table3_path: str | None, foods: list[str]) -> dict[str, float]:
    tie = {food: 1.0 for food in foods}
    if not table3_path or not Path(table3_path).exists():
        return tie
    table = pd.read_csv(table3_path)
    if "Food" not in table.columns:
        return tie
    if "M2_P_num" in table.columns:
        min_p = (
            table.groupby("Food", as_index=False)["M2_P_num"]
            .min()
            .set_index("Food")["M2_P_num"]
            .to_dict()
        )
        tie.update({k: float(v) for k, v in min_p.items() if pd.notna(v)})
    elif "Overall_P_M2" in table.columns:
        overall = (
            table.groupby("Food", as_index=False)["Overall_P_M2"]
            .min()
            .set_index("Food")["Overall_P_M2"]
            .to_dict()
        )
        tie.update({k: float(v) for k, v in overall.items() if pd.notna(v)})
    return tie


def residualize_foods(
    df: pd.DataFrame,
    foods: list[str],
    covariates: list[str],
    event_col: str = "event_amd",
    min_rows: int = 500,
) -> pd.DataFrame:
    return residualize_foods_with_event(df, foods, covariates, event_col=event_col, min_rows=min_rows)


def residualize_foods_with_event(
    df: pd.DataFrame,
    foods: list[str],
    covariates: list[str],
    event_col: str = "event_amd",
    min_rows: int = 500,
) -> pd.DataFrame:
    if event_col not in df.columns:
        raise KeyError(f"Requested event_col={event_col!r} is missing from the wide table.")
    rhs = " + ".join(covariates)
    event_values = pd.to_numeric(df[event_col], errors="coerce")
    residuals: dict[str, np.ndarray] = {
        "eid": df["eid"].to_numpy(),
        "event_amd": event_values.to_numpy(),
    }
    for food in foods:
        subset = df[["eid", event_col, food] + covariates].copy()
        subset[event_col] = pd.to_numeric(subset[event_col], errors="coerce")
        subset = subset.dropna()
        if len(subset) < min_rows:
            continue
        subset["log_food"] = np.log1p(subset[food].clip(lower=0))
        model = smf.ols(f"log_food ~ {rhs}", data=subset).fit()
        aligned = np.full(len(df), np.nan, dtype=float)
        loc = df.index.get_indexer(subset.index)
        aligned[loc] = model.resid.to_numpy() + float(subset["log_food"].mean())
        residuals[food] = aligned
    resid_df = pd.DataFrame(residuals)
    keep_cols = ["eid", "event_amd"] + [
        food for food in foods if food in resid_df.columns and resid_df[food].notna().sum() > 0
    ]
    resid_df = resid_df[keep_cols].dropna().reset_index(drop=True)
    resid_df["event_amd"] = resid_df["event_amd"].astype(int)
    return resid_df


def build_original_log1p_matrix(
    wide: pd.DataFrame,
    residuals_df: pd.DataFrame,
    foods: list[str],
    event_col: str,
    covariates: list[str] | None = None,
    transform: str = "log1p",
) -> pd.DataFrame:
    """Build original food features aligned to the residualized complete-case rows."""
    covariates = covariates or []
    needed = ["eid", event_col] + covariates + foods
    missing = [col for col in needed if col not in wide.columns]
    if missing:
        raise KeyError(f"Wide table missing original SFS feature columns: {missing}")

    base = residuals_df[["eid", "event_amd"]].merge(
        wide[needed],
        on="eid",
        how="left",
        validate="1:1",
    )
    out = pd.DataFrame({"eid": base["eid"], "event_amd": base["event_amd"].astype(int)})
    for cov in covariates:
        out[cov] = pd.to_numeric(base[cov], errors="coerce")
    for food in foods:
        values = pd.to_numeric(base[food], errors="coerce").clip(lower=0)
        if transform == "log1p":
            out[food] = np.log1p(values)
        elif transform == "raw":
            out[food] = values
        else:
            raise ValueError(f"Unsupported original food transform: {transform!r}")
    return out.dropna().reset_index(drop=True)


def cluster_representatives(
    residuals_df: pd.DataFrame,
    foods: list[str],
    tie_breaker: dict[str, float],
    target_clusters: int,
) -> tuple[list[str], pd.DataFrame]:
    corr = residuals_df[foods].corr(method="spearman")
    dist = np.nan_to_num(1 - np.abs(corr))
    linkage = hierarchy.linkage(squareform(dist, checks=False), method="average")
    clusters = hierarchy.fcluster(linkage, t=target_clusters, criterion="maxclust")
    cluster_rows = []
    representatives: list[str] = []
    for cluster_id in sorted(set(clusters)):
        members = [foods[i] for i, c in enumerate(clusters) if c == cluster_id]
        rep = min(members, key=lambda x: (tie_breaker.get(x, 1.0), x))
        representatives.append(rep)
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "representative": rep,
                "members": "|".join(sorted(members)),
                "n_members": len(members),
                "tie_breaker_p": tie_breaker.get(rep, np.nan),
            }
        )
    cluster_df = pd.DataFrame(cluster_rows).sort_values(["cluster_id", "representative"])
    representatives = [food for food in foods if food in set(representatives)]
    return representatives, cluster_df


def build_lgbm(cpu_threads: int, lgbm_params: dict[str, float | int]) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        is_unbalance=True,
        n_estimators=int(lgbm_params["n_estimators"]),
        learning_rate=float(lgbm_params["learning_rate"]),
        num_leaves=int(lgbm_params["num_leaves"]),
        max_depth=int(lgbm_params["max_depth"]),
        subsample=float(lgbm_params["subsample"]),
        colsample_bytree=float(lgbm_params["colsample_bytree"]),
        min_child_samples=int(lgbm_params["min_child_samples"]),
        reg_lambda=float(lgbm_params["reg_lambda"]),
        verbosity=-1,
        random_state=42,
        n_jobs=int(cpu_threads),
    )


def normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    total = float(np.nansum(arr))
    if not np.isfinite(total) or total <= 0:
        return np.zeros_like(arr, dtype=float)
    return arr / total


def build_fold_splits(y: pd.Series, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(splitter.split(np.zeros(len(y)), y))


def compute_lgbm_importance(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    cpu_threads: int,
    lgbm_params: dict[str, float | int],
    fixed_features: list[str] | None = None,
) -> pd.DataFrame:
    fixed_features = fixed_features or []
    model_features = fixed_features + feature_names
    fold_series = []
    for fold_id, (tr_idx, _) in enumerate(splits, start=1):
        estimator = build_lgbm(cpu_threads=cpu_threads, lgbm_params=lgbm_params)
        estimator.fit(X.iloc[tr_idx][model_features], y.iloc[tr_idx])
        imp = estimator.booster_.feature_importance(importance_type="gain")
        imp_series = pd.Series(imp, index=model_features, name=fold_id)
        fold_series.append(pd.Series(normalize(imp_series.reindex(feature_names).to_numpy()), index=feature_names, name=fold_id))
    imp_df = pd.DataFrame(fold_series).fillna(0.0)
    out = pd.DataFrame({"Food": feature_names, "Importance": imp_df.mean(axis=0).reindex(feature_names).to_numpy()})
    return out.sort_values("Importance", ascending=False).reset_index(drop=True)


def cross_validated_loss(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    cpu_threads: int,
    lgbm_params: dict[str, float | int],
) -> tuple[float, float]:
    fold_losses = []
    for tr_idx, va_idx in splits:
        estimator = build_lgbm(cpu_threads=cpu_threads, lgbm_params=lgbm_params)
        estimator.fit(X.iloc[tr_idx][features], y.iloc[tr_idx])
        prob = estimator.predict_proba(X.iloc[va_idx][features])[:, 1]
        fold_losses.append(log_loss(y.iloc[va_idx], prob))
    return float(np.mean(fold_losses)), float(np.std(fold_losses, ddof=0))


def sequential_lgbm_loss(
    X: pd.DataFrame,
    y: pd.Series,
    ranking: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    cpu_threads: int,
    lgbm_params: dict[str, float | int],
    stop_threshold: float,
    stop_mode: str,
    fixed_features: list[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    rows = []
    previous_loss = None
    fixed_features = fixed_features or []
    for rank, food in enumerate(ranking, start=1):
        features = fixed_features + ranking[:rank]
        loss_mean, loss_sd = cross_validated_loss(
            X=X,
            y=y,
            features=features,
            splits=splits,
            cpu_threads=cpu_threads,
            lgbm_params=lgbm_params,
        )
        delta = np.nan if previous_loss is None else float(previous_loss - loss_mean)
        rows.append(
            {
                "Ranking": rank,
                "Food groups": food,
                "Loss": loss_mean,
                "Loss S.D.": loss_sd,
                "Delta loss": delta,
            }
        )
        previous_loss = loss_mean
    curve = pd.DataFrame(rows)
    first_below = curve.loc[curve["Delta loss"].fillna(stop_threshold) < stop_threshold, "Ranking"]
    if len(first_below):
        boundary = int(first_below.iloc[0])
        if stop_mode == "one_after_first_below":
            best_k = min(len(curve), boundary)
        else:
            best_k = max(1, boundary - 1)
    else:
        best_k = len(curve)
    curve["Select"] = (curve["Ranking"] <= best_k).astype(int)
    return curve, best_k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--table3_tiebreaker_csv", default="")
    parser.add_argument("--candidate_foods_file", required=False, default="",
                        help="Path to candidate food list (one per line); empty = use all available foods")
    parser.add_argument("--target_clusters", type=int, default=22)
    parser.add_argument("--cv_folds", type=int, default=10)
    parser.add_argument("--stop_threshold", type=float, default=0.005)
    parser.add_argument("--stop_mode", choices=["first_below", "one_after_first_below"], default="first_below")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--max_depth", type=int, default=-1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--min_child_samples", type=int, default=20)
    parser.add_argument("--reg_lambda", type=float, default=0.0)
    parser.add_argument("--cpu_threads", type=int, default=16)
    parser.add_argument("--event_col", default="event_amd")
    parser.add_argument("--drop_covariates", default="")
    parser.add_argument(
        "--sfs_feature_source",
        choices=["residualized", "original_log1p", "original_raw"],
        default="residualized",
        help=(
            "Feature matrix used for LightGBM importance/SFS. "
            "Residuals are always used for correlation clustering; original_log1p uses "
            "log1p raw foods for the selected representatives after residual clustering."
        ),
    )
    parser.add_argument(
        "--sfs_include_covariates",
        action="store_true",
        help="Always include the M2 covariates as fixed base features in LightGBM importance/SFS.",
    )
    args = parser.parse_args()

    start = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wide = pd.read_csv(args.wide_csv, low_memory=False)
    all_available_foods = detect_foods(list(wide.columns))
    foods = resolve_candidate_foods(all_available_foods, args.candidate_foods_file)
    covs = [cov for cov in M2_COVS if cov in wide.columns]
    covs = drop_covariates(covs, args.drop_covariates)
    log(f"candidate foods: {len(foods)}, rows: {len(wide)}, covariates: {len(covs)}")

    tie_breaker = summarize_tie_breaker(args.table3_tiebreaker_csv, foods)
    residuals = residualize_foods_with_event(wide, foods, covs, event_col=args.event_col)
    residual_foods = [food for food in foods if food in residuals.columns]
    residuals.to_csv(out_dir / "Residualized_FoodMatrix.csv", index=False)
    log(f"residual matrix: rows={len(residuals)}, foods={len(residual_foods)}")

    representatives, cluster_df = cluster_representatives(
        residuals_df=residuals,
        foods=residual_foods,
        tie_breaker=tie_breaker,
        target_clusters=args.target_clusters,
    )
    cluster_df.to_csv(out_dir / "Cluster22_Representatives.csv", index=False)
    log(f"representatives: {len(representatives)}")

    fixed_features: list[str] = []
    if args.sfs_feature_source in {"original_log1p", "original_raw"}:
        transform = "log1p" if args.sfs_feature_source == "original_log1p" else "raw"
        sfs_matrix = build_original_log1p_matrix(
            wide=wide,
            residuals_df=residuals,
            foods=representatives,
            event_col=args.event_col,
            covariates=covs if args.sfs_include_covariates else [],
            transform=transform,
        )
        matrix_name = "OriginalLog1p_SFS_FoodMatrix.csv" if transform == "log1p" else "OriginalRaw_SFS_FoodMatrix.csv"
        sfs_matrix.to_csv(out_dir / matrix_name, index=False)
        X = sfs_matrix[representatives]
        if args.sfs_include_covariates:
            fixed_features = covs
            X = sfs_matrix[fixed_features + representatives]
        y = sfs_matrix["event_amd"].astype(int)
        n_sfs_rows = int(len(sfs_matrix))
    else:
        X = residuals[representatives]
        y = residuals["event_amd"].astype(int)
        n_sfs_rows = int(len(residuals))
    splits = build_fold_splits(y, n_splits=args.cv_folds)
    lgbm_params = {
        "n_estimators": int(args.n_estimators),
        "learning_rate": float(args.learning_rate),
        "num_leaves": int(args.num_leaves),
        "max_depth": int(args.max_depth),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "min_child_samples": int(args.min_child_samples),
        "reg_lambda": float(args.reg_lambda),
    }

    lgbm_importance = compute_lgbm_importance(
        X=X,
        y=y,
        feature_names=representatives,
        splits=splits,
        cpu_threads=args.cpu_threads,
        lgbm_params=lgbm_params,
        fixed_features=fixed_features,
    )
    lgbm_importance.to_csv(out_dir / "LightGBM_Importance.csv", index=False)
    log("LightGBM importance done")

    lgbm_ranking = lgbm_importance["Food"].tolist()
    lgbm_curve, best_k = sequential_lgbm_loss(
        X=X,
        y=y,
        ranking=lgbm_ranking,
        splits=splits,
        cpu_threads=args.cpu_threads,
        lgbm_params=lgbm_params,
        stop_threshold=args.stop_threshold,
        stop_mode=args.stop_mode,
        fixed_features=fixed_features,
    )
    lgbm_curve.to_csv(out_dir / "LightGBM_SFS_LossCurve.csv", index=False)
    selected_lgbm = lgbm_ranking[:best_k]
    pd.DataFrame({"Food": selected_lgbm, "Rank": range(1, len(selected_lgbm) + 1)}).to_csv(
        out_dir / "LightGBM_Selected_ModelBased.csv",
        index=False,
    )
    log(f"best_k: {best_k}")

    rows = []
    max_len = len(lgbm_curve)
    for i in range(max_len):
        rows.append(
            {
                "Ranking": i + 1,
                "LightGBM__Food groups": lgbm_curve.iloc[i]["Food groups"],
                "LightGBM__Importance": lgbm_importance.iloc[i]["Importance"] if i < len(lgbm_importance) else np.nan,
                "LightGBM__Loss": lgbm_curve.iloc[i]["Loss"],
                "LightGBM__Loss S.D.": lgbm_curve.iloc[i]["Loss S.D."],
                "LightGBM__Delta loss": lgbm_curve.iloc[i]["Delta loss"],
                "LightGBM__Select": lgbm_curve.iloc[i]["Select"],
            }
        )
    table6 = pd.DataFrame(rows)
    table6.to_csv(out_dir / "Table6_ML_Selection.csv", index=False)

    metadata = {
        "wide_csv": args.wide_csv,
        "event_col": args.event_col,
        "output_dir": str(out_dir),
        "candidate_foods": foods,
        "n_rows": int(len(wide)),
        "n_foods": int(len(foods)),
        "n_covariates": int(len(covs)),
        "n_residual_rows": int(len(residuals)),
        "n_sfs_rows": n_sfs_rows,
        "n_representatives": int(len(representatives)),
        "representatives": representatives,
        "sfs_feature_source": args.sfs_feature_source,
        "sfs_include_covariates": bool(args.sfs_include_covariates),
        "sfs_fixed_covariates": fixed_features,
        "best_k": int(best_k),
        "selected_lgbm": selected_lgbm,
        "cv_folds": int(args.cv_folds),
        "stop_threshold": float(args.stop_threshold),
        "stop_mode": str(args.stop_mode),
        "lgbm_params": lgbm_params,
        "runtime_seconds": round(time.time() - start, 2),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"done in {metadata['runtime_seconds']}s, best_k={best_k}, output={out_dir}")


if __name__ == "__main__":
    main()
