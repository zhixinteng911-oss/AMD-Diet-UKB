import os

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import warnings

warnings.filterwarnings("ignore")

BASE = "."
os.makedirs(f"{BASE}/output", exist_ok=True)
SEED = 42
B = 1000

COVARS = ["sex_binary", "Energy_Total_z", "tdi_normal_z", "EduYears",
          "smoker_current", "smoker_former", "PA_METs", "alcohol_frequent", "AMD_PRS_z"]


def fit_cox(df, exposure_col, covars):
    """Cox model for one exposure; returns (HR, CI_lo, CI_hi, P, N)."""
    cols = ["age_at_index", "age_at_exit", "event_amd", exposure_col] + covars
    sub = df[cols].dropna()
    cph = CoxPHFitter()
    cph.fit(sub, duration_col="age_at_exit", event_col="event_amd",
            entry_col="age_at_index", show_progress=False)
    hr = np.exp(cph.params_[exposure_col])
    ci = np.exp(cph.confidence_intervals_.loc[exposure_col])
    p = cph.summary.loc[exposure_col, "p"]
    return hr, ci.values[0], ci.values[1], p, len(sub)


def calc_cindex(df, exposure_col, covars):
    cols = ["age_at_index", "age_at_exit", "event_amd", exposure_col] + covars
    sub = df[cols].dropna()
    cph = CoxPHFitter()
    cph.fit(sub, duration_col="age_at_exit", event_col="event_amd",
            entry_col="age_at_index", show_progress=False)
    return cph.concordance_index_


def main():
    test = pd.read_csv(f"{BASE}/input/test_A_random.csv")
    print(f"validation set: N={len(test)}, events={test.event_amd.sum()}")

    missing = [c for c in ["age_at_index", "age_at_exit"] if c not in test.columns]
    if missing and "duration" in test.columns:
        test["age_at_index"] = 0
        test["age_at_exit"] = test["duration"]

    test["PANDA_z"] = (test["PANDA_score"] - test["PANDA_score"].mean()) / test["PANDA_score"].std()
    test["aMED_z"] = (test["Mediterranean_score"] - test["Mediterranean_score"].mean()) / test["Mediterranean_score"].std()

    hr_p, lo_p, hi_p, p_p, n_p = fit_cox(test, "PANDA_z", COVARS)
    hr_a, lo_a, hi_a, p_a, n_a = fit_cox(test, "aMED_z", COVARS)
    print(f"PANDA per 1 SD: HR={hr_p:.4f} ({lo_p:.4f}-{hi_p:.4f}), P={p_p:.6f}, N={n_p}")
    print(f"aMED  per 1 SD: HR={hr_a:.4f} ({lo_a:.4f}-{hi_a:.4f}), P={p_a:.6f}, N={n_a}")

    cols_mut = ["age_at_index", "age_at_exit", "event_amd", "PANDA_z", "aMED_z"] + COVARS
    mut = test[cols_mut].dropna()
    cph_mut = CoxPHFitter()
    cph_mut.fit(mut, duration_col="age_at_exit", event_col="event_amd",
                entry_col="age_at_index", show_progress=False)
    hr_pm = np.exp(cph_mut.params_["PANDA_z"])
    ci_pm = np.exp(cph_mut.confidence_intervals_.loc["PANDA_z"])
    p_pm = cph_mut.summary.loc["PANDA_z", "p"]
    hr_am = np.exp(cph_mut.params_["aMED_z"])
    ci_am = np.exp(cph_mut.confidence_intervals_.loc["aMED_z"])
    p_am = cph_mut.summary.loc["aMED_z", "p"]
    print(f"PANDA adj. for aMED: HR={hr_pm:.4f} ({ci_pm.values[0]:.4f}-{ci_pm.values[1]:.4f}), P={p_pm:.6f}")
    print(f"aMED  adj. for PANDA: HR={hr_am:.4f} ({ci_am.values[0]:.4f}-{ci_am.values[1]:.4f}), P={p_am:.6f}")

    rng = np.random.RandomState(SEED)
    deltas = []
    for _ in range(B):
        boot = test.iloc[rng.choice(len(test), size=len(test), replace=True)]
        try:
            hr_boot_p = fit_cox(boot, "PANDA_z", COVARS)[0]
            hr_boot_a = fit_cox(boot, "aMED_z", COVARS)[0]
            deltas.append(np.log(hr_boot_p) - np.log(hr_boot_a))
        except Exception:
            continue

    deltas = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    print(f"bootstrap: {len(deltas)}/{B} succeeded; mean delta-logHR={deltas.mean():.4f}, 95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"HR ratio (PANDA/aMED): {np.exp(deltas.mean()):.4f} ({np.exp(ci_lo):.4f}-{np.exp(ci_hi):.4f})")

    cindex_p = calc_cindex(test, "PANDA_z", COVARS)
    cindex_a = calc_cindex(test, "aMED_z", COVARS)
    print(f"C-index: PANDA={cindex_p:.4f}, aMED={cindex_a:.4f}, diff={cindex_p - cindex_a:.4f}")

    results = {
        "validation_set": "test_A_random",
        "N": len(test),
        "events": int(test.event_amd.sum()),
        "PANDA_HR_perSD": hr_p, "PANDA_CI_lo": lo_p, "PANDA_CI_hi": hi_p, "PANDA_P": p_p,
        "aMED_HR_perSD": hr_a, "aMED_CI_lo": lo_a, "aMED_CI_hi": hi_a, "aMED_P": p_a,
        "PANDA_mutual_HR": hr_pm, "PANDA_mutual_P": p_pm,
        "aMED_mutual_HR": hr_am, "aMED_mutual_P": p_am,
        "delta_logHR_mean": deltas.mean(),
        "delta_logHR_CI_lo": ci_lo, "delta_logHR_CI_hi": ci_hi,
        "bootstrap_B": len(deltas), "seed": SEED,
        "PANDA_Cindex": cindex_p, "aMED_Cindex": cindex_a,
    }
    out_path = f"{BASE}/output/head_to_head_final.csv"
    pd.DataFrame([results]).to_csv(out_path, index=False)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
