import os
import pandas as pd

INPUT_DIR = "input"
OUTPUT_DIR = "output"

AMED_CSV = os.path.join(INPUT_DIR, "PersonLevel_aMED_CorrectedScores_20260427.csv")
INDEX_CSV = os.path.join(OUTPUT_DIR, "index_version_panda_amed.csv")
INDEX_FULL_CSV = os.path.join(OUTPUT_DIR, "index_panda_amed_ms_full.csv")
INDEX_FINAL_CSV = os.path.join(OUTPUT_DIR, "index_panda_amed_final.csv")

FILES = [
    "analysis_master_dataset.csv",
    "train_A_random.csv",
    "test_A_random.csv",
    "train_B_centre.csv",
    "test_B_centre.csv",
    "train_C_time.csv",
    "test_C_time.csv",
]


def load_amed():
    amed = pd.read_csv(AMED_CSV, usecols=["eid", "Mediterranean_score_corrected9"])
    return amed.rename(columns={"Mediterranean_score_corrected9": "Mediterranean_score_new"})


def index_version_summary(amed):
    """Merge aMED into the index-version dataset and tabulate quartiles."""
    idx = pd.read_csv(INDEX_CSV, usecols=["eid", "event_amd", "duration"])
    df = idx.merge(amed, on="eid", how="inner")
    df["q"] = pd.cut(df.Mediterranean_score_new, bins=[-0.1, 0.5, 3.5, 6.5, 9.5], labels=["Q1", "Q2", "Q3", "Q4"])
    print(f"index version: N={len(df)}, events={df.event_amd.sum()}")
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        s = df[df.q == q]
        print(f"{q}: N={len(s)}, events={s.event_amd.sum()}")

    full = pd.read_csv(INDEX_FULL_CSV, usecols=["eid", "event_amd", "duration", "panda_ms"]).merge(amed, on="eid", how="inner")
    full = full.rename(columns={"panda_ms": "PANDA_score", "Mediterranean_score_new": "Mediterranean_score"})
    full["PANDA_score_pct"] = full["PANDA_score"].rank(pct=True) * 100
    full["Mediterranean_score_pct"] = full["Mediterranean_score"].rank(pct=True) * 100
    full.to_csv(INDEX_FINAL_CSV, index=False)
    print(f"index full saved: n={len(full)}")


def update_analysis_files(amed):
    for fname in FILES:
        fpath = os.path.join(INPUT_DIR, fname)
        if not os.path.exists(fpath):
            print(f"{fname}: not found, skipped")
            continue
        df = pd.read_csv(fpath)
        if "eid" not in df.columns:
            print(f"{fname}: no eid column, skipped")
            continue
        df = df.drop(columns=["Mediterranean_score", "Mediterranean_score_pct"], errors="ignore")
        df = df.merge(amed, on="eid", how="left").rename(columns={"Mediterranean_score_new": "Mediterranean_score"})
        df["Mediterranean_score_pct"] = df["Mediterranean_score"].rank(pct=True) * 100
        if "aMED_z" in df.columns:
            df["aMED_z"] = (df["Mediterranean_score"] - df["Mediterranean_score"].mean()) / df["Mediterranean_score"].std()
        df.to_csv(fpath, index=False)
        print(f"{fname}: n={len(df)}, aMED missing={df['Mediterranean_score'].isna().sum()}")


def main():
    amed = load_amed()
    print(f"aMED corrected: n={len(amed)}")
    index_version_summary(amed)
    update_analysis_files(amed)


if __name__ == "__main__":
    main()
