import os
import pandas as pd

INPUT_DIR = "input"

FILES = [
    "analysis_master_dataset.csv",
    "train_A_random.csv",
    "test_A_random.csv",
    "train_B_centre.csv",
    "test_B_centre.csv",
    "train_C_time.csv",
    "test_C_time.csv",
]


def panda_components(df: pd.DataFrame) -> pd.Series:
    """Seven binary components of the manuscript PANDA score."""
    berries = ((df["Berries"] > 0) & (df["Berries"] <= 0.75)).astype(int)
    poultry = df["poultry__score"].astype(int)
    red = (df["Red meats"] == 0).astype(int)
    eggs = df["eggs__score"].astype(int)
    green = (df["Green leafy vegetables"] > 0).astype(int)
    fish = df["fish_and_other_seafood__score"].astype(int)
    whole_grains = ((df["Whole grains"] > 0) & (df["Whole grains"] <= 2.06521739130435)).astype(int)
    return berries + poultry + red + eggs + green + fish + whole_grains


def main():
    for fname in FILES:
        fpath = os.path.join(INPUT_DIR, fname)
        if not os.path.exists(fpath):
            print(f"{fname}: not found, skipped")
            continue
        df = pd.read_csv(fpath)
        panda = panda_components(df)
        n_changed = int((df["PANDA_score"] != panda).sum()) if "PANDA_score" in df.columns else None
        df["PANDA_score"] = panda
        df["PANDA_score_pct"] = panda.rank(pct=True) * 100
        df.to_csv(fpath, index=False)
        print(f"{fname}: n={len(df)}, PANDA changed={n_changed}")


if __name__ == "__main__":
    main()
