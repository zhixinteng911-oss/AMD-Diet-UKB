import os

import pandas as pd
from sklearn.model_selection import train_test_split

os.makedirs("output", exist_ok=True)

BASE = "."

df = pd.read_csv(f"{BASE}/input/analysis_master_dataset.csv", low_memory=False)
train, test = train_test_split(
    df, test_size=0.3, random_state=42, stratify=df["event_amd"]
)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
print(f"train: N={len(train)}, events={train['event_amd'].sum()}")
print(f"test: N={len(test)}, events={test['event_amd'].sum()}")

train.to_csv(f"{BASE}/input/train_A_random.csv", index=False)
test.to_csv(f"{BASE}/input/test_A_random.csv", index=False)
train[["eid"]].to_csv(f"{BASE}/input/train_A_random_eid.csv", index=False)
test[["eid"]].to_csv(f"{BASE}/input/test_A_random_eid.csv", index=False)

def fmt(d):
    return (f"N={len(d)}, Events={d.event_amd.sum()}, "
            f"followup_mean={d.followup.mean():.4f}, followup_med={d.followup.median():.4f}, "
            f"followup_sd={d.followup.std():.4f}")


def write_summary():
    idx_path = f"{BASE}/output/index_panda_amed_final.csv"
    if not os.path.exists(idx_path):
        print(f"split done; {idx_path} not found, summary skipped")
        return
    idx = pd.read_csv(idx_path)
    age = df[["eid", "age_at_exit", "age_at_index"]].copy()
    age["followup"] = age["age_at_exit"] - age["age_at_index"]
    idx = idx.merge(age[["eid", "followup"]], on="eid", how="left")
    train_idx = idx[idx["eid"].isin(train["eid"])]
    test_idx = idx[idx["eid"].isin(test["eid"])]
    with open(f"{BASE}/output/three_datasets_summary.txt", "w") as f:
        f.write(f"Full: {fmt(idx)}\n")
        f.write(f"Train: {fmt(train_idx)}\n")
        f.write(f"Test: {fmt(test_idx)}\n")
        f.write(f"train missing in index: {len(set(train.eid) - set(idx.eid))}, "
                f"test missing in index: {len(set(test.eid) - set(idx.eid))}\n")
        for q, lo, hi in [(0, 0, 0), (1, 1, 2), (2, 3, 4), (3, 5, 7)]:
            s = test_idx[(test_idx.PANDA_score >= lo) & (test_idx.PANDA_score <= hi)]
            f.write(f"Test_PANDA_Q{q + 1}: N={len(s)}, Events={s.event_amd.sum()}, followup={s.followup.mean():.2f}\n")
        for q, lo, hi in [(0, 0, 0), (1, 1, 3), (2, 4, 6), (3, 7, 9)]:
            s = test_idx[(test_idx.Mediterranean_score >= lo) & (test_idx.Mediterranean_score <= hi)]
            f.write(f"Test_aMED_Q{q + 1}: N={len(s)}, Events={s.event_amd.sum()}, followup={s.followup.mean():.2f}\n")
    print("split done; summary written to output/three_datasets_summary.txt")


write_summary()
