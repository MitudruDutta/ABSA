#!/usr/bin/env python3
"""
Train/val/test split of the labeled tickets.

- Split at TICKET level (a ticket's 9 aspect labels stay together -> no leakage).
- Stratify by queue so all 10 routing classes appear in every split.
- 70 / 15 / 15.

Outputs: data/train.csv, data/val.csv, data/test.csv
Each: ticket_id, body, queue, type, priority, asp_<9 aspects>.
TEST IS LOCKED — do not look at it until final evaluation.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from taxonomy import ASPECTS

IN = os.path.join(ROOT, "data", "labeled_llm.csv")
OUT = os.path.join(ROOT, "data")
SEED = 42


def main():
    df = pd.read_csv(IN)
    cols = ["ticket_id", "body", "queue", "type", "priority"] + [f"asp_{a}" for a in ASPECTS]
    df = df[cols].copy()
    for a in ASPECTS:
        df[f"asp_{a}"] = df[f"asp_{a}"].fillna("not_present").replace("", "not_present")

    # 70 / 30 then 15 / 15
    train, temp = train_test_split(df, test_size=0.30, random_state=SEED,
                                   stratify=df["queue"])
    val, test = train_test_split(temp, test_size=0.50, random_state=SEED,
                                 stratify=temp["queue"])

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)

    print(f"train={len(train)}  val={len(val)}  test={len(test)}")
    print("\nqueue per split (count):")
    rep = pd.DataFrame({
        "train": train["queue"].value_counts(),
        "val": val["queue"].value_counts(),
        "test": test["queue"].value_counts(),
    }).fillna(0).astype(int)
    print(rep.to_string())

    print("\naspect PRESENT count per split (neg+neu):")
    for a in ASPECTS:
        def pres(p):
            return int(p[f"asp_{a}"].isin(["negative", "neutral"]).sum())
        print(f"  {a:12s} train={pres(train):4d}  val={pres(val):3d}  test={pres(test):3d}")
    print("\nTEST set locked -> data/test.csv (do not inspect until final eval)")


if __name__ == "__main__":
    main()
