#!/usr/bin/env python3
"""
Stratified sampler: pick a labeling pool from tickets_clean.csv.

Goal: a ~600-ticket pool that (a) covers all 9 aspects with a floor each
(so rare aspects like hardware/billing have enough positives to train the
sentiment head), and (b) covers all 10 queues with a floor each.

Output: ABSA/data/label_pool.csv with 9 empty per-aspect sentiment columns
        (asp_<aspect>) for the labeler to fill with one of:
        not_present / negative / neutral / positive.

`aspect_tags` is carried over as a WEAK PRIOR to speed labeling, not an answer.
"""

import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from taxonomy import ASPECTS, QUEUES

IN_FILE = os.path.join(ROOT, "data", "tickets_clean.csv")
OUT_FILE = os.path.join(ROOT, "data", "label_pool.csv")

N_TARGET = 600
PER_ASPECT_FLOOR = 55   # rows containing each aspect (in the prior)
PER_QUEUE_FLOOR = 15
SEED = 7


def main():
    df = pd.read_csv(IN_FILE)
    df["aspect_tags"] = df["aspect_tags"].fillna("")
    df["_aspset"] = df["aspect_tags"].apply(
        lambda s: set(s.split("|")) if s else set()
    )

    selected = set()

    # Phase A — aspect coverage
    for a in ASPECTS:
        pool = df[df["_aspset"].apply(lambda s: a in s)]
        take = pool.sample(min(PER_ASPECT_FLOOR, len(pool)), random_state=SEED)
        selected.update(take["ticket_id"].tolist())

    # Phase B — queue coverage
    for q in QUEUES:
        have = df[df["ticket_id"].isin(selected) & (df["queue"] == q)]
        if len(have) < PER_QUEUE_FLOOR:
            pool = df[(df["queue"] == q) & (~df["ticket_id"].isin(selected))]
            need = PER_QUEUE_FLOOR - len(have)
            take = pool.sample(min(need, len(pool)), random_state=SEED)
            selected.update(take["ticket_id"].tolist())

    # Phase C — top up to N_TARGET with random remainder
    if len(selected) < N_TARGET:
        pool = df[~df["ticket_id"].isin(selected)]
        take = pool.sample(min(N_TARGET - len(selected), len(pool)), random_state=SEED)
        selected.update(take["ticket_id"].tolist())

    out = df[df["ticket_id"].isin(selected)].copy().reset_index(drop=True)
    out = out.drop(columns=["_aspset", "aspect_sentiment", "n_chars"], errors="ignore")

    for a in ASPECTS:
        out[f"asp_{a}"] = ""        # to fill: not_present|negative|neutral|positive
    out["label_source"] = ""        # llm / human / corrected
    out["notes"] = ""

    out.to_csv(OUT_FILE, index=False)

    print("=== LABEL POOL ===")
    print(f"rows: {len(out)}  ->  {OUT_FILE}")
    print("\nqueue coverage:")
    print(out["queue"].value_counts().reindex(QUEUES).to_string())
    print("\naspect coverage (prior, multi-label):")
    asp = out["aspect_tags"].fillna("").str.split("|").explode()
    print(asp[asp != ""].value_counts().reindex(ASPECTS).to_string())


if __name__ == "__main__":
    main()
