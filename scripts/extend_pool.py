#!/usr/bin/env python3
"""
Grow the labeled set toward ~2500 tickets.

Keeps the existing labeled rows in data/labeled_llm.csv untouched, then appends
N_NEW fresh stratified tickets (blank labels) drawn from tickets_clean.csv,
excluding anything already in the pool. Running label_llm.py afterwards resumes
and labels only the new blank rows.

Stratified to keep rare aspects (account/billing/hardware) and rare queues
(HR/General Inquiry) represented.
"""

import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from taxonomy import ASPECTS, QUEUES

CLEAN = os.path.join(ROOT, "data", "tickets_clean.csv")
LABELED = os.path.join(ROOT, "data", "labeled_llm.csv")

TARGET_TOTAL = 20000
PER_ASPECT_FLOOR = 1800   # aim ~1800 present per aspect across the full pool
PER_QUEUE_FLOOR = 400
SEED = 23


def main():
    clean = pd.read_csv(CLEAN)
    clean["aspect_tags"] = clean["aspect_tags"].fillna("")
    clean["_asp"] = clean["aspect_tags"].apply(lambda s: set(s.split("|")) if s else set())

    lab = pd.read_csv(LABELED)
    have_ids = set(lab["ticket_id"])
    n_new = TARGET_TOTAL - len(lab)
    if n_new <= 0:
        print(f"already {len(lab)} labeled rows >= target {TARGET_TOTAL}; nothing to add.")
        return
    print(f"existing labeled: {len(lab)}   adding: {n_new}")

    pool = clean[~clean["ticket_id"].isin(have_ids)].copy()
    picked = set()

    # aspect coverage (count what existing pool already has, top up the deficit)
    existing_asp = {a: lab["asp_" + a].fillna("").isin(["negative", "neutral"]).sum()
                    for a in ASPECTS}
    for a in ASPECTS:
        need = max(0, PER_ASPECT_FLOOR - int(existing_asp[a]))
        if need <= 0:
            continue
        cand = pool[pool["_asp"].apply(lambda s: a in s) & (~pool["ticket_id"].isin(picked))]
        take = cand.sample(min(need, len(cand)), random_state=SEED)
        picked.update(take["ticket_id"])

    # queue coverage
    for q in QUEUES:
        have = lab[lab["queue"] == q].shape[0] + \
               clean[clean["ticket_id"].isin(picked) & (clean["queue"] == q)].shape[0]
        if have < PER_QUEUE_FLOOR:
            cand = pool[(pool["queue"] == q) & (~pool["ticket_id"].isin(picked))]
            take = cand.sample(min(PER_QUEUE_FLOOR - have, len(cand)), random_state=SEED)
            picked.update(take["ticket_id"])

    # top up to n_new
    if len(picked) < n_new:
        cand = pool[~pool["ticket_id"].isin(picked)]
        take = cand.sample(min(n_new - len(picked), len(cand)), random_state=SEED)
        picked.update(take["ticket_id"])
    # trim if overshot
    picked = list(picked)[:n_new]

    new = clean[clean["ticket_id"].isin(picked)][
        ["ticket_id", "body", "queue", "type", "priority", "aspect_tags"]].copy()
    for a in ASPECTS:
        new["asp_" + a] = ""
    new["label_source"] = ""
    new["notes"] = ""

    # align columns to labeled file and append
    out = pd.concat([lab, new[lab.columns.intersection(new.columns).tolist()
                              if set(lab.columns) == set(new.columns) else new.columns]],
                    ignore_index=True)
    # ensure same column order as lab
    for c in lab.columns:
        if c not in out.columns:
            out[c] = ""
    out = out[lab.columns]
    out.to_csv(LABELED, index=False)
    print(f"pool now: {len(out)} rows ({len(new)} new blanks) -> {LABELED}")
    print("next: python scripts/label_llm.py   (resumes, labels the new rows)")


if __name__ == "__main__":
    main()
