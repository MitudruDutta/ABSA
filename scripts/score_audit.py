#!/usr/bin/env python3
"""
Score LLM labels against the human blind audit.

Reads:
  data/audit_sheet.csv  (your filled human_<aspect>; blank = not_present)
  data/audit_key.csv    (llm_<aspect>)

Reports:
  - Cell-level exact agreement over all aspect x ticket cells.
  - Cohen's kappa (chance-corrected).
  - Aspect PRESENCE detection: does the LLM flag the right aspects?
    (present = negative|neutral, vs not_present) -> precision/recall/F1.
  - POLARITY agreement on cells both call present: negative-vs-neutral accuracy.
  - Per-aspect exact agreement.
"""

import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from taxonomy import ASPECTS

SHEET = os.path.join(ROOT, "data", "audit_sheet.csv")
KEY = os.path.join(ROOT, "data", "audit_key.csv")
VALID = {"negative", "neutral", "not_present"}


def norm(v):
    v = str(v).strip().lower()
    if v in ("", "nan", "none"):
        return "not_present"
    return v if v in VALID else "not_present"


def main():
    h = pd.read_csv(SHEET).set_index("ticket_id")
    k = pd.read_csv(KEY).set_index("ticket_id")
    ids = h.index.intersection(k.index)
    if len(ids) == 0:
        raise SystemExit("no overlapping ticket_ids — did you fill the sheet?")

    filled = h.loc[ids, [f"human_{a}" for a in ASPECTS]].apply(
        lambda c: c.map(norm)).replace("not_present", "")
    if (filled == "").all().all():
        raise SystemExit("audit_sheet.csv has no human labels filled in yet.")

    pairs = []   # (aspect, human, llm)
    for tid in ids:
        for a in ASPECTS:
            hv = norm(h.at[tid, f"human_{a}"])
            lv = norm(k.at[tid, f"llm_{a}"])
            pairs.append((a, hv, lv))
    dfp = pd.DataFrame(pairs, columns=["aspect", "human", "llm"])

    # cell-level exact agreement
    exact = (dfp["human"] == dfp["llm"]).mean()

    # Cohen's kappa
    labels = ["not_present", "negative", "neutral"]
    obs = (dfp["human"] == dfp["llm"]).mean()
    ph = dfp["human"].value_counts(normalize=True)
    pl = dfp["llm"].value_counts(normalize=True)
    pe = sum(ph.get(x, 0) * pl.get(x, 0) for x in labels)
    kappa = (obs - pe) / (1 - pe) if pe < 1 else 0.0

    # presence detection (present = not not_present)
    hp = dfp["human"] != "not_present"
    lp = dfp["llm"] != "not_present"
    tp = int((hp & lp).sum()); fp = int((~hp & lp).sum()); fn = int((hp & ~lp).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    # polarity agreement where BOTH present
    both = dfp[(hp.values) & (lp.values)]
    pol_acc = (both["human"] == both["llm"]).mean() if len(both) else float("nan")

    print(f"audited tickets        : {len(ids)}  ({len(dfp)} aspect-cells)")
    print(f"cell exact agreement   : {exact:.1%}")
    print(f"Cohen's kappa          : {kappa:.3f}")
    print("\naspect PRESENCE detection (LLM vs human):")
    print(f"  precision={prec:.1%}  recall={rec:.1%}  F1={f1:.1%}   (tp={tp} fp={fp} fn={fn})")
    print(f"\nPOLARITY agreement on co-present cells ({len(both)}): {pol_acc:.1%}")
    print("\nper-aspect exact agreement:")
    for a in ASPECTS:
        sub = dfp[dfp["aspect"] == a]
        print(f"  {a:12s} {(sub['human']==sub['llm']).mean():.1%}")


if __name__ == "__main__":
    main()
