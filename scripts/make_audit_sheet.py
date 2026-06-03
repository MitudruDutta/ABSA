#!/usr/bin/env python3
"""
Build a BLIND human-audit sheet from the LLM-labeled pool.

Why blind: you label without seeing the LLM's answers, so the agreement number
is honest (not anchored to the model). The LLM labels are stashed in a separate
key file the scorer reads later.

Outputs:
  data/audit_sheet.csv  -> ticket_id, body, human_<aspect> (BLANK, you fill)
  data/audit_key.csv    -> ticket_id, llm_<aspect>          (answer key, do not open)

How to label audit_sheet.csv:
  For each ticket, read the body. For each aspect that the ticket actually
  refers to, type 'negative' or 'neutral' in human_<aspect>.
  Leave the cell BLANK if the aspect is not present (blank == not_present).
"""

import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from taxonomy import ASPECTS

IN = os.path.join(ROOT, "data", "labeled_llm.csv")
SHEET = os.path.join(ROOT, "data", "audit_sheet.csv")
KEY = os.path.join(ROOT, "data", "audit_key.csv")

N = 80
SEED = 11


def main():
    df = pd.read_csv(IN)
    sample = df.sample(min(N, len(df)), random_state=SEED).reset_index(drop=True)

    sheet = pd.DataFrame({"ticket_id": sample["ticket_id"], "body": sample["body"]})
    for a in ASPECTS:
        sheet[f"human_{a}"] = ""        # you fill: negative | neutral | (blank=not_present)
    sheet.to_csv(SHEET, index=False)

    key = pd.DataFrame({"ticket_id": sample["ticket_id"]})
    for a in ASPECTS:
        key[f"llm_{a}"] = sample[f"asp_{a}"].fillna("not_present")
    key.to_csv(KEY, index=False)

    print(f"audit sheet : {SHEET}  ({len(sheet)} rows, BLANK human_ columns)")
    print(f"answer key  : {KEY}  (do not open while labeling)")
    print("\nFill human_<aspect> with negative/neutral where present; blank = not_present.")
    print("Then run: python scripts/score_audit.py")


if __name__ == "__main__":
    main()
