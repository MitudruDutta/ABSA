#!/usr/bin/env python3
"""
Apply your `verdict` choices from disagreements.csv to produce GOLD audit labels.

verdict values:
  human / (blank)   -> keep the expert label (already in audit_sheet.csv)
  llm               -> adopt Qwen's value for that cell
  negative/neutral/not_present -> use that explicit correction

Writes the resolved labels back into audit_sheet.csv (the human_<aspect> cols),
then re-scores so you see the gold-vs-LLM agreement.
"""

import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from taxonomy import ASPECTS

SHEET = os.path.join(ROOT, "data", "audit_sheet.csv")
DIS = os.path.join(ROOT, "data", "disagreements.csv")
VALID = {"negative", "neutral", "not_present"}


def main():
    h = pd.read_csv(SHEET, dtype=str).fillna("")
    h["ticket_id"] = h["ticket_id"].astype(int)
    d = pd.read_csv(DIS, dtype=str).fillna("")
    d["ticket_id"] = d["ticket_id"].astype(int)

    changed = 0
    for _, r in d.iterrows():
        v = str(r["verdict"]).strip().lower()
        if v in ("", "human"):
            continue  # keep expert label
        if v == "llm":
            new = str(r["llm"]).strip().lower()
        elif v in VALID:
            new = v
        else:
            print(f"  ! bad verdict '{v}' on ticket {r['ticket_id']}/{r['aspect']} - skipped")
            continue
        cell = "" if new == "not_present" else new
        h.loc[h["ticket_id"] == r["ticket_id"], f"human_{r['aspect']}"] = cell
        changed += 1

    h.to_csv(SHEET, index=False)
    print(f"applied {changed} corrections -> {SHEET}")
    print("now re-run: python scripts/score_audit.py")


if __name__ == "__main__":
    main()
