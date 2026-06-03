#!/usr/bin/env python3
"""
Export only the cells where the expert-pass (human) label disagrees with the
LLM (Qwen) label, for personal review.

Output: data/disagreements.csv with columns:
  ticket_id, aspect, human, llm, verdict, body
You fill `verdict` per row with ONE of:
  human   -> the expert label is right (Qwen wrong)
  llm     -> Qwen is right (expert label wrong) -> we'll adopt Qwen's value
  other   -> both wrong; put the correct label in `verdict` directly
             (negative / neutral / not_present)
Blank verdict defaults to keeping the human/expert label.

Then run: python scripts/apply_disagreements.py
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
OUT = os.path.join(ROOT, "data", "disagreements.csv")


def norm(v):
    v = str(v).strip().lower()
    return "not_present" if v in ("", "nan", "none") else v


def main():
    h = pd.read_csv(SHEET, dtype=str).fillna("")
    k = pd.read_csv(KEY, dtype=str).fillna("")
    h["ticket_id"] = h["ticket_id"].astype(int)
    k["ticket_id"] = k["ticket_id"].astype(int)
    body = dict(zip(h["ticket_id"], h["body"]))
    kk = k.set_index("ticket_id")

    rows = []
    for _, r in h.iterrows():
        tid = r["ticket_id"]
        for a in ASPECTS:
            hv = norm(r[f"human_{a}"])
            lv = norm(kk.at[tid, f"llm_{a}"])
            if hv != lv:
                rows.append({
                    "ticket_id": tid, "aspect": a,
                    "human": hv, "llm": lv,
                    "verdict": "",
                    "body": body[tid],
                })
    out = pd.DataFrame(rows)
    # sort so a ticket's cells sit together
    out = out.sort_values(["ticket_id", "aspect"]).reset_index(drop=True)
    out.to_csv(OUT, index=False)
    print(f"{len(out)} disagreement cells -> {OUT}")
    print("\nbreakdown:")
    print("  human says present, llm not_present (Qwen MISS):",
          ((out.llm == "not_present") & (out.human != "not_present")).sum())
    print("  llm says present, human not_present (Qwen OVER-TAG):",
          ((out.human == "not_present") & (out.llm != "not_present")).sum())
    print("  both present, polarity differs:",
          ((out.human != "not_present") & (out.llm != "not_present")).sum())
    print("\nfill `verdict`: human | llm | negative | neutral | not_present  (blank=keep human)")


if __name__ == "__main__":
    main()
