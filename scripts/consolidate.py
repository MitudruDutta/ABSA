#!/usr/bin/env python3
"""
Consolidate the 3 English-filtered support-ticket CSVs into one clean file
for the aspect-based support-ticket routing project.

Input  : ABSA/datasets/*.csv   (already filtered to language == 'en')
Output : ABSA/data/tickets_clean.csv

Final schema:
    ticket_id | body | queue | type | priority | aspect_tags | aspect_sentiment | n_chars

Notes
-----
- `aspect_sentiment` is intentionally left EMPTY. It is the ABSA label you must
  still generate (LLM-label ~400-600 rows against the taxonomy, then hand-check).
  This script does NOT fabricate sentiment.
- `aspect_tags` is a WEAK aspect-detection signal derived from the raw `tag_*`
  columns. It is not ground truth, but a useful prior / seed for labeling.
"""

import glob
import os
import re
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from taxonomy import map_aspects  # single source of truth for the aspect map

SRC_DIR = os.path.join(ROOT, "datasets")
OUT_DIR = os.path.join(ROOT, "data")
OUT_FILE = os.path.join(OUT_DIR, "tickets_clean.csv")

# Columns we read from each source (union; some files lack some of these).
KEEP_INPUT = ["body", "queue", "type", "priority"]
TAG_COLS = [f"tag_{i}" for i in range(1, 10)]  # tag_1..tag_9

MIN_BODY_CHARS = 30


def clean_text(s):
    """Strip literal/real newlines, collapse whitespace, trim."""
    if not isinstance(s, str):
        return ""
    s = s.replace("\\n", " ").replace("\\r", " ")   # literal backslash-n
    s = re.sub(r"\s+", " ", s)                       # real newlines + runs of space
    return s.strip()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC_DIR, "*.csv")))
    if not files:
        raise SystemExit(f"No CSVs found in {SRC_DIR}")

    frames = []
    print("=== loading ===")
    for f in files:
        df = pd.read_csv(f)
        # align: keep core input cols + whatever tag cols exist in this file
        tagcols_here = [c for c in TAG_COLS if c in df.columns]
        cols = [c for c in KEEP_INPUT if c in df.columns] + tagcols_here
        df = df[cols].copy()
        # ensure all tag cols exist so concat aligns
        for c in TAG_COLS:
            if c not in df.columns:
                df[c] = pd.NA
        frames.append(df[KEEP_INPUT + TAG_COLS])
        print(f"  {os.path.basename(f):50s} {len(df):6d} rows")

    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    print(f"\nconcatenated: {n0} rows")

    # --- clean body ---
    df["body"] = df["body"].apply(clean_text)

    # --- drop empty / junk-short bodies ---
    df["n_chars"] = df["body"].str.len()
    df = df[df["n_chars"] >= MIN_BODY_CHARS].copy()
    print(f"after dropping bodies < {MIN_BODY_CHARS} chars: {len(df)}  (removed {n0 - len(df)})")

    # --- dedup on cleaned body (28k & 20k overlap heavily) ---
    before = len(df)
    df = df.drop_duplicates(subset="body").reset_index(drop=True)
    print(f"after dedup on body: {len(df)}  (removed {before - len(df)} duplicates)")

    # --- merge tag_1..tag_9 -> aspect_tags (pipe-joined) ---
    def row_aspects(row):
        return "|".join(map_aspects([row[c] for c in TAG_COLS]))
    df["aspect_tags"] = df.apply(row_aspects, axis=1)

    # --- normalize labels ---
    for c in ["queue", "type", "priority"]:
        df[c] = df[c].astype(str).str.strip()
    df["priority"] = df["priority"].str.lower()

    # --- empty label placeholder (you fill this in the labeling step) ---
    df["aspect_sentiment"] = ""

    # --- ids + final column order ---
    df.insert(0, "ticket_id", range(1, len(df) + 1))
    final = ["ticket_id", "body", "queue", "type", "priority",
             "aspect_tags", "aspect_sentiment", "n_chars"]
    df = df[final]

    df.to_csv(OUT_FILE, index=False)

    # --- report ---
    print("\n=== REPORT ===")
    print(f"rows written : {len(df)}")
    print(f"output       : {OUT_FILE}")
    print(f"columns      : {final}")
    print("\nqueue distribution:")
    print(df["queue"].value_counts().to_string())
    n_no_aspect = (df["aspect_tags"] == "").sum()
    print(f"\nrows with >=1 mapped aspect : {len(df) - n_no_aspect}")
    print(f"rows with NO mapped aspect  : {n_no_aspect}  (tags were generic/dept-only)")
    print("\naspect frequency (multi-label):")
    exploded = df["aspect_tags"].str.split("|").explode()
    print(exploded[exploded != ""].value_counts().to_string())
    print(f"\nbody chars: median {int(df.n_chars.median())}, "
          f"min {int(df.n_chars.min())}, max {int(df.n_chars.max())}")
    print("\nREMINDER: 'aspect_sentiment' is EMPTY by design. "
          "Generate it via LLM-labeling + hand-check before training the ABSA head.")


if __name__ == "__main__":
    main()
