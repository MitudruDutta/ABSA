#!/usr/bin/env python3
"""
Apply the careful expert-pass human labels into audit_sheet.csv.

Conventions used (confirmed by user):
  - resolved/fixed issue  -> neutral (not negative)
  - crash/outage/slow/downtime -> performance(neg) + the cause aspect
  - marketing/brand-strategy tickets -> all blank unless a real aspect is named
  - request/inquiry/"how-to"/"enhance" -> neutral
  - blank cell == not_present

Labels keyed by ticket_id. Only present aspects listed; rest stay blank.
"""

import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from taxonomy import ASPECTS

SHEET = os.path.join(ROOT, "data", "audit_sheet.csv")

N = "negative"
U = "neutral"

LABELS = {
    12844: {"network": N, "software": N},
    14164: {"security": N, "network": N},
    20635: {"security": N, "software": N},
    728:   {"software": N, "performance": N},
    21731: {"security": N, "software": N},
    12002: {"security": N, "software": N},
    13963: {"feature": N, "software": N},
    5652:  {"feature": U},
    23613: {"bug": N, "software": N},
    13771: {"security": U},
    14942: {"security": N, "software": N},
    7581:  {"billing": U},
    23920: {"billing": U},
    15039: {"billing": N},
    9921:  {},                                  # marketing strategy -> blank
    5340:  {"performance": U, "software": U},    # resolved -> neutral
    959:   {"performance": U, "software": U},    # stabilized/resolved -> neutral
    9066:  {"bug": N, "software": N},
    11116: {"billing": N},
    3037:  {"security": N},
    13550: {"security": N, "software": N},
    11295: {},                                  # marketing strategy -> blank
    16932: {"performance": N, "software": N},
    10959: {"billing": N},
    8783:  {"billing": N},
    14976: {"feature": U},                       # Elasticsearch search optimize -> feature inquiry
    17845: {"feature": U, "billing": U},
    14611: {},                                  # marketing strategy -> blank
    6413:  {"security": U},
    8500:  {"security": N, "account": N, "software": N},
    22528: {"network": N, "performance": N, "software": N},
    24501: {"software": U},
    16405: {"security": U, "billing": U},
    13301: {"performance": N, "software": N},
    11507: {},                                  # marketing performance -> blank
    12659: {"network": N},
    3500:  {"security": U},
    1382:  {},                                  # vague, no clear aspect -> blank
    10867: {"feature": U},
    2839:  {"network": N},                       # "network weakness" (could be security)
    23433: {"performance": N, "network": N},
    11587: {"billing": U},
    9933:  {"billing": U},
    4209:  {"performance": N, "software": N},
    19905: {"billing": N},
    17058: {"bug": N, "software": N},
    4686:  {"billing": U},
    24124: {"hardware": N},
    21186: {"billing": N},
    221:   {"hardware": N, "bug": N},
    22476: {"bug": N, "software": N, "performance": N},
    13590: {"performance": N, "software": N},
    4946:  {"security": U, "feature": U},
    5012:  {"feature": U},
    1969:  {"security": N},
    20884: {"security": U},
    23080: {"billing": N},
    15330: {"feature": U, "billing": U},
    4809:  {"performance": N, "software": N},
    7926:  {"performance": N, "software": N},
    5086:  {"software": N, "bug": N},
    24200: {"billing": N, "bug": N},
    12110: {"security": U},
    11786: {"software": N, "feature": N},        # sync = feature
    6935:  {"software": N, "feature": N},        # compatibility = feature
    24314: {"feature": U},                       # integration/enhancement request
    7751:  {"billing": N, "bug": N},
    3702:  {"security": N},
    15884: {"security": U},                      # improvement request -> neutral
    7479:  {"software": N, "hardware": N},
    12605: {"feature": U},
    24913: {"network": N, "performance": N, "software": N},
    12034: {"security": U},                      # enhancement request -> neutral
    15974: {},                                  # marketing strategy -> blank
    23535: {"bug": N, "software": N, "performance": N},
    20255: {"account": N, "software": N},
    11083: {"feature": U},
    19375: {"billing": N},
    132:   {"hardware": U, "performance": U, "network": U, "software": U},  # resolved -> neutral
    24408: {"network": N, "software": N},
}


def main():
    df = pd.read_csv(SHEET, dtype=str).fillna("")
    df["ticket_id"] = df["ticket_id"].astype(int)
    cols = [f"human_{a}" for a in ASPECTS]
    df[cols] = ""
    miss = 0
    for tid, lab in LABELS.items():
        m = df["ticket_id"] == tid
        if not m.any():
            miss += 1; continue
        for a, v in lab.items():
            df.loc[m, f"human_{a}"] = v
    df.to_csv(SHEET, index=False)
    filled = (df[cols] != "").sum().sum()
    print(f"applied labels to {len(LABELS)} tickets ({miss} ids not found)")
    print(f"total non-blank human cells: {filled}")
    print("\nper-aspect human label counts:")
    for a in ASPECTS:
        vc = df[f"human_{a}"].value_counts()
        d = {k: int(v) for k, v in vc.items() if k}
        print(f"  {a:12s} {d}")


if __name__ == "__main__":
    main()
