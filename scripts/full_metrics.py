#!/usr/bin/env python3
"""
Full metric suite for the RoBERTa ABSA model on the locked TEST set.

The single 0.740 headline is macro-F1 INCLUDING not_present — the harshest
possible measure. Real ABSA reporting uses several views; this prints them all
so the model's true performance is visible:

  1. Aspect DETECTION   : present vs not_present  -> precision/recall/F1  (routing)
  2. Polarity (present) : negative vs neutral, only where aspect is present
  3. macro-F1 (3-class) : the strict number (current 0.740)
  4. weighted-F1        : frequency-weighted (standard for imbalanced)
  5. accuracy
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from taxonomy import ASPECTS, ASPECT_DEFINITIONS

D = os.path.join(ROOT, "data")
MODELDIR = os.path.join(ROOT, "models", "roberta-absa")
LABEL2ID = {"not_present": 0, "negative": 1, "neutral": 2}
HYP = {a: f"{a}: {ASPECT_DEFINITIONS[a]}" for a in ASPECTS}


def main():
    te = pd.read_csv(os.path.join(D, "test.csv"))
    tok = AutoTokenizer.from_pretrained(MODELDIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODELDIR).cuda().eval()

    all_gold, all_pred = [], []
    per = {}
    for a in ASPECTS:
        gold = te[f"asp_{a}"].fillna("not_present").replace("", "not_present").map(LABEL2ID).values
        preds = []
        bodies = te["body"].fillna("").tolist()
        for i in range(0, len(bodies), 64):
            b = bodies[i:i + 64]
            enc = tok(b, [HYP[a]] * len(b), truncation=True, max_length=256,
                      padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                preds.extend(model(**enc).logits.argmax(-1).cpu().numpy().tolist())
        preds = np.array(preds)
        all_gold.append(gold); all_pred.append(preds); per[a] = (gold, preds)

    g = np.concatenate(all_gold); p = np.concatenate(all_pred)

    # 1. detection: present(1) vs not(0)
    gd = (g != 0).astype(int); pd_ = (p != 0).astype(int)
    det_p = precision_score(gd, pd_, zero_division=0)
    det_r = recall_score(gd, pd_, zero_division=0)
    det_f = f1_score(gd, pd_, zero_division=0)

    # 2. polarity where BOTH present
    mask = (g != 0) & (p != 0)
    pol_acc = accuracy_score(g[mask], p[mask]) if mask.sum() else float("nan")
    pol_f = f1_score(g[mask], p[mask], average="macro", zero_division=0) if mask.sum() else float("nan")

    print("=== FULL METRIC SUITE — RoBERTa on TEST (3000 tickets x 9 aspects) ===\n")
    print("1. ASPECT DETECTION (present vs not_present)  [the routing signal]")
    print(f"     precision {det_p:.3f}   recall {det_r:.3f}   F1 {det_f:.3f}")
    print("\n2. POLARITY on present aspects (negative vs neutral)")
    print(f"     accuracy {pol_acc:.3f}   macro-F1 {pol_f:.3f}")
    print("\n3. OVERALL 3-class (the harsh headline)")
    print(f"     macro-F1    {f1_score(g, p, average='macro', zero_division=0):.3f}  (current 0.740 = mean-of-per-aspect)")
    print(f"     weighted-F1 {f1_score(g, p, average='weighted', zero_division=0):.3f}")
    print(f"     micro-F1    {f1_score(g, p, average='micro', zero_division=0):.3f}")
    print(f"     accuracy    {accuracy_score(g, p):.3f}")

    print("\n4. PER-ASPECT detection-F1 vs 3-class-macro-F1:")
    print(f"   {'aspect':12s} {'detect-F1':>9s} {'macro-F1':>9s}")
    for a in ASPECTS:
        ga, pa = per[a]
        det = f1_score((ga != 0).astype(int), (pa != 0).astype(int), zero_division=0)
        mac = f1_score(ga, pa, average="macro", zero_division=0)
        print(f"   {a:12s} {det:9.3f} {mac:9.3f}")


if __name__ == "__main__":
    main()
