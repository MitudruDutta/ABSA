#!/usr/bin/env python3
"""
Per-aspect comparison: RoBERTa ABSA vs TF-IDF baseline, on VAL.

Re-trains the TF-IDF baseline (fast) and runs the saved RoBERTa model, then
prints per-aspect macro-F1 side by side + the delta, so you see where the
transformer earns its keep (expected: the aspects TF-IDF was weak on).
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from taxonomy import ASPECTS, ASPECT_DEFINITIONS

D = os.path.join(ROOT, "data")
MODELDIR = os.path.join(ROOT, "models", "roberta-absa")
LABEL2ID = {"not_present": 0, "negative": 1, "neutral": 2}
HYP = {a: f"{a}: {ASPECT_DEFINITIONS[a]}" for a in ASPECTS}


def tfidf_per_aspect(tr, va):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000,
                          sublinear_tf=True, strip_accents="unicode")
    Xtr = vec.fit_transform(tr["body"].fillna(""))
    Xva = vec.transform(va["body"].fillna(""))
    out = {}
    for a in ASPECTS:
        ytr = tr[f"asp_{a}"].fillna("not_present").replace("", "not_present")
        yva = va[f"asp_{a}"].fillna("not_present").replace("", "not_present")
        m = LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0).fit(Xtr, ytr)
        out[a] = f1_score(yva, m.predict(Xva), average="macro", zero_division=0)
    return out


def roberta_per_aspect(va):
    tok = AutoTokenizer.from_pretrained(MODELDIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODELDIR).cuda().eval()
    out = {}
    for a in ASPECTS:
        gold = va[f"asp_{a}"].fillna("not_present").replace("", "not_present").map(LABEL2ID).values
        preds = []
        bodies = va["body"].fillna("").tolist()
        for i in range(0, len(bodies), 32):
            batch = bodies[i:i + 32]
            enc = tok(batch, [HYP[a]] * len(batch), truncation=True, max_length=256,
                      padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits = model(**enc).logits
            preds.extend(logits.argmax(-1).cpu().numpy().tolist())
        out[a] = f1_score(gold, np.array(preds), average="macro", zero_division=0)
    return out


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    tr = pd.read_csv(os.path.join(D, "train.csv"))
    va = pd.read_csv(os.path.join(D, f"{split}.csv"))
    print(f"evaluating on: {split.upper()}")
    print("computing TF-IDF baseline...")
    tf = tfidf_per_aspect(tr, va)
    print("running RoBERTa...")
    rb = roberta_per_aspect(va)

    print("\n=== per-aspect macro-F1 (VAL) ===")
    print(f"{'aspect':12s} {'TF-IDF':>8s} {'RoBERTa':>8s} {'delta':>8s}")
    deltas = []
    for a in ASPECTS:
        d = rb[a] - tf[a]
        deltas.append(d)
        flag = " <-- " + ("roberta" if d > 0.02 else ("tfidf" if d < -0.02 else "tie")) if abs(d) > 0.02 else ""
        print(f"{a:12s} {tf[a]:8.3f} {rb[a]:8.3f} {d:+8.3f}{flag}")
    print(f"\n{'MEAN':12s} {np.mean(list(tf.values())):8.3f} {np.mean(list(rb.values())):8.3f} {np.mean(deltas):+8.3f}")


if __name__ == "__main__":
    main()
