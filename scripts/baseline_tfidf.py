#!/usr/bin/env python3
"""
TF-IDF + Logistic Regression baseline (the cheap floor).

Two tasks, evaluated on the VAL set (test stays locked):
  1. Queue routing  : body -> queue (10-class)
  2. Aspect ABSA    : body -> per-aspect sentiment (9 x 3-class
                      {not_present, negative, neutral})

Reports macro-F1 (right metric under class imbalance) + accuracy.
This is the baseline the DeBERTa model must beat, and gives the
cost/latency-vs-accuracy comparison line for the writeup.
"""

import os
import sys
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from taxonomy import ASPECTS

D = os.path.join(ROOT, "data")


def load(name):
    return pd.read_csv(os.path.join(D, f"{name}.csv"))


def main():
    tr, va = load("train"), load("val")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000,
                          sublinear_tf=True, strip_accents="unicode")
    t0 = time.time()
    Xtr = vec.fit_transform(tr["body"].fillna(""))
    Xva = vec.transform(va["body"].fillna(""))
    fit_t = time.time() - t0
    print(f"TF-IDF features: {Xtr.shape[1]}  (vectorize {fit_t:.1f}s)\n")

    # --- Task 1: queue routing ---
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0)
    clf.fit(Xtr, tr["queue"])
    pq = clf.predict(Xva)
    print("=== QUEUE ROUTING (10-class) ===")
    print(f"  accuracy : {accuracy_score(va['queue'], pq):.3f}")
    print(f"  macro-F1 : {f1_score(va['queue'], pq, average='macro'):.3f}")
    print(f"  micro-F1 : {f1_score(va['queue'], pq, average='micro'):.3f}\n")

    # --- Task 2: per-aspect sentiment ---
    print("=== ASPECT SENTIMENT (per aspect, 3-class) ===")
    macros = []
    for a in ASPECTS:
        col = f"asp_{a}"
        ytr = tr[col].fillna("not_present")
        yva = va[col].fillna("not_present")
        m = LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0)
        m.fit(Xtr, ytr)
        pred = m.predict(Xva)
        f1m = f1_score(yva, pred, average="macro", zero_division=0)
        acc = accuracy_score(yva, pred)
        macros.append(f1m)
        print(f"  {a:12s} macro-F1={f1m:.3f}  acc={acc:.3f}")
    print(f"\n  MEAN aspect macro-F1: {sum(macros)/len(macros):.3f}")
    print("\n(baseline on VAL; test.csv untouched)")


if __name__ == "__main__":
    main()
