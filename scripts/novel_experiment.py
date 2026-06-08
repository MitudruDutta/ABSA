#!/usr/bin/env python3
"""
NOVEL EXPERIMENT: one shared ABSA model vs per-queue specialist models.

Question the project is built around: does a single shared model (one unified
aspect taxonomy, trained on all tickets) match N per-queue specialist models —
and what is the accuracy-vs-maintenance tradeoff?

Setup (TF-IDF + LogReg for both, so it's apples-to-apples and feasible even on
small queues where a transformer can't be trained):
  - SHARED      : trained once on ALL train tickets; evaluated per queue.
  - PER-QUEUE   : a separate model trained on EACH queue's train tickets only,
                  evaluated on that queue's test tickets.

Metric: per-queue mean aspect macro-F1 on the locked test set.
Reports the per-queue comparison + the maintenance ratio (1 model vs 10).
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from taxonomy import ASPECTS, QUEUES

D = os.path.join(ROOT, "data")


def fit_models(tr):
    """fit a TF-IDF vec + per-aspect LogReg on the given train slice."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000,
                          sublinear_tf=True, strip_accents="unicode")
    X = vec.fit_transform(tr["body"].fillna(""))
    clfs = {}
    for a in ASPECTS:
        y = tr[f"asp_{a}"].fillna("not_present").replace("", "not_present")
        if y.nunique() < 2:                      # degenerate (one class only)
            clfs[a] = ("const", y.iloc[0])
        else:
            clfs[a] = ("model", LogisticRegression(
                max_iter=2000, class_weight="balanced", C=2.0).fit(X, y))
    return vec, clfs


def score_slice(vec, clfs, te_slice):
    """mean aspect macro-F1 of (vec,clfs) on a test slice."""
    if len(te_slice) == 0:
        return np.nan
    X = vec.transform(te_slice["body"].fillna(""))
    fs = []
    for a in ASPECTS:
        y = te_slice[f"asp_{a}"].fillna("not_present").replace("", "not_present")
        kind, obj = clfs[a]
        pred = np.array([obj] * len(te_slice)) if kind == "const" else obj.predict(X)
        fs.append(f1_score(y, pred, average="macro", zero_division=0))
    return float(np.mean(fs))


def main():
    tr = pd.read_csv(os.path.join(D, "train.csv"))
    te = pd.read_csv(os.path.join(D, "test.csv"))

    # SHARED model (trained once on all)
    vec_s, clf_s = fit_models(tr)

    print(f"{'queue':33s} {'n_tr':>5s} {'n_te':>5s} {'shared':>7s} {'perqueue':>9s} {'delta':>7s}")
    rows = []
    for q in QUEUES:
        tr_q = tr[tr["queue"] == q]
        te_q = te[te["queue"] == q]
        shared_f1 = score_slice(vec_s, clf_s, te_q)
        # per-queue specialist
        if len(tr_q) >= 30:
            vec_q, clf_q = fit_models(tr_q)
            spec_f1 = score_slice(vec_q, clf_q, te_q)
        else:
            spec_f1 = np.nan
        d = spec_f1 - shared_f1 if not np.isnan(spec_f1) else np.nan
        rows.append((q, len(tr_q), len(te_q), shared_f1, spec_f1, d))
        ds = f"{d:+.3f}" if not np.isnan(d) else "  n/a"
        sp = f"{spec_f1:.3f}" if not np.isnan(spec_f1) else "  n/a"
        print(f"{q:33s} {len(tr_q):5d} {len(te_q):5d} {shared_f1:7.3f} {sp:>9s} {ds:>7s}")

    df = pd.DataFrame(rows, columns=["queue", "n_tr", "n_te", "shared", "spec", "delta"])
    # test-size-weighted means (over queues where specialist trained)
    valid = df.dropna(subset=["spec"])
    w = valid["n_te"]
    shared_w = np.average(valid["shared"], weights=w)
    spec_w = np.average(valid["spec"], weights=w)
    print("\n--- weighted by test size (queues with a trainable specialist) ---")
    print(f"  SHARED  mean aspect macro-F1 : {shared_w:.3f}")
    print(f"  PER-QUEUE specialists        : {spec_w:.3f}")
    print(f"  delta (specialist - shared)  : {spec_w - shared_w:+.3f}")
    print(f"\n  models to maintain: 1 (shared)  vs  {len(valid)} (per-queue)")
    win = (valid["delta"] > 0.005).sum()
    lose = (valid["delta"] < -0.005).sum()
    print(f"  specialist clearly better on {win}/{len(valid)} queues, "
          f"shared better on {lose}/{len(valid)}")


if __name__ == "__main__":
    main()
