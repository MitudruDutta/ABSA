#!/usr/bin/env python3
"""
Per-aspect ensemble: for each aspect, pick the model that wins on VAL
(TF-IDF or RoBERTa), then evaluate that routed combination on TEST.

Rationale: eval showed the two models win different aspects (TF-IDF on bug,
RoBERTa on most). Routing each aspect to its better model is ~free and
captures both strengths.
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


def gold(df, a):
    return df[f"asp_{a}"].fillna("not_present").replace("", "not_present").map(LABEL2ID).values


def tfidf_preds(tr, va, te):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000,
                          sublinear_tf=True, strip_accents="unicode")
    Xtr = vec.fit_transform(tr["body"].fillna(""))
    Xva = vec.transform(va["body"].fillna(""))
    Xte = vec.transform(te["body"].fillna(""))
    pv, pt = {}, {}
    for a in ASPECTS:
        ytr = tr[f"asp_{a}"].fillna("not_present").replace("", "not_present")
        m = LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0).fit(Xtr, ytr)
        pv[a] = pd.Series(m.predict(Xva)).map(LABEL2ID).values
        pt[a] = pd.Series(m.predict(Xte)).map(LABEL2ID).values
    return pv, pt


def roberta_preds(va, te):
    tok = AutoTokenizer.from_pretrained(MODELDIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODELDIR).cuda().eval()
    def run(df, a):
        out = []
        bodies = df["body"].fillna("").tolist()
        for i in range(0, len(bodies), 64):
            b = bodies[i:i + 64]
            enc = tok(b, [HYP[a]] * len(b), truncation=True, max_length=256,
                      padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out.extend(model(**enc).logits.argmax(-1).cpu().numpy().tolist())
        return np.array(out)
    pv, pt = {}, {}
    for a in ASPECTS:
        pv[a] = run(va, a); pt[a] = run(te, a)
    return pv, pt


def main():
    tr = pd.read_csv(os.path.join(D, "train.csv"))
    va = pd.read_csv(os.path.join(D, "val.csv"))
    te = pd.read_csv(os.path.join(D, "test.csv"))

    print("TF-IDF preds...")
    tf_v, tf_t = tfidf_preds(tr, va, te)
    print("RoBERTa preds...")
    rb_v, rb_t = roberta_preds(va, te)

    # choose per-aspect winner on VAL
    print("\naspect        pick     TF(test) RB(test) ENS(test)")
    ens, tf_s, rb_s = [], [], []
    for a in ASPECTS:
        gv = gold(va, a)
        f_tf = f1_score(gv, tf_v[a], average="macro", zero_division=0)
        f_rb = f1_score(gv, rb_v[a], average="macro", zero_division=0)
        pick = "roberta" if f_rb >= f_tf else "tfidf"
        gt = gold(te, a)
        tt = f1_score(gt, tf_t[a], average="macro", zero_division=0)
        rt = f1_score(gt, rb_t[a], average="macro", zero_division=0)
        et = rt if pick == "roberta" else tt   # ensemble = chosen model on test
        ens.append(et); tf_s.append(tt); rb_s.append(rt)
        print(f"{a:12s} {pick:8s} {tt:8.3f} {rt:8.3f} {et:8.3f}")
    print(f"\n{'MEAN':12s} {'':8s} {np.mean(tf_s):8.3f} {np.mean(rb_s):8.3f} {np.mean(ens):8.3f}")
    print(f"\nensemble (test) mean aspect macro-F1: {np.mean(ens):.3f}")


if __name__ == "__main__":
    main()
