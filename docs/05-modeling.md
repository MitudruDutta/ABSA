# Phase 5 — Modeling

[← 04 Audit](04-audit.md) · [docs index](README.md) · next: [06 Results & Experiments →](06-results-and-experiments.md)

---

## 5.1 Train / val / test split

`14,000 / 3,000 / 3,000`, **stratified by queue**, split at the **ticket level** (a
ticket's 9 aspect labels stay together → no leakage). Verified: 0 ticket-id overlap and
0 body overlap across splits. **The test set was locked** and only evaluated for the
final number.

## 5.2 Baseline — TF-IDF + Logistic Regression

One 3-class classifier per aspect on TF-IDF(body), `class_weight='balanced'`, persisted
as `models/tfidf/*.joblib`. Purpose: an honest floor + the cost/latency comparison
point. **Test mean per-aspect macro-F1: 0.724.**

## 5.3 The model — RoBERTa sentence-pair

The framing that makes one model serve all 9 aspects:

```
text_a = ticket body
text_b = "billing: charges, invoices, payments, refunds, ..." (aspect + its definition)
 ────────────── RoBERTa (self-attention over both) ──────────────
 → {not_present, negative, neutral}
```

Run once per aspect. Self-attention lets the ticket tokens attend to the aspect
definition, so a single shared model answers *"is this aspect present, and what's the
sentiment?"* This is the unified-model design the whole project is built around (and
what the [novel experiment](06-results-and-experiments.md) tests).

### Why RoBERTa, and what didn't work
- **DeBERTa-v3 (tried first)** — usually strong on ABSA, but produced **`nan` loss**
 under mixed precision: first `ValueError: Attempting to unscale FP16 gradients`, then
 `nan` even under bf16. The model collapsed to the majority class (macro-F1 0.295).
 A known instability in this transformers version. **Abandoned.**
- **RoBERTa-base** — stable, already cached locally, fits the 6 GB GPU. Trained with
 `bf16`. This is the production model.

### Training configuration
`roberta-base`, sentence-pair, `bf16`, `lr=3e-5`, **3 epochs**, `warmup_ratio=0.1`,
`weight_decay=0.01`, batch 8 × grad-accum 2. Best epoch selected by mean-aspect
macro-F1. ~1 hour on the RTX 4050; resumable from checkpoints.

## 5.4 Handling class imbalance (the key technical issue)

The label distribution is heavily skewed: **80% `not_present`, 16% negative, 3%
neutral.** Three deliberate responses:

1. **Class-weighted loss** — inverse-frequency weights, **sqrt-damped** so the very
 rare `neutral` class doesn't destabilize training. Boosts the gradient on the rare
 classes.
2. **`class_weight='balanced'`** in the TF-IDF baseline too (apples-to-apples).
3. **macro-F1 as the model-selection metric** — chose the best epoch by macro-F1, not
 accuracy, so training optimized for the rare classes rather than the easy majority.

Effect: between iterations, weighted loss specifically rescued the weak aspects —
`security` 0.62 → 0.70, `network` 0.60 → 0.74.

### The residual limit
Even with weighting, `neutral` stays the weak class because there is too little of it:
`bug` has ~52 neutral cells out of 4,122. That is a **data ceiling**, not a tuning
failure — no loss weighting invents signal that isn't in the labels.

---

**Artifacts:** `models/roberta-absa` (safetensors), `models/tfidf/*.joblib`. Full
training code (guard-wrapped) in `notebooks/02_modeling_and_eval.ipynb`.
