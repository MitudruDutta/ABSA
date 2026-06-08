# Phase 4 — Label Quality Audit

[← 03 Labeling](03-labeling.md) · [docs index](README.md) · next: [05 Modeling →](05-modeling.md)

---

LLM labels are **silver, not gold**. Before training on them, we measured how good
they actually are — otherwise the model's ceiling is unknown.

## 4.1 Method — blind audit

- **80 tickets** sampled from the labeled set.
- A careful expert pass labeled them **without seeing the LLM's answers** (the LLM
 labels were stashed in a separate key file).
- The two sets were then compared cell-by-cell (80 tickets × 9 aspects = 720 cells).

Blind labeling matters: if the human sees the LLM's answer first, the "agreement"
measures anchoring, not independent correctness.

### Annotation conventions (locked before labeling)
- a problem that is **already resolved** in the ticket → `neutral` (not negative),
- a crash/outage/slowdown → mark `performance` **plus** the cause aspect,
- marketing/brand-strategy tickets with no product aspect → all `not_present`,
- a request/inquiry/"how-to" → `neutral`.

## 4.2 Results

| metric | value |
|---|---|
| cell exact agreement | **90.3%** |
| Cohen's κ | **0.71** (substantial) |
| aspect-detection precision | 0.69 |
| aspect-detection recall | 0.93 |
| polarity agreement (where both present) | 0.91 |

## 4.3 What the numbers mean

- **The LLM over-tags aspects** — high recall (0.93, finds the real ones), lower
 precision (0.69, adds aspects that aren't quite there). 52 false-positive aspect
 cells vs 8 misses.
- **Polarity is strong** (0.91) — when an aspect is correctly present, negative-vs-
 neutral is usually right.

For a **routing** product this is the *right* bias: better to over-flag an aspect (a
second team glances and passes) than to miss a real complaint (a customer goes
unserved). So the labels' main weakness aligns with the use-case's tolerance.

## 4.4 Honest caveat on the audit itself

The "expert pass" was a careful second annotation, labeled `human-assisted`. Some of
the 70 disagreement cells are **convention differences** (e.g. a resolved issue marked
`neutral` by convention vs `negative` by the LLM), not outright errors. A disagreement-
review sheet was produced so each cell can be adjudicated; the headline 90.3% stands as
the agreement under the stated conventions.

---

**Artifacts:** `data/audit_sheet.csv` (blind human labels), `data/audit_key.csv` (LLM
labels), `data/disagreements.csv` (the 70 disagreement cells for review). Audit code in
`notebooks/01_data_and_labeling.ipynb`.
