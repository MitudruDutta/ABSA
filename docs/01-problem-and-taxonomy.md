# Phase 1 — Problem Definition & Taxonomy

[← docs index](README.md) · next: [02 Data →](02-data.md)

---

## 1.1 The problem

Standard sentiment analysis says *"this ticket is negative."* That is not actionable —
a support team needs to know **what** the customer is unhappy about and **where to send
the ticket**.

**Aspect-Based Sentiment Analysis (ABSA)** decomposes each ticket into per-aspect
sentiment:

> *"I was overcharged and the app keeps crashing"* → **billing: negative, software:
> negative, performance: not_present, …**

On top of that the system routes the ticket to a queue and assigns a priority.

## 1.2 Scope guardrail (the most important early decision)

Fixed aspect **categories**, *not* open-ended span extraction.

This converts ABSA from a research-grade sequence-labeling problem into a tractable
classification problem. Open-ended aspect-term extraction is where ABSA side-projects
rabbit-hole and never finish. Fixing the categories up front kept the project shippable.

## 1.3 Aspect taxonomy — 9 aspects

`billing · account · performance · security · bug · feature · network · hardware · software`

These were **derived from the data, not invented.** Process:

1. The raw dataset has free-text `tag_*` columns — **886 unique tags**.
2. Dumped the full tag vocabulary; bucketed every tag with count ≥ 15.
3. Collapsed them into 9 canonical aspects via a keyword map (`ASPECT_MAP` in
 `taxonomy.py`).

### Decisions made here
- **Fold, don't over-split.** `analytics → feature`, `database/backup/storage →
 software`, `data-loss/privacy → security`. A separate 10th "data" aspect was
 considered and rejected (too sparse).
- **Multi-map allowed.** A tag may seed more than one aspect
 (`Recovery → account + performance`, `Firewall → security + network`). It is only a
 weak prior, so over-mapping is harmless and improves recall.
- **Exclude department/process tags.** `Tech Support`, `IT`, `Feedback`, `Sales`,
 marketing terms — not product aspects, so dropped.

The keyword map produces `aspect_tags`, a **weak prior** (≈94% of tickets get ≥1
aspect). **It is not ground truth** — spot-checking showed ~20–30% noise (e.g. a
QuickBooks-sync ticket tagged `billing`). Its only roles: seed and stratify the
labeling step. We never train on it directly.

## 1.4 Sentiment label space — 3 classes per aspect

`not_present / negative / neutral`

### Why `positive` was dropped
The first LLM-labeling pass produced ~221 "positive" cells. Spot-checking found
**13/13 were actually neutral inquiries** mislabeled positive — the model reads
"request enhancement", "optimize", "scalability" as praise. Genuine praise is
near-zero in support tickets.

A hardened prompt + few-shot examples cut the volume (395 → 221) **but not the error
rate**. So `positive` was folded into `neutral` — and because the errors *were*
neutral, the fold is approximately correct. Result: a clean, learnable 3-class space.

## 1.5 Routing target

`queue` — a 10-class label that ships with the data (Technical Support, Billing and
Payments, IT Support, …). Used for the shared-vs-per-queue experiment ([phase 6](06-results-and-experiments.md)).

---

**Artifacts:** `taxonomy.py` (locked single source of truth: aspects, definitions,
sentiment labels, keyword map, queues).
