# Phase 2 — Data

[← 01 Problem & Taxonomy](01-problem-and-taxonomy.md) · [docs index](README.md) · next: [03 Labeling →](03-labeling.md)

---

## 2.1 Dataset selection (inspected, not assumed)

Three candidate datasets were downloaded and actually inspected:

| dataset | rows | verdict | reason |
|---|---|---|---|
| **Bueck multi-lang IT-support tickets** | 28k+20k+4k | **used** | real varied paragraph text; clean `queue`/`type`/`priority`; topic tags |
| `customer_support_tickets.csv` (suraj520) | 8,469 | rejected | **8469/8469 rows** contain `{product_purchased}` placeholders — 100% templated junk text |
| `Support_tickets.csv` | 50,000 | rejected | **no free text** — 33 engineered tabular columns only |

The lesson applied: *judge a dataset by reading the raw rows, not the row count.* The
suraj520 set looks usable until you open it and see incoherent templated text; the 50k
set has zero ticket body despite a `description_length` column.

## 2.2 Language filtering

The Bueck source is multilingual (de 36k, en 30k, es/fr/pt ~470–810 each). Filtered to
**English only** — es/fr/pt were too small to train per-language, and mixing languages
would confound the model. The German-only files were deleted (no English content).

## 2.3 Deduplication

The 28k and 20k source files **shared 8,399 identical bodies** (one is largely a
superset of the other). Deduped on body. The 4k file added no overlap.

**Result: 24,934 unique English tickets.**

## 2.4 Cleaning (`tickets_clean.csv`)

| step | detail |
|---|---|
| strip literal `\n` | 740 rows stored escaped `\n\n` instead of real newlines |
| collapse whitespace | normalize runs of spaces/newlines |
| drop junk-short | bodies < 30 chars removed |
| drop empties | no-text rows removed |
| dedup | on cleaned body |
| map tags | `tag_*` → `aspect_tags` weak prior (phase 1) |
| drop columns | `answer` (the agent reply = **leakage**), `version`, `language`, `subject` (2607 nulls) |

Final schema: `ticket_id, body, queue, type, priority, aspect_tags`.

Column-shift was verified (bodies contain commas/quotes): the `queue`/`type`/`priority`
columns hold only their valid values, so CSV quoting parsed correctly — no misalignment.

## 2.5 Honest limitation — synthetic text

The Bueck text is **LLM-generated/synthetic**: polished, formal, nearly every ticket
opens "Dear Customer Support Team." It is *not* real production data — it lacks the
typos, anger, and fragments of genuine tickets.

This is stated openly in the README rather than hidden. The practical consequence
(quantified later): on clean synthetic text at scale, a cheap linear baseline nearly
matches a fine-tuned transformer; the transformer's edge would **widen** on noisier
real-world tickets.

---

**Artifacts:** `datasets/*.csv` (raw English source), `data/tickets_clean.csv` (24,934
cleaned unique tickets).
