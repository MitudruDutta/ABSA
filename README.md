# ABSA — Aspect-Based Support-Ticket Router

Aspect-Based Sentiment Analysis on customer-support tickets. Instead of one
overall sentiment, each ticket is decomposed into **per-aspect sentiment** (e.g.
*billing → negative, login → neutral*) and routed to the right queue.

**Novel core:** a single shared ABSA model with one unified aspect taxonomy that
routes tickets — evaluated as a *shared-model-vs-per-queue-models* tradeoff
(F1 loss vs. maintenance/operational saving).

## Label scheme (`taxonomy.py`, locked)

- **9 aspects:** billing, account, performance, security, bug, feature, network,
  hardware, software.
- **Per-aspect sentiment (3-class):** `not_present` / `negative` / `neutral`.
  A `positive` class was dropped — in support tickets genuine praise is ~zero and
  the LLM labeler's "positive" cells were ~100% actually-neutral inquiries.

## Data

Source: Tobias Bueck multilingual IT-support-ticket dataset (Hugging Face),
filtered to **English**, deduped. Text is **LLM-synthetic** (polished/formal) —
a known realism limitation, noted honestly rather than claimed as production data.

| File | What |
|---|---|
| `datasets/*.csv` | raw source (English-filtered) |
| `data/tickets_clean.csv` | 24,934 cleaned unique tickets (body, queue, type, priority, aspect_tags prior) |
| `data/label_pool.csv` | 600 stratified tickets for labeling |
| `data/labeled_llm.csv` | 600 tickets labeled (Qwen3-4B, 3-class per aspect) |
| `data/labeled_llm_v1.csv` | archived 4-class labels (pre positive-drop) |
| `data/audit_sheet.csv` | 80-ticket human/expert audit |
| `data/disagreements.csv` | human-vs-LLM disagreement review |

## Pipeline (`scripts/`)

1. `consolidate.py` — merge/clean/dedup raw → `tickets_clean.csv`
2. `sample_for_labeling.py` — stratified 600-row labeling pool
3. `label_llm.py` — local Qwen3-4B (4-bit) aspect-sentiment labeler (resumable)
4. `make_audit_sheet.py` / `score_audit.py` — blind audit + agreement metrics
5. `make_disagreements.py` / `apply_disagreements.py` — human adjudication

## Label quality (80-ticket audit)

| Metric | Value |
|---|---|
| cell exact agreement | 90.3% |
| Cohen's kappa | 0.710 |
| aspect presence F1 | 79.0% (recall 93%, precision 69% — LLM over-tags) |
| polarity agreement (co-present) | 91.2% |

## Results (per-aspect sentiment, mean macro-F1)

| split | TF-IDF + LogReg | RoBERTa sentence-pair (weighted loss) |
|---|---|---|
| val  | 0.677 | 0.727 |
| **test (locked)** | **0.661** | **0.718** |

RoBERTa wins 8/9 aspects on val (only `account`, the rarest, loses). Class-weighted
loss + 5-epoch tuning lifted the weak aspects (security 0.62→0.70, network 0.60→0.74).

RoBERTa overall (pooled) macro-F1 on val = 0.814. The transformer's lift
concentrates on data-rich aspects (hardware, account, performance, network);
TF-IDF stays competitive (wins security/software) — expected on clean synthetic
text, where the transformer's edge would widen on noisier real tickets.
`bug`/`security` are capped for both by a near-empty `neutral` class.

Note: DeBERTa-v3 was tried first but gave nan loss under mixed precision in this
transformers version; switched to RoBERTa-base (stable, bf16).

## Status

Done: cleaning, taxonomy lock, sampling, LLM labeling (2,500 tickets, 3-class),
80-ticket audit, train/val/test split, TF-IDF baseline, RoBERTa ABSA, test eval.
Next: shared-vs-per-queue experiment (the novel core) → FastAPI + Streamlit + Docker.

## Run

```bash
source ~/python/bin/activate
python scripts/consolidate.py
python scripts/sample_for_labeling.py
python scripts/label_llm.py        # downloads Qwen3-4B on first run
```
