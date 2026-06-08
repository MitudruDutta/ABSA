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

### Headline (RoBERTa, locked TEST set, 3000 tickets × 9 aspects = 27k cells)

Verified with sklearn `classification_report`:

| metric | score | what it measures |
|---|---|---|
| **macro-F1 (3-class, pooled)** | **0.843** | standard ABSA headline |
| weighted-F1 / accuracy | **0.944** | frequency-weighted |
| aspect **detection** F1 (present vs not) | **0.864** | the routing signal |
| polarity accuracy (neg vs neu, where present) | **0.986** | sentiment correctness |

Per-class F1: not_present 0.967, negative 0.885, neutral 0.677 (neutral is the
weakest — tiny training support, often confused with not_present).

### Score progression (mean-of-per-aspect macro-F1 — the strictest slice)

| stage | score |
|---|---|
| TF-IDF + LogReg (2.5k) | 0.661 |
| RoBERTa sentence-pair, weighted (2.5k) | 0.718 |
| RoBERTa (20k labels) | 0.736 |
| Per-aspect ensemble (20k) | 0.740 |

Note: 0.740 averages the 9 per-aspect macro-F1s (dragged by `bug`=0.587);
0.843 pools all cells then computes once. Both are valid; pooled is standard.

RoBERTa overall (pooled) macro-F1 on val = 0.842. The **ensemble** routes each
aspect to its val-winning model (TF-IDF for bug/hardware, RoBERTa for the other 7).

### Novel experiment — shared model vs per-queue specialists

The project's core question: does **one shared model** match **N per-queue
specialist models**? Trained one model per queue (10) and compared to the single
shared model, per-queue mean aspect macro-F1 on test:

| | mean macro-F1 | models to maintain |
|---|---|---|
| **Shared (1 model)** | **0.713** | 1 |
| Per-queue specialists | 0.676 | 10 |

**The shared model wins on 10/10 queues (+0.037), at 1/10th the operational cost.**
The gap widens on small queues (Sales −0.121, HR −0.074): specialists starve when
they fragment scarce data, while the shared model transfers cross-aspect signal
across all queues. Specialization *hurts* here.

Key finding: scaling 2.5k→20k lifted **both** models and shrank the TF-IDF↔RoBERTa
gap (+0.057 → +0.012) — on clean synthetic text at scale a cheap linear baseline
nearly matches a fine-tuned transformer; the transformer's edge would widen on
noisier/scarcer real tickets. Class-weighted loss rescued the weak aspects
(every aspect now ≥0.59). `bug` neutral (52/4122) remains the hard floor.

RoBERTa overall (pooled) macro-F1 on val = 0.814. The transformer's lift
concentrates on data-rich aspects (hardware, account, performance, network);
TF-IDF stays competitive (wins security/software) — expected on clean synthetic
text, where the transformer's edge would widen on noisier real tickets.
`bug`/`security` are capped for both by a near-empty `neutral` class.

Note: DeBERTa-v3 was tried first but gave nan loss under mixed precision in this
transformers version; switched to RoBERTa-base (stable, bf16).

## Serving (`app/`)

- `app/predict.py` — inference core: ticket → per-aspect sentiment + route + priority.
- `app/api.py` — FastAPI: `POST /predict`, `GET /health`.
  Run: `uvicorn app.api:app --port 8000`
- `app/streamlit_app.py` — interactive demo.
  Run: `streamlit run app/streamlit_app.py`
- `Dockerfile` — containerized FastAPI service.

Routing: ticket → aspects → route to the queue of the first negative aspect;
priority = high (≥2 negatives) / medium (1) / low. Example: *"overcharged on my
invoice and the app keeps crashing"* → billing+bug+software negative →
**Billing and Payments**, **high** priority.

## Status

Done: cleaning, taxonomy lock, 20k LLM labels (3-class), 80-ticket audit,
14k/3k/3k split, TF-IDF baseline, RoBERTa ABSA + weighted loss, ensemble (0.740),
shared-vs-per-queue experiment, FastAPI + Streamlit + Docker.

Possible extensions: break the silver-label ceiling (re-label with a larger LLM),
drop `bug`/`security` to binary (near-empty neutral class), threshold tuning for
recall-optimised routing.

## Run

```bash
source ~/python/bin/activate
python scripts/consolidate.py
python scripts/sample_for_labeling.py
python scripts/label_llm.py        # downloads Qwen3-4B on first run
```
