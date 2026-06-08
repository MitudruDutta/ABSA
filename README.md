# ABSA — Aspect-Based Support-Ticket Router

Aspect-Based Sentiment Analysis on customer-support tickets. Instead of one overall
sentiment, each ticket is decomposed into **per-aspect sentiment** (e.g.
*billing → negative, login → neutral*) and routed to the right queue with a priority.

**Novel result:** a single shared model with one unified aspect taxonomy **beats 10
per-queue specialist models on 10/10 queues (+0.037 F1) at 1/10th the maintenance** —
specialization fragments scarce data and loses cross-aspect transfer.

📄 **Full process writeup — every decision and why — in [DOCUMENTATION.md](DOCUMENTATION.md).**

---

## Results (held-out test set, 3000 tickets × 9 aspects = 27k cells)

Every number below is **reproduced and verified inside the notebooks** — macro-F1 is
re-derived by hand from the confusion matrix and cross-checked against sklearn.

| metric | score | note |
|---|---|---|
| **macro-F1 (3-class, pooled)** | **0.843** | the imbalance-robust headline |
| aspect-detection F1 (present vs not) | 0.864 | the routing signal |
| polarity accuracy (neg vs neu, where present) | 0.986 | sentiment correctness |
| weighted-F1 / accuracy | 0.944 | *inflated by the 80% `not_present` majority — not the headline* |

Per-class F1: not_present 0.967, negative 0.885, **neutral 0.677** (the weak point —
tiny support, 3.3% of cells).

> **On the class imbalance:** 80% of cells are `not_present`, so accuracy/weighted-F1
> (0.944) flatter the result. **macro-F1 (0.843) is the honest headline** because it
> weights all three classes equally and is not fooled by the majority class. Training
> used class-weighted loss; model selection used macro-F1.

### Model progression (mean-of-per-aspect macro-F1 — the strictest slice)

| stage | score |
|---|---|
| TF-IDF + LogReg baseline | 0.724 |
| RoBERTa sentence-pair + weighted loss | 0.736 |
| per-aspect ensemble | 0.740 |

Scaling 2.5k→20k labels lifted **both** models and shrank the TF-IDF↔RoBERTa gap —
on clean synthetic text at scale a linear baseline nearly matches a fine-tuned
transformer; the transformer's edge would widen on noisier real tickets.
(DeBERTa-v3 was tried first but gave nan loss under mixed precision → RoBERTa-base.)

---

## Approach

**Label scheme** (`taxonomy.py`, locked): 9 aspects (billing, account, performance,
security, bug, feature, network, hardware, software) × 3-class sentiment
(`not_present / negative / neutral`). A `positive` class was dropped — support tickets
are ~never genuine praise; the LLM's "positive" cells were ~100% neutral inquiries.

**Model:** RoBERTa sentence-pair — `[ticket] [SEP] [aspect definition] →
{not_present, negative, neutral}`, one shared model over all 9 aspects.

**Labels:** 20,000 tickets labeled with local Qwen3-4B (4-bit). A blind 80-ticket
human audit measured **90.3% cell agreement, Cohen's κ 0.71** — usable *silver*
labels (LLM over-tags aspects: high recall, lower precision; polarity strong).

**Data:** Tobias Bueck multilingual IT-support-ticket dataset (HuggingFace), filtered
to English, deduped. Text is **LLM-synthetic** (a stated realism limitation).

---

## Repository

```
notebooks/
  01_data_and_labeling.ipynb       clean → taxonomy → Qwen3-4B labeling → audit
  02_modeling_and_eval.ipynb       baseline → RoBERTa → metric VERIFICATION
  03_experiments_and_serving.ipynb ensemble → shared-vs-per-queue → demo
app/
  predict.py                       inference core (ticket → aspects → route)
  api.py                           FastAPI  (POST /predict, GET /health)
  streamlit_app.py                 UI — calls the API over HTTP
taxonomy.py                        locked label space (single source of truth)
data/                              tickets_clean, train/val/test, labels, audit
models/                            roberta-absa (safetensors) + tfidf/*.joblib  [gitignored]
datasets/                         raw English source CSVs
Dockerfile, Dockerfile.streamlit, docker-compose.yml, requirements.txt
```

The notebooks contain the **full** labeler and trainer code, guard-wrapped: heavy
steps (labeling ~hrs, training ~1h) run only if the artifact is missing, else load it.

---

## Run

```bash
pip install -r requirements.txt

# reproduce / inspect the pipeline
jupyter lab notebooks/

# serve: FastAPI hosts the model, Streamlit is a thin HTTP client
docker compose up --build          # API :8000, UI :8501
# or locally:
uvicorn app.api:app --port 8000
ABSA_API_URL=http://localhost:8000 streamlit run app/streamlit_app.py
```

Routing: ticket → aspects → queue of the first negative aspect; priority = high
(≥2 complaints) / medium (1) / low. *"overcharged on my invoice and the app keeps
crashing"* → billing+bug+software negative → **Billing and Payments**, **high**.

---

## Honest limitations

- Labels are **LLM silver** (~90% human agreement), not gold.
- Text is **LLM-synthetic** — cleaner than real tickets.
- The **`neutral` class** is data-limited (esp. `bug`: ~52 neutral of 4122) and is the
  model's weakest point — a data ceiling, not a tuning failure.

Possible extensions: re-label with a larger LLM to break the silver ceiling; drop
`bug`/`security` to binary; threshold tuning for recall-optimised routing.
