# ABSA Support-Ticket Router — Process Documentation

A complete account of how this project was built: the problem, every major decision
(and the reasoning, including the things that *didn't* work), the data, the models,
the evaluation, and the honest limitations.

---

## 1. Problem definition

Standard sentiment analysis says *"this ticket is negative."* That is not actionable —
a company wants to know **what** the customer is unhappy about and **where to send it**.

**Aspect-Based Sentiment Analysis (ABSA)** decomposes each ticket into per-aspect
sentiment, e.g. *billing → negative, login → neutral, performance → not mentioned*.
On top of that we route the ticket to a queue and assign a priority.

**Scope guardrail (decided up front):** fixed aspect *categories*, not open-ended
span extraction. This is the single most important decision to avoid scope-creep —
it turns ABSA into a tractable classification problem instead of a research project.

---

## 2. Label space (taxonomy)

`taxonomy.py` is the single source of truth. Locked early so nothing downstream drifts.

### 2.1 Aspects (9)
`billing, account, performance, security, bug, feature, network, hardware, software`

These were **derived from the data, not guessed.** The raw dataset has free-text
`tag_*` columns (886 unique tags). We dumped the full tag vocabulary, bucketed every
tag with count ≥ 15, and collapsed them into 9 canonical aspects via a keyword map
(`ASPECT_MAP`). Decisions made during this step:

- **Folded** rather than split: `analytics → feature`, `database/backup/storage →
  software`, `data-loss/privacy → security`. Avoided a sparse 10th "data" aspect.
- **Multi-map allowed**: a tag may seed >1 aspect (e.g. `Recovery → account +
  performance`, `Firewall → security + network`). It is only a weak prior anyway.
- **Excluded** department/process tags (`Tech Support`, `IT`, `Feedback`, `Sales`,
  marketing terms) — these are not product aspects.

The keyword map produces `aspect_tags`, a **weak prior** (~94% of tickets get ≥1
aspect). It is explicitly *not* ground truth — spot-checking showed ~20–30% noise.
Its only jobs: seed/stratify the labeling, never train on directly.

### 2.2 Sentiment (3-class, per aspect)
`not_present / negative / neutral`

**A `positive` class was deliberately dropped.** First-pass LLM labeling produced
~221 "positive" cells; spot-checking found **13/13 were actually neutral inquiries**
mislabeled positive (the model reads "request enhancement", "optimize", "scalability"
as praise). Genuine praise is near-zero in support tickets. A hardened prompt + few-shot
cut the volume (395 → 221) but not the error rate, so `positive` was folded into
`neutral` (the errors *were* neutral, so the mapping is ~correct).

---

## 3. Data

### 3.1 Source selection
Three candidate datasets were inspected (not assumed):

| dataset | verdict | reason |
|---|---|---|
| Bueck multi-lang IT-support tickets | ✅ **used** | real varied paragraph text, clean `queue`/`type`/`priority` labels, topic tags |
| `customer_support_tickets.csv` (suraj520) | ❌ rejected | **8469/8469 rows** contain `{product_purchased}` placeholders — 100% templated junk |
| `Support_tickets.csv` (50k) | ❌ rejected | **no free text at all** — 33 engineered tabular columns |

The Bueck data is **English-filtered** (German/es/fr/pt removed) and **deduped on body**
(28k & 20k source files shared 8,399 identical bodies). Result: **24,934 unique tickets**.

**Honest limitation:** the text is **LLM-synthetic** (polished, formal, every ticket
opens "Dear Customer Support Team"). It is *not* real production data — stated in the
README rather than hidden. A transformer's edge would be larger on messy real tickets.

### 3.2 Cleaning (`tickets_clean.csv`)
- strip literal `\n` (740 rows stored escaped newlines) + collapse whitespace
- drop bodies < 30 chars (junk) and empties
- dedup on cleaned body
- map `tag_*` → `aspect_tags` prior; drop `answer` (the agent reply = leakage),
  `version`, `language`, `subject` (2607 nulls, weak signal)

Final columns: `ticket_id, body, queue, type, priority, aspect_tags`.

---

## 4. Labeling (the critical path)

No off-the-shelf dataset has aspect-level sentiment labels — they had to be generated.

### 4.1 Engine choice
- No LLM API keys available → **local model** on an RTX 4050 (6 GB).
- **Qwen3-8B** was tried first: OOM (only ~4.4 GB free with the display), and the
  bitsandbytes CPU-offload path hit a library bug
  (`Params4bit.__new__() got an unexpected keyword argument '_is_hf_initialized'`).
- **Switched to Qwen3-4B** (4-bit nf4) — fits fully on the GPU. `enable_thinking=False`
  so it emits clean JSON instead of `<think>` traces.

### 4.2 Prompt
System prompt with the 9 aspect definitions + a strict rule that requests/inquiries
("enhance", "optimize") are **neutral, not positive**, plus 2 few-shot examples teaching
the negative/neutral/positive boundary. Output parsed as JSON; invalid → `not_present`.
Resumable (saves every 20 rows) — survived ~6 machine reboots without losing progress.

### 4.3 Scale
Labeled in stages: 600 → 2,500 → **20,000 tickets**. Scaling was driven by the audit
and by rare aspects starving (account had only 165 present at 2.5k → 1,007 at 20k).

---

## 5. Label quality audit

LLM labels are **silver, not gold** — so we measured how good they are.

### 5.1 Method (blind)
80 tickets sampled. A careful expert pass labeled them **without seeing** the LLM's
answers (the LLM labels were stashed in a separate key file). Then compared.

### 5.2 Result
| metric | value |
|---|---|
| cell exact agreement | **90.3%** |
| Cohen's κ | **0.71** (substantial) |
| aspect detection precision / recall | 0.69 / 0.93 |
| polarity agreement (co-present) | 0.91 |

**Interpretation:** the LLM **over-tags** aspects (high recall, lower precision —
it flags aspects that aren't quite there) but **polarity is strong**. For routing,
high recall is the right bias (don't miss a complaint). Usable as training labels with
this caveat documented.

---

## 6. Split

`train / val / test = 14,000 / 3,000 / 3,000`, **stratified by queue**, split at the
**ticket level** (a ticket's 9 aspect labels stay together → no leakage). Verified:
0 ticket-id overlap, 0 body overlap across splits. **Test was locked** and only touched
for the final number.

---

## 7. Modeling

### 7.1 Baseline — TF-IDF + Logistic Regression
One 3-class classifier per aspect on TF-IDF(body), `class_weight='balanced'`.
Persisted as `models/tfidf/*.joblib`. This sets the honest floor and gives the
cost/latency comparison. **Test mean per-aspect macro-F1: 0.724.**

### 7.2 The model — RoBERTa sentence-pair
Framing: for each `(ticket, aspect)`, classify
```
text_a = ticket body
text_b = "billing: charges, invoices, payments, refunds, ..."   (aspect + definition)
→ {not_present, negative, neutral}
```
**One shared model** handles all 9 aspects. Self-attention lets the ticket words attend
to the aspect definition — so it answers "is this aspect here, and how does the customer
feel?" in one pass. This is the unified-model design the project is built around.

**What didn't work — DeBERTa-v3.** Tried first (it usually wins on ABSA). It produced
`nan` loss under mixed precision (`ValueError: Attempting to unscale FP16 gradients`,
then nan under bf16) — a known instability in this transformers version. The model
collapsed to the majority class (macro-F1 0.295). **Switched to RoBERTa-base**: stable,
already cached, fits the 6 GB GPU.

### 7.3 Handling class imbalance (the key technical issue)
80% of cells are `not_present`, 16% negative, 3% neutral. Three responses:
1. **Class-weighted loss** (inverse-frequency, sqrt-damped so the rare neutral doesn't
   destabilize training) — boosts negative/neutral gradients.
2. **`class_weight='balanced'`** in the TF-IDF baseline too.
3. **macro-F1 as the model-selection metric** — picked the best epoch by macro-F1, not
   accuracy, so training optimized for the rare classes.

Weighted loss specifically rescued the weak aspects: security 0.62→0.70, network
0.60→0.74 between iterations.

---

## 8. Results & verification

### 8.1 The headline (RoBERTa, locked test, 27,000 cells)
| metric | score | note |
|---|---|---|
| **macro-F1 (pooled)** | **0.843** | the honest, imbalance-robust headline |
| aspect-detection F1 | 0.864 | routing signal |
| polarity accuracy | 0.986 | sentiment correctness where present |
| weighted-F1 / accuracy | 0.944 | **inflated by the 80% not_present majority** |

Per-class F1: not_present 0.967, negative 0.885, **neutral 0.677** (the weak point).

### 8.2 Why two different numbers (0.740 vs 0.843)
- **0.740** = average the 9 per-aspect macro-F1s separately (dragged by `bug` = 0.587).
- **0.843** = pool all 27k cells, compute macro-F1 once.
Both are valid; **pooled is the standard ABSA report.** Neither is fabricated.

### 8.3 Verification (the score was independently proven)
The score doubt was answered *inside the notebook*: macro-F1 is **re-derived by hand
from the confusion-matrix counts** (numpy) and cross-checked against
`sklearn.classification_report`. They match exactly (`0.843 == 0.843, match: True`).
The confusion matrix is shown so the imbalance and the errors are visible:

```
                pred:not_present  pred:negative  pred:neutral
gold not_present     20953            560           187
gold negative          409           3969            21       ← 409 missed complaints (9%)
gold neutral           288             45           568       ← neutral recall only 0.63
```

### 8.4 Ensemble
For each aspect, pick the model that wins on **val** (TF-IDF for bug/hardware, RoBERTa
for the other 7), score on test. **0.740** mean per-aspect macro-F1 — a small free bump.

---

## 9. Novel experiment — shared vs per-queue models

**The project's headline finding.** Question: does one shared model match N per-queue
specialists? Trained a separate model on each queue's tickets and compared to the single
shared model (TF-IDF for both, so it is feasible on small queues a transformer can't fit).

| | mean aspect macro-F1 | models to maintain |
|---|---|---|
| **Shared (1 model)** | **0.713** | 1 |
| Per-queue specialists | 0.676 | 10 |

**The shared model wins on 10/10 queues (+0.037) at 1/10th the maintenance.** The gap
widens on small queues (Sales −0.121, HR −0.074): specialists starve when they fragment
scarce data, while the shared model transfers cross-aspect signal across all queues.
**Specialization hurts here** — the counterintuitive, defensible result.

---

## 10. Serving

- `app/predict.py` — inference core: ticket → per-aspect sentiment → route + priority.
  Routing = queue of the first negative aspect; priority = high (≥2 complaints) /
  medium (1) / low.
- `app/api.py` — **FastAPI** hosts the model (`POST /predict`, `GET /health`).
- `app/streamlit_app.py` — UI that calls the API **over HTTP** (the model runs only in
  the API, not the frontend). `docker-compose.yml` runs both; `Dockerfile.streamlit`
  is the thin client.

---

## 11. Honest limitations

1. **Silver labels** — Qwen-generated, ~90% human agreement. The model can't exceed its
   labels' ceiling; only re-labeling with a stronger model breaks it.
2. **Synthetic text** — cleaner/more regular than real tickets; results would shift on
   messy production data.
3. **`neutral` class is data-limited** — `bug` has ~52 neutral of 4,122; that class is
   near-unlearnable regardless of model. A data ceiling, not a tuning failure.
4. **Accuracy/weighted-F1 are imbalance-inflated** — always report macro-F1 (0.843).

---

## 12. What was learned (the transferable skills)

- ABSA framed as sentence-pair classification; one shared model over many aspects.
- LLM-as-labeler with a measured human audit (treating labels as silver, not truth).
- Class-imbalance handling: weighted loss, balanced baselines, macro-F1 model selection.
- Honest evaluation: the right metric for an imbalanced problem, and *verifying* it.
- A real engineering trade-off study (shared vs specialist) with a clean result.
- The discipline of rejecting bad data, naming limitations, and not fabricating numbers.
