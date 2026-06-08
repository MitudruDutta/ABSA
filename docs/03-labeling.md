# Phase 3 — LLM Labeling

[← 02 Data](02-data.md) · [docs index](README.md) · next: [04 Audit →](04-audit.md)

---

This is the **critical path**. No off-the-shelf dataset has aspect-level sentiment
labels — they had to be generated. The whole project waits on this step.

## 3.1 Engine choice (and the constraints that forced it)

- **No LLM API keys** available → must use a **local model** on an RTX 4050 (6 GB).
- **Qwen3-8B (tried first):**
 - OOM on load — only ~4.4 GB VRAM free (the laptop display eats ~1.2 GB), and the
 8B 4-bit weights are ~4.5 GB.
 - Attempted CPU-offload of overflow layers → hit a library bug:
 `TypeError: Params4bit.__new__() got an unexpected keyword argument '_is_hf_initialized'`
 (bitsandbytes 4-bit + accelerate device-split is broken in this transformers
 version).
- **Qwen3-4B (chosen):** fits fully on the GPU (4-bit nf4, ~2.5 GB). Set
 `enable_thinking=False` so it emits clean JSON instead of `<think>…</think>` traces.

The 8B model (16 GB download) was deleted from disk once 4B was adopted.

## 3.2 The prompt

A system prompt containing:
- the 9 aspect definitions,
- the 3-class rule, with an **explicit guard**: requests/inquiries ("enhance",
 "optimize", "scalability") are **neutral, not positive**,
- **2 few-shot examples** teaching the negative/neutral/positive boundary.

Output is parsed as a strict JSON object mapping each aspect → label; anything invalid
or unparseable falls back to `not_present`. (The `positive` class was later folded into
`neutral` — see [phase 1](01-problem-and-taxonomy.md#14-sentiment-label-space--3-classes-per-aspect).)

## 3.3 Robustness — resumable labeling

Support-ticket labeling at 20k scale on a laptop GPU is slow (~0.3–1.3 tickets/sec, with
thermal throttling) and the machine **rebooted ~6 times** during the run. The labeler:
- saves every 20 rows,
- on restart, skips already-labeled rows and resumes.

Zero progress was lost across all reboots — verified: 0 duplicates, 0 gaps, every row
processed exactly once.

## 3.4 Scaling strategy

Labeled in stages, each driven by evidence:

| stage | tickets | trigger |
|---|---|---|
| pilot | 600 | validate the prompt + pipeline |
| v1 | 2,500 | first trainable set |
| **final** | **20,000** | rare aspects were starving (e.g. `account` 165 present @2.5k) |

At 20k the rare aspects became viable: `account` 165 → **1,007**, `billing` 222 →
**939**, `network` 354 → **2,463**, `hardware` 414 → **2,482**.

## 3.5 Final label integrity (verified)

| check | result |
|---|---|
| total rows | 20,000 |
| duplicate ticket-ids / bodies | 0 / 0 |
| invalid label cells | 0 |
| `positive` leftover | 0 (fold held across all 20k) |
| parse failures | 0 |
| avg aspects per ticket | 1.76 |
| tickets with 0 aspects | 3,997 (20%) — genuinely aspectless (HR/general) + some recall misses |

Labels are **silver, not gold** — quality measured next in [phase 4](04-audit.md).

---

**Artifacts:** `data/labeled_llm.csv` (20,000 tickets, 3-class per aspect,
`label_source = llm-qwen3-4b-4bit`). Full guard-wrapped labeler code lives in
`notebooks/01_data_and_labeling.ipynb`.
