# Process Documentation

A phase-by-phase account of how this project was built - every major decision, the
reasoning behind it, and the approaches that did not work. Read in order, or jump to a
phase.

| # | phase | what it covers |
|---|---|---|
| 01 | [Problem & Taxonomy](01-problem-and-taxonomy.md) | the problem, scope guardrail, how the 9 aspects were derived, why the positive class was dropped |
| 02 | [Data](02-data.md) | dataset selection (and what was rejected, with reasons), cleaning, the synthetic-text caveat |
| 03 | [Labeling](03-labeling.md) | local Qwen3-4B labeling, the 8B to 4B pivot, the prompt, scaling 600 to 20k, reboot survival |
| 04 | [Audit](04-audit.md) | blind human audit method, 90.3% agreement / Cohen's kappa 0.71, the over-tagging finding |
| 05 | [Modeling](05-modeling.md) | split, TF-IDF baseline, RoBERTa sentence-pair, the DeBERTa nan failure, class-imbalance handling |
| 06 | [Results & Experiments](06-results-and-experiments.md) | metrics + the manual-vs-sklearn verification, 0.740 vs 0.843, ensemble, shared-vs-per-queue |
| 07 | [Serving](07-serving.md) | inference core, FastAPI, Streamlit-over-HTTP, Docker, persisted artifacts |

## At a glance

- **Task:** Aspect-Based Sentiment Analysis on support tickets - per-aspect sentiment,
  then routing + priority.
- **Data:** 24,934 cleaned English tickets; 20,000 labeled with Qwen3-4B (silver,
  90.3% human-audited).
- **Model:** RoBERTa sentence-pair, one shared model over 9 aspects, class-weighted loss.
- **Headline (held-out test):** macro-F1 **0.843**, aspect-detection F1 0.864, polarity
  accuracy 0.986. macro-F1 is the imbalance-robust number; accuracy/weighted-F1 (0.944)
  are inflated by the 80% not_present majority.
- **Novel result:** one shared model beats 10 per-queue specialists on 10/10 queues
  (+0.037) at 1/10th the maintenance.

For the runnable pipeline see [`../notebooks/`](../notebooks); for the top-level
overview see [`../README.md`](../README.md).
