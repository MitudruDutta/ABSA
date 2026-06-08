"""
Inference core: ticket text -> per-aspect sentiment + routing + priority.

Loads the trained RoBERTa sentence-pair model once and scores all 9 aspects.
Used by both the FastAPI service and the Streamlit demo.
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from taxonomy import ASPECTS, ASPECT_DEFINITIONS

MODELDIR = os.path.join(ROOT, "models", "roberta-absa")
ID2LABEL = {0: "not_present", 1: "negative", 2: "neutral"}
HYP = {a: f"{a}: {ASPECT_DEFINITIONS[a]}" for a in ASPECTS}

# aspect -> the queue a ticket about that aspect should route to
ASPECT_TO_QUEUE = {
    "billing": "Billing and Payments",
    "account": "IT Support",
    "performance": "Service Outages and Maintenance",
    "security": "IT Support",
    "bug": "Technical Support",
    "feature": "Product Support",
    "network": "IT Support",
    "hardware": "Technical Support",
    "software": "Technical Support",
}

_tok = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load():
    global _tok, _model
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(MODELDIR)
        _model = AutoModelForSequenceClassification.from_pretrained(MODELDIR).to(_device).eval()
    return _tok, _model


def predict(body: str):
    """Return dict: per-aspect sentiment, detected aspects, route, priority."""
    tok, model = _load()
    pairs_b = [HYP[a] for a in ASPECTS]
    enc = tok([body] * len(ASPECTS), pairs_b, truncation=True, max_length=256,
              padding=True, return_tensors="pt").to(_device)
    with torch.no_grad():
        logits = model(**enc).logits
    preds = logits.argmax(-1).cpu().tolist()

    aspects = {a: ID2LABEL[p] for a, p in zip(ASPECTS, preds)}
    present = {a: s for a, s in aspects.items() if s != "not_present"}
    negatives = [a for a, s in present.items() if s == "negative"]

    # routing: route to the queue of the most-negative aspect (first by taxonomy
    # order); fall back to any present aspect, else General Inquiry.
    route = "General Inquiry"
    for a in ASPECTS:
        if a in negatives:
            route = ASPECT_TO_QUEUE[a]; break
    else:
        for a in ASPECTS:
            if a in present:
                route = ASPECT_TO_QUEUE[a]; break

    # priority: more negatives -> higher
    n_neg = len(negatives)
    priority = "high" if n_neg >= 2 else ("medium" if n_neg == 1 else "low")

    return {
        "aspects": aspects,            # all 9
        "present": present,            # only mentioned
        "negatives": negatives,
        "route": route,
        "priority": priority,
    }


if __name__ == "__main__":
    import json
    demo = "I was overcharged on my invoice and the app keeps crashing during checkout."
    print(json.dumps(predict(demo), indent=2))
