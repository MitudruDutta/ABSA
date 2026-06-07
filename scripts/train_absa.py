#!/usr/bin/env python3
"""
DeBERTa-v3 sentence-pair ABSA — the real model.

Framing: for each ticket x aspect, classify
    text_a = ticket body
    text_b = aspect name + scope definition
  -> {not_present, negative, neutral}

ONE unified model handles all 9 aspects (the shared-model design). Train set
expands to 1750 x 9 = 15,750 pairs; val 375 x 9.

Tuned for ~4-6GB GPU: deberta-v3-base, fp16, batch 8 + grad-accum, max_len 256.
Reports overall + per-aspect macro-F1 on VAL; compare to TF-IDF baseline 0.677.
Test set stays locked.
"""

import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from taxonomy import ASPECTS, ASPECT_DEFINITIONS

D = os.path.join(ROOT, "data")
OUTDIR = os.path.join(ROOT, "models", "roberta-absa")
MODEL_ID = "roberta-base"   # deberta-v3 gave nan loss under mixed precision; roberta is stable + cached
MAXLEN = 256

LABEL2ID = {"not_present": 0, "negative": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
HYP = {a: f"{a}: {ASPECT_DEFINITIONS[a]}" for a in ASPECTS}


def expand(df):
    """ticket rows -> (body, aspect_hypothesis, label, aspect) pairs."""
    rows = []
    for _, r in df.iterrows():
        for a in ASPECTS:
            lab = str(r[f"asp_{a}"]) if str(r[f"asp_{a}"]) in LABEL2ID else "not_present"
            rows.append({"text_a": str(r["body"]), "text_b": HYP[a],
                         "label": LABEL2ID[lab], "aspect": a})
    return pd.DataFrame(rows)


def main():
    tr = expand(pd.read_csv(os.path.join(D, "train.csv")))
    va = expand(pd.read_csv(os.path.join(D, "val.csv")))
    print(f"train pairs={len(tr)}  val pairs={len(va)}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    def tok_fn(b):
        return tok(b["text_a"], b["text_b"], truncation=True, max_length=MAXLEN)

    dtr = Dataset.from_pandas(tr).map(tok_fn, batched=True,
                                      remove_columns=["text_a", "text_b"])
    dva = Dataset.from_pandas(va).map(tok_fn, batched=True,
                                      remove_columns=["text_a", "text_b"])
    va_aspect = np.array(va["aspect"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID)

    def metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        out = {"macro_f1": f1_score(labels, preds, average="macro", zero_division=0)}
        # per-aspect macro-F1, then mean (comparable to baseline 0.677)
        per = []
        for a in ASPECTS:
            m = va_aspect == a
            per.append(f1_score(labels[m], preds[m], average="macro", zero_division=0))
        out["mean_aspect_macro_f1"] = float(np.mean(per))
        return out

    # class weights from pooled train labels (boost rare negative/neutral vs
    # dominant not_present); sqrt-dampened so the very rare neutral doesn't
    # destabilize training.
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.array([0, 1, 2])
    cw = compute_class_weight("balanced", classes=classes, y=tr["label"].values)
    cw = np.sqrt(cw)                      # dampen
    cw = cw / cw.mean()
    class_weights = torch.tensor(cw, dtype=torch.float32)
    print(f"class weights (not_present/neg/neu): {cw.round(3)}")

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels,
                weight=class_weights.to(outputs.logits.device))
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=OUTDIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=3e-5,
        num_train_epochs=5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,          # deberta-v3 breaks under fp16 grad-scaler; bf16 is stable on Ampere
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mean_aspect_macro_f1",
        logging_steps=200,
        report_to="none",
        save_total_limit=1,
    )
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=dtr, eval_dataset=dva,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics,
    )
    # auto-resume from last checkpoint if a previous run was interrupted (reboots)
    import glob
    ckpts = glob.glob(os.path.join(OUTDIR, "checkpoint-*"))
    resume = bool(ckpts)
    print(f"resuming from checkpoint: {resume}")
    trainer.train(resume_from_checkpoint=resume)
    res = trainer.evaluate()
    print("\n=== FINAL (VAL) ===")
    print(f"  overall macro-F1        : {res['eval_macro_f1']:.3f}")
    print(f"  mean aspect macro-F1    : {res['eval_mean_aspect_macro_f1']:.3f}")
    print(f"  TF-IDF baseline (val)   : 0.677")
    trainer.save_model(OUTDIR)
    tok.save_pretrained(OUTDIR)
    print(f"\nsaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
