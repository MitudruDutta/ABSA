#!/usr/bin/env python3
"""
LLM aspect-sentiment labeler — local Qwen3-8B, 4-bit (nf4), thinking OFF.

For each ticket in label_pool.csv, asks the model to assign, for ALL 9 aspects,
one of: not_present | negative | neutral | positive. Output is strict JSON.

Tuned for a 6GB GPU: nf4 quant, batch=1, body truncated, short generation,
Qwen3 thinking mode disabled (enable_thinking=False) so it returns JSON directly
instead of <think>...</think> traces.

If 6GB OOMs on Qwen3-8B, set MODEL_ID = "Qwen/Qwen3-4B" (fits easily, faster).

Resumable: rows already filled (asp_billing non-empty) are skipped. Saves
incrementally so a crash/OOM doesn't lose progress.

Usage:
    python scripts/label_llm.py --limit 3      # smoke test (3 rows)
    python scripts/label_llm.py                # full run
"""

import argparse
import json
import os
import re
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from taxonomy import ASPECTS, ASPECT_DEFINITIONS, SENTIMENT_LABELS

POOL = os.path.join(ROOT, "data", "label_pool.csv")
OUT = os.path.join(ROOT, "data", "labeled_llm.csv")
MODEL_ID = "Qwen/Qwen3-4B"          # fits ~6GB GPU fully (8B needs broken CPU-offload path)

MAX_BODY_CHARS = 1200
MAX_NEW_TOKENS = 160
SAVE_EVERY = 20
VALID = set(SENTIMENT_LABELS)

DEFS = "\n".join(f"- {a}: {ASPECT_DEFINITIONS[a]}" for a in ASPECTS)
SYSTEM = (
    "You are an expert support-ticket annotator. Given a ticket, decide for "
    "EACH aspect whether it is mentioned and, if so, the customer's sentiment "
    "toward it.\n\n"
    f"Aspects and their scope:\n{DEFS}\n\n"
    "For every aspect output exactly one of: not_present, negative, neutral, "
    "positive.\n"
    "- not_present: the aspect is not referred to in the ticket.\n"
    "- negative: a problem, complaint, failure, error, outage, or dissatisfaction "
    "involving that aspect.\n"
    "- neutral: mentioned only as a factual statement, a question, or a REQUEST "
    "for something (e.g. asking for an enhancement, info, or change), with no "
    "complaint and no praise.\n"
    "- positive: the customer EXPLICITLY praises or expresses satisfaction that "
    "the aspect already works well.\n\n"
    "CRITICAL: support tickets are almost never positive. Words like 'enhance', "
    "'improve', 'optimize', 'scalability', 'upgrade', or wanting something to work "
    "better are NOT positive — they are a request (neutral) or, if a problem is "
    "described, negative. Only use positive for clear, explicit thanks/praise about "
    "something that is currently working. When unsure between positive and "
    "neutral/negative, do NOT pick positive.\n\n"
    "Respond with ONLY a JSON object mapping each of the 9 aspect names to its "
    "label. No prose, no markdown fences."
)

# Few-shot: teach the positive/neutral/negative boundary explicitly.
FEWSHOT = [
    {
        "role": "user",
        "content": (
            'Ticket:\n"""We request an enhancement to the system\'s scalability to '
            'prevent the recent service outages affecting our software and hardware."""\n\n'
            f"Return JSON with keys exactly: {ASPECTS}"
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "billing": "not_present", "account": "not_present",
            "performance": "negative", "security": "not_present",
            "bug": "not_present", "feature": "neutral",
            "network": "not_present", "hardware": "negative",
            "software": "negative",
        }),
    },
    {
        "role": "user",
        "content": (
            'Ticket:\n"""Thank you, the new dashboard is fast and the login works '
            'perfectly now. Could you share the API documentation?"""\n\n'
            f"Return JSON with keys exactly: {ASPECTS}"
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "billing": "not_present", "account": "positive",
            "performance": "positive", "security": "not_present",
            "bug": "not_present", "feature": "neutral",
            "network": "not_present", "hardware": "not_present",
            "software": "not_present",
        }),
    },
]


def build_messages(body):
    body = str(body)[:MAX_BODY_CHARS]
    user = (
        f'Ticket:\n"""{body}"""\n\n'
        f"Return JSON with keys exactly: {ASPECTS}"
    )
    return [{"role": "system", "content": SYSTEM}, *FEWSHOT,
            {"role": "user", "content": user}]


def parse(text):
    """Extract the JSON object, validate, fill missing as not_present."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for a in ASPECTS:
        v = str(obj.get(a, "not_present")).strip().lower()
        out[a] = v if v in VALID else "not_present"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="label only N rows (smoke test)")
    args = ap.parse_args()

    src = OUT if os.path.exists(OUT) else POOL
    df = pd.read_csv(src)
    # force label columns to string-object so empty cells are "" not NaN/float
    # (blank cells otherwise load as float64 -> dtype-assign FutureWarning).
    label_cols = [f"asp_{a}" for a in ASPECTS] + ["label_source", "notes"]
    for c in label_cols:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].astype("object").where(df[c].notna(), "").astype(str).replace("nan", "")

    todo = df.index[df["asp_billing"].str.len() == 0].tolist()
    if args.limit:
        todo = todo[: args.limit]
    print(f"source={src}  to_label={len(todo)} rows")
    if not todo:
        print("nothing to label."); return

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"loading {MODEL_ID} (4-bit, thinking OFF)...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map={"": 0},
        dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    model.eval()

    done = 0
    for i in todo:
        msgs = build_messages(df.at[i, "body"])
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,          # Qwen3: direct answer, no <think>
        )
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1600).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        res = parse(text)
        if res is None:
            df.at[i, "notes"] = "PARSE_FAIL"
            for a in ASPECTS:
                df.at[i, f"asp_{a}"] = "not_present"
        else:
            for a in ASPECTS:
                df.at[i, f"asp_{a}"] = res[a]
        df.at[i, "label_source"] = "llm-qwen3-4b-4bit"
        done += 1
        if done % SAVE_EVERY == 0:
            df.to_csv(OUT, index=False)
            print(f"  {done}/{len(todo)} saved")

    df.to_csv(OUT, index=False)
    print(f"\nDONE. labeled {done} -> {OUT}")
    labeled = df[df["label_source"].str.len() > 0]
    print(f"\nlabel distribution over {len(labeled)} labeled rows (present aspects only):")
    for a in ASPECTS:
        vc = labeled[f"asp_{a}"].value_counts()
        present = {k: int(v) for k, v in vc.items() if k not in ("", "not_present")}
        print(f"  {a:12s} present={sum(present.values()):3d}  {present}")


if __name__ == "__main__":
    main()
