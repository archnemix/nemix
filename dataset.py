"""Curate a high-quality Python dataset for NEMIX-v2."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Iterable

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


def _safe_get(ex, *keys, default=""):
    for k in keys:
        if k in ex and ex[k] is not None:
            return ex[k]
    return default


def _execute(code, tests, timeout=5):
    if not tests:
        return False
    src = code + "\n" + "\n".join(tests) + "\n"
    try:
        r = subprocess.run(
            [sys.executable, "-c", src],
            capture_output=True, timeout=timeout, text=True,
        )
        return r.returncode == 0 and "Traceback" not in r.stderr
    except subprocess.TimeoutExpired:
        return False


def _from_mbpp(ex):
    """MBPP schema: task_id, text, code, test_list, test_setup_code, ..."""
    text = ex.get("text", "")
    code = ex.get("code", "")
    if not text or not code:
        return None
    prompt = f"# {text.strip()}\n"
    target = code.strip() + "\n"
    return {"prompt": prompt, "target": target, "tests": ex.get("test_list", [])}


def _from_opencodeinstruct(ex, verify: bool):
    """Defensive getter; field names vary across versions."""
    if ex.get("language") != "python":
        return None
    prob = _safe_get(ex, "problem", "input", "prompt")
    sol = _safe_get(ex, "solution", "output", "code")
    tests = _safe_get(ex, "tests", "test_list", default=[])
    if not prob or not sol:
        return None
    if verify and tests and not _execute(sol, tests):
        return None
    return {
        "prompt": f"# {prob.strip()}\n",
        "target": sol.strip() + "\n",
        "tests": tests,
    }


def build_dataset(tokenizer_path: str,
                  opencode_limit: int = 200_000,
                  verify: bool = True,
                  out_path: str = "python_train.jsonl"):
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    records = []
    for ex in load_dataset("google-research-datasets/mbpp", "full", split="train"):
        rec = _from_mbpp(ex)
        if rec:
            records.append(rec)
    print(f"MBPP: {len(records)} records")

    if opencode_limit > 0:
        stream = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)
        added = 0
        for ex in stream:
            rec = _from_opencodeinstruct(ex, verify)
            if rec is None:
                continue
            records.append(rec)
            added += 1
            if added >= opencode_limit:
                break
        print(f"OpenCodeInstruct: {added} added")

    def encode(rec):
        prompt_ids = tok(rec["prompt"], truncation=True, max_length=256).input_ids
        target_ids = tok(rec["target"], truncation=True, max_length=512).input_ids
        return {"prompt_ids": prompt_ids, "target_ids": target_ids}

    ds = Dataset.from_list(records).map(encode, remove_columns=list(records[0].keys()),
                                         keep_in_memory=True)
    with open(out_path, "w") as f:
        for i in range(len(ds)):
            f.write(json.dumps({
                "prompt_ids": ds[i]["prompt_ids"],
                "target_ids": ds[i]["target_ids"],
            }) + "\n")
    print(f"wrote {len(ds)} examples to {out_path}")
    return out_path


def load_training_pairs(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f]