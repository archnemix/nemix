"""Execution-grounded evaluation on HumanEval and MBPP."""
from __future__ import annotations

import subprocess
import sys
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def _execute(code: str, test: str, timeout=5) -> bool:
    try:
        r = subprocess.run(
            [sys.executable, "-c", code + "\n" + test],
            capture_output=True, timeout=timeout, text=True,
        )
        return r.returncode == 0 and "Traceback" not in r.stderr
    except subprocess.TimeoutExpired:
        return False


def pass_at_k(model, tokenizer, problems, k=10, device="cpu", out_len=256, **gen_kw):
    total = 0.0
    for prob in problems:
        prompt_text = prob["prompt"]
        tests = prob.get("test") or prob.get("tests") or []
        if isinstance(tests, str):
            tests = [tests]
        prompt_ids = torch.tensor(tokenizer.encode(prompt_text), device=device)
        succeeded = False
        for _ in range(k):
            out = model.generate(prompt_ids, out_len=out_len, **gen_kw)
            tokens = out["tokens"]
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            code = prompt_text + tokenizer.decode(tokens)
            if all(_execute(code, t) for t in tests):
                succeeded = True
                break
        total += 1.0 if succeeded else 0.0
    return total / len(problems)


def s_star_select(model, tokenizer, prompt, tests, n_drafts=8, device="cpu", **gen_kw):
    prompt_ids = torch.tensor(tokenizer.encode(prompt), device=device)
    last = None
    for _ in range(n_drafts):
        out = model.generate(prompt_ids, out_len=256, **gen_kw)
        tokens = out["tokens"]
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        code = prompt + tokenizer.decode(tokens)
        last = code
        if all(_execute(code, t) for t in tests):
            return code, True
    return last, False


def evaluate(model, cfg, name="nemix", k=10, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    mbpp = load_dataset("google-research-datasets/mbpp", "full", split="test")
    try:
        humaneval = load_dataset("openai/openai_humaneval", split="test")
    except Exception:
        humaneval = []
    print(f"{name} MBPP pass@{k}: {pass_at_k(model, tok, list(mbpp), k=k, device=device):.3f}")
    if humaneval:
        print(f"{name} HumanEval pass@{k}: {pass_at_k(model, tok, list(humaneval), k=k, device=device):.3f}")