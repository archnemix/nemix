"""Verifiers: arithmetic, confidence, and code execution.

Each verifier below keeps its original signature (so existing code that
calls e.g. arithmetic_verifier(prompt_text, out_text) directly still
works), and each has a *token-adapter* factory (make_*_verifier) that
returns a callable(tokens) -> dict suitable for deliberation.deliberate
and refinement.refine, which both invoke verifier(tokens) on a raw token
tensor. Previously those built-ins could not be plugged into either
function without a TypeError on the first draft.
"""
from __future__ import annotations

import re
import subprocess
import sys
from typing import Callable, Optional

import torch

_EQ_RE = re.compile(r"(\d+)\s*\+\s*(\d+)\s*=")


def arithmetic_verifier(prompt_text, out_text):
    m = _EQ_RE.search(prompt_text or "")
    out = (out_text or "").strip()
    if not m:
        return {"ok": False, "bad": list(range(len(out) or 1)),
                "reason": "no equation", "score": 0.0}
    expected = int(m.group(1)) + int(m.group(2))
    if out.isdigit() and int(out) == expected:
        return {"ok": True, "bad": [], "expected": expected, "score": 1.0}
    exp_s = str(expected)
    bad = [i for i, ch in enumerate(out) if i >= len(exp_s) or ch != exp_s[i]]
    if not bad:
        bad = list(range(len(out)))
    return {"ok": False, "bad": bad, "expected": expected,
            "score": 1.0 - len(bad) / max(len(out), 1)}


def confidence_verifier(probs, tokens, thr=0.35):
    p = probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    bad = (p < thr).nonzero(as_tuple=False).squeeze(-1).tolist()
    if isinstance(bad, int):
        bad = [bad]
    return {"ok": len(bad) == 0, "bad": bad, "score": float(p.mean())}


def code_execution_verifier(code, tests, timeout=5):
    if not tests:
        return {"ok": False, "bad": [0], "score": 0.0, "error": "no tests"}
    for test in tests:
        try:
            r = subprocess.run(
                [sys.executable, "-c", code + "\n" + test],
                capture_output=True, timeout=timeout, text=True,
            )
            if r.returncode != 0 or "Traceback" in r.stderr:
                return {"ok": False, "bad": [0], "score": 0.0, "error": r.stderr}
        except subprocess.TimeoutExpired:
            return {"ok": False, "bad": [0], "score": 0.0, "error": "timeout"}
    return {"ok": True, "bad": [], "score": 1.0}


def pick_best(candidates):
    return max(candidates,
               key=lambda c: (c.get("ok") is True, c.get("score", 0.0)))


# ---------------------------------------------------------------------------
# Token adapters: turn a raw token tensor into the shape each verifier wants.
# ---------------------------------------------------------------------------
def _tokens_to_text(tokens, tokenizer) -> str:
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.detach().flatten().tolist()
    try:
        return tokenizer.decode([int(t) for t in tokens])
    except Exception:
        return ""


def make_arithmetic_verifier(prompt_text: str, tokenizer) -> Callable:
    """Wrap arithmetic_verifier for use as verifier(tokens)."""
    def _verifier(tokens):
        return arithmetic_verifier(prompt_text, _tokens_to_text(tokens, tokenizer))
    return _verifier


def make_code_verifier(prompt_text: str, tests, tokenizer,
                       timeout: int = 5) -> Callable:
    """Wrap code_execution_verifier for use as verifier(tokens)."""
    def _verifier(tokens):
        code = prompt_text + _tokens_to_text(tokens, tokenizer)
        return code_execution_verifier(code, tests, timeout=timeout)
    return _verifier


def make_confidence_verifier(probs: torch.Tensor, thr: float = 0.35) -> Callable:
    """Wrap confidence_verifier for use as verifier(tokens).

    probs must be the per-position [T, V] distribution the draft was drawn
    from; the adapter truncates it to len(tokens) positions.
    """
    def _verifier(tokens):
        if isinstance(tokens, torch.Tensor):
            t = tokens.detach().flatten().to(dtype=torch.long)
        else:
            t = torch.tensor(list(tokens), dtype=torch.long)
        n = min(probs.shape[0], t.numel())
        return confidence_verifier(probs[:n], t[:n], thr=thr)
    return _verifier