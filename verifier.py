"""Verifiers: arithmetic, confidence, and code execution."""
from __future__ import annotations

import re
import subprocess
import sys
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
    """Run every test; return ok + score."""
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