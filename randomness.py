"""Randomness-consistency loss.

The model's training distribution causes random-number bias: it
reproduces human biases from text. A decode-time uniform sampler only
hides the symptom. A small policy-gradient update on a randomness
objective flattens the distribution and transfers across tasks.

This module embeds that as an auxiliary loss during training. It infers
a valid answer set from the prompt and pushes the model's distribution
toward uniform over that set.
"""
from __future__ import annotations

import re
from typing import Optional

import torch
import torch.nn.functional as F

_RANGE_RE = re.compile(r"between\s+(\d+)\s+and\s+(\d+)", re.IGNORECASE)
_ONE_TO_RE = re.compile(r"1\s*(?:-|to|through)\s*(\d+)", re.IGNORECASE)
_DIGIT_RE = re.compile(r"\b(digit|number|integer)\b", re.IGNORECASE)
_COLOR_RE = re.compile(r"\b(colou?r)\b", re.IGNORECASE)

COLORS = ["red", "blue", "green", "yellow", "purple", "orange",
          "pink", "brown", "black", "white", "gray", "cyan"]


def extract_valid_strings(prompt_text: str) -> Optional[list[str]]:
    if not prompt_text:
        return None
    t = prompt_text.lower()
    m = _RANGE_RE.search(t) or _ONE_TO_RE.search(t)
    if m:
        try:
            hi = int(m.group(1))
            return [str(i) for i in range(1, hi + 1)]
        except Exception:
            pass
    if _COLOR_RE.search(t):
        return COLORS
    return None


def strings_to_token_ids(strings: list[str], tokenizer, max_per_str: int = 4) -> list[int]:
    ids: set[int] = set()
    for s in strings:
        for variant in (s, " " + s):
            try:
                enc = tokenizer.encode(variant)
            except Exception:
                continue
            ids.update(int(x) for x in enc[:max_per_str])
    return sorted(ids)


def randomness_consistency_loss(
    logits: torch.Tensor,
    valid_token_ids: list[int],
    weight: float = 0.05,
) -> torch.Tensor:
    """KL(uniform || softmax(logits[valid])) scaled by weight. Applied per
    position; expects logits shape [V]."""
    if not valid_token_ids:
        return logits.new_zeros(())
    device = logits.device
    idx = torch.tensor(valid_token_ids, device=device, dtype=torch.long)
    valid_logits = logits.index_select(-1, idx)
    log_p = F.log_softmax(valid_logits, dim=-1)
    n = len(valid_token_ids)
    uniform = -torch.log(torch.tensor(float(n), device=device))
    kl = uniform - log_p.mean()
    return weight * kl


def should_apply_randomness_loss(prompt_text: str) -> bool:
    from .sampler import detect_randomness_request
    return detect_randomness_request(prompt_text) and extract_valid_strings(prompt_text) is not None