"""Sampling: provably unbiased selection.  Inverse transform for uniform
modes; temperature calibration for entropy modes.  Both function API and a
backward-compatible ``Sampler`` class.

detect_randomness_request() + resolve_sampling_mode() close a gap that
previously existed between the uniform/semantic_uniform modes (which were
implemented correctly) and actually using them: nothing in model.generate()
or refinement.refine() ever passed mode="uniform", so a prompt literally
asking for "a random number" still went through the model's ordinary
logit distribution -- which is exactly the dataset-frequency bias the
original requirement was about. The two helpers below let a caller route
on the PROMPT TEXT automatically instead of requiring every call site to
remember to ask for uniform sampling by hand."""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F

_RANDOMNESS_RE = re.compile(
    r"\b(random(ly)?|pick\s+(one|any)|choose\s+(one|any)|"
    r"roll\s+a\s+die|coin\s+flip|arbitrar(y|ily))\b",
    re.IGNORECASE,
)


def detect_randomness_request(prompt_text: str) -> bool:
    """Heuristic, text-level detector for 'give me a random X' style
    requests. Deliberately conservative (keyword/phrase based, not a
    learned classifier) so it's auditable and fails safe: false negatives
    just mean ordinary model-distribution sampling is used (today's
    status quo), false positives mean a request gets uniform sampling it
    didn't strictly need, which is a much smaller cost than silently
    staying biased on a request that explicitly asked not to be."""
    if not prompt_text:
        return False
    return bool(_RANDOMNESS_RE.search(prompt_text))


def resolve_sampling_mode(prompt_text: str, requested_mode: str | None = None) -> str:
    """What mode model.generate() should actually use. An explicit
    requested_mode always wins (never override a caller's deliberate
    choice); otherwise auto-detect a randomness request from the prompt
    text and switch to semantic_uniform (stays within plausible/on-topic
    candidates via top-k, but picks among them with flat probability
    instead of the model's learned, dataset-frequency-biased weights) --
    a better default than fully flat 'uniform' over the whole vocabulary,
    which would ignore the prompt's topic/type entirely (e.g. return
    token garbage instead of a plausible-looking number or color)."""
    if requested_mode is not None:
        return requested_mode
    if detect_randomness_request(prompt_text):
        return "semantic_uniform"
    return "model"


def compute_entropy(probs):
    p = probs.clamp_min(1e-12)
    return float(-(p * p.log()).sum())


def calibrate_temperature(logits, target_entropy, iters=50):
    lo, hi = 0.02, 12.0
    if compute_entropy(F.softmax(logits / hi, dim=-1)) <= target_entropy:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ent = compute_entropy(F.softmax(logits / mid, dim=-1))
        if ent < target_entropy:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def inverse_transform_sample(probs, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = float(torch.rand(1, generator=g).item())
    cdf = torch.cumsum(probs, dim=0).tolist()
    for i, c in enumerate(cdf):
        if u <= c:
            return i
    return len(cdf) - 1


def semantic_uniform_seeded(logits, top_k=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(logits.shape, generator=g).clamp_min(1e-6)
    noise = -torch.log(-torch.log(u))
    perturbed = logits + 0.1 * noise
    cand = torch.topk(perturbed, min(top_k, logits.numel())).indices
    return int(cand[inverse_transform_sample(
        torch.full((cand.numel(),), 1.0 / cand.numel()), seed=seed + 1
    )])


class Sampler:
    def __init__(self, seed=0):
        self.seed = seed

    def sample(self, logits, mode="model", temperature=1.0, top_k=None,
               target_entropy=None, seed=None):
        s = self.seed if seed is None else seed
        if mode == "greedy":
            tok = int(logits.argmax())
            probs = F.softmax(logits, dim=-1)
        elif mode == "model":
            probs = F.softmax(logits / max(temperature, 1e-5), dim=-1)
            tok = inverse_transform_sample(probs, seed=s)
        elif mode == "uniform":
            V = logits.numel()
            tok = inverse_transform_sample(torch.full((V,), 1.0 / V), seed=s)
            probs = torch.full((V,), 1.0 / V)
        elif mode == "semantic_uniform":
            tok = semantic_uniform_seeded(logits, top_k or 10, seed=s)
            probs = F.softmax(logits, dim=-1)
        elif mode == "entropy_calibrated":
            T = calibrate_temperature(logits, target_entropy or 2.0)
            probs = F.softmax(logits / T, dim=-1)
            tok = inverse_transform_sample(probs, seed=s)
        else:
            raise ValueError(f"unknown mode: {mode}")
        return {"tokens": torch.tensor([tok]), "probs": probs,
                "entropy": compute_entropy(probs), "mode": mode}