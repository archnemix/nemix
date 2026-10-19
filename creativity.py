"""Creativity: novelty pressure, entropy-targeted sampling, distant recombination."""
from __future__ import annotations

import random
import torch
import torch.nn.functional as F

from .sampler import Sampler, compute_entropy


def novelty_reweight(logits, embed_weight, memory_vectors, lam=0.5):
    if memory_vectors is None or memory_vectors.numel() == 0:
        return logits
    w = F.normalize(embed_weight, dim=-1)
    m = F.normalize(memory_vectors, dim=-1)
    sim = (w @ m.t()).max(dim=-1).values
    return logits - lam * sim


def entropy_calibrated_sample(logits, target_entropy, seed=None, base_seed=0):
    s = Sampler(seed=base_seed)
    return s.sample(logits, mode="entropy_calibrated",
                    target_entropy=target_entropy, seed=seed)


def _dist(a, b):
    return float(torch.norm(a - b))


def recombine(memory_vectors, n=2, seed=0):
    M = memory_vectors.shape[0]
    if M == 0:
        raise ValueError("memory is empty")
    n = min(n, M)
    rng = random.Random(seed)
    if n == 1:
        pick = [rng.randrange(M)]
    else:
        pick = [rng.randrange(M)]
        while len(pick) < n:
            best, best_d = None, -1.0
            for i in range(M):
                if i in pick:
                    continue
                d = min(_dist(memory_vectors[i], memory_vectors[j]) for j in pick)
                if d > best_d:
                    best, best_d = i, d
            pick.append(best)
    pts = memory_vectors[pick]
    return {
        "indices": pick,
        "combined": pts.mean(dim=0),
        "pairwise_distance": _dist(pts[0], pts[1]) if len(pick) > 1 else 0.0,
    }


def divergence_ratio(logits_before, logits_after):
    pb = torch.softmax(logits_before, dim=-1)
    pa = torch.softmax(logits_after, dim=-1)
    kl = (pa * (pa.clamp_min(1e-12).log() - pb.clamp_min(1e-12).log())).sum()
    return float(kl)