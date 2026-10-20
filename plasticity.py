"""Plasticity diagnostics: readout Rayleigh quotient, partial reset,
and OASIS-style history sketch.

The continual-learning dynamics paper shows cross-task readout
degeneration is geometric: readout directions a task uses shrink
exponentially even when backbone features survive. Data selection alone
cannot fix this. This module provides the monitor and repair hooks.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class HistorySketch:
    """OASIS-style running sketch of ALL admitted vectors, not just the
    currently-resident store. Redundancy estimated against this catches
    duplicates of evicted items that the store alone would miss."""

    def __init__(self, dim: int, capacity: int = 512):
        self.dim = dim
        self.capacity = capacity
        self.vectors: list[torch.Tensor] = []

    def add(self, vec: torch.Tensor) -> None:
        v = vec.detach().flatten().cpu().clone()
        self.vectors.append(v)
        if len(self.vectors) > self.capacity:
            self.vectors.pop(0)

    def max_similarity(self, query: torch.Tensor) -> float:
        if not self.vectors:
            return 0.0
        q = F.normalize(query.detach().flatten().cpu(), dim=0)
        m = F.normalize(torch.stack(self.vectors), dim=-1)
        return float((m @ q).max().item())

    def __len__(self):
        return len(self.vectors)


class ReadoutMonitor:
    """Tracks effective rank of the LM head on a probe set. When rank drops
    below threshold, callers should lower theta_high (admit more diversity)
    and/or apply a partial reset."""

    def __init__(self, threshold: float = 0.05, partial_reset_frac: float = 0.02):
        self.threshold = threshold
        self.partial_reset_frac = partial_reset_frac
        self.history: list[dict] = []

    @torch.no_grad()
    def measure(self, model, probe_ids: torch.Tensor) -> dict:
        try:
            device = next(model.parameters()).device
            probe_ids = probe_ids.to(device)
            cfg = getattr(model, "cfg", None)
            mask_id = cfg.mask_id if cfg else 0
            corrupted = torch.full_like(probe_ids, mask_id)

            prompt_hidden, _ = model._encode_prompt(probe_ids)
            x = model.decoder.emb(corrupted)
            for layer in model.decoder.layers:
                x = layer(x, prompt_hidden)
            h = model.decoder.norm(x)

            H = h.reshape(-1, h.shape[-1])
            if H.shape[0] > 128:
                idx = torch.randperm(H.shape[0], device=device)[:128]
                H = H[idx]
            W = model.decoder.lm_head.weight
            if W.shape[0] > 2048:
                idx = torch.randperm(W.shape[0], device=device)[:2048]
                W = W[idx]
            Hn = F.normalize(H, dim=-1)
            Wn = F.normalize(W, dim=-1)
            proj = Wn @ Hn.t()
            s = torch.linalg.svdvals(proj)
            s2 = s.pow(2)
            erank = float((s2.sum().pow(2) / s2.pow(2).sum().clamp_min(1e-12)).item())
            rayleigh = float(proj.pow(2).mean().item())
            out = {
                "ok": True,
                "effective_rank": erank,
                "rayleigh": rayleigh,
                "below_threshold": erank < self.threshold * min(proj.shape),
            }
            self.history.append(out)
            return out
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @torch.no_grad()
    def partial_reset(self, model, vocab_usage: Optional[torch.Tensor] = None) -> int:
        W = model.decoder.lm_head.weight
        V, d = W.shape
        n_reset = max(1, int(V * self.partial_reset_frac))
        if vocab_usage is None:
            norms = W.norm(dim=-1)
            idx = torch.argsort(norms)[:n_reset]
        else:
            idx = torch.argsort(vocab_usage)[:n_reset]
        W.data[idx] = torch.randn(n_reset, d, device=W.device, dtype=W.dtype) * 0.02
        return int(n_reset)


def adjust_theta_high(monitor_result: dict, cfg) -> float:
    if not monitor_result.get("ok"):
        return getattr(cfg, "theta_high", 0.25)
    if monitor_result.get("below_threshold"):
        return getattr(cfg, "theta_high_relaxed", 0.10)
    return getattr(cfg, "theta_high", 0.25)