"""MEGa: gated low-rank adapters.  One memory = one key + N layer-adapters
that share one gate weight."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MEGaAdapter(nn.Module):
    def __init__(self, d, rank):
        super().__init__()
        self.A = nn.Linear(d, rank, bias=False)
        self.B = nn.Linear(rank, d, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.B(self.A(x))


class _Memory(nn.Module):
    """One memory: single key, one adapter per target layer, shared gate."""

    def __init__(self, key_vec, d, rank, n_layers):
        super().__init__()
        self.key = nn.Parameter(key_vec.detach().clone())
        self.adapters = nn.ModuleList(MEGaAdapter(d, rank) for _ in range(n_layers))

    def apply(self, x, layer_idx):
        return self.adapters[layer_idx](x)


class MEGaBank(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.memories = nn.ModuleList()

    def add_memory(self, key_vec, n_layers):
        if len(self.memories) >= self.cfg.mega_max_adapters:
            return
        self.memories.append(_Memory(key_vec, self.cfg.d_model, self.cfg.mega_rank, n_layers))

    def gate_weights(self, query_vec):
        if len(self.memories) == 0:
            return None, None
        keys = torch.stack([m.key for m in self.memories])
        q = F.normalize(query_vec.flatten(), dim=-1)
        k = F.normalize(keys, dim=-1)
        sim = q @ k.T
        w = F.softmax(sim / self.cfg.mega_gate_temperature, dim=-1)
        return w, sim

    def inject(self, x, layer_idx, query_vec):
        w, sim = self.gate_weights(query_vec)
        if w is None or float(sim.max()) < self.cfg.mega_min_similarity:
            return x
        delta = 0.0
        for i, mem in enumerate(self.memories):
            delta = delta + w[i] * mem.apply(x, layer_idx)
        return x + delta