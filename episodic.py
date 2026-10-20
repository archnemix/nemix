"""Sparse episodic store (NoteBank) + Engram hard-bypass gate (AION).

Gist memory is compressed and cannot recover exact tokens (a variable
defined 2M tokens ago). NoteBank decouples storage from computation: it
archives selected raw text, then replays retrieved text through the
backbone's native forward pass. AION shows a further optimization: when
a query is confidently answerable from memory, skip the transformer
entirely (O(1) lookup). The gate is trained to be confident on factual
queries and unconfident on generative/reasoning queries.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseEpisodicStore:
    def __init__(
        self,
        capacity: int = 4096,
        path: str = "",
        min_novelty: float = 0.30,
    ):
        self.capacity = capacity
        self.path = path
        self.min_novelty = min_novelty
        self.entries: list[dict] = []
        if self.path and os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self.entries = json.load(f)
            except Exception:
                self.entries = []

    def maybe_write(
        self,
        token_ids: list[int],
        novelty: float,
        salience: float = 0.0,
        vector: Optional[torch.Tensor] = None,
    ) -> bool:
        if novelty < self.min_novelty or not token_ids:
            return False
        entry = {
            "ids": list(token_ids),
            "salience": float(salience),
            "novelty": float(novelty),
            "vec": vector.detach().flatten().tolist() if vector is not None else None,
        }
        self.entries.append(entry)
        if len(self.entries) > self.capacity:
            self.entries.sort(key=lambda e: e["salience"] * e["novelty"], reverse=True)
            self.entries = self.entries[: self.capacity]
        self._flush()
        return True

    def _flush(self) -> None:
        if not self.path:
            return
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.entries, f)
            os.replace(tmp, self.path)
        except Exception:
            pass

    @torch.no_grad()
    def retrieve(self, query_vec: torch.Tensor, topk: int = 3) -> list[dict]:
        if not self.entries:
            return []
        q = F.normalize(query_vec.detach().flatten().cpu(), dim=0)
        scored = []
        for e in self.entries:
            if e["vec"] is None:
                continue
            v = F.normalize(torch.tensor(e["vec"]), dim=0)
            s = float(torch.dot(q, v))
            scored.append({"score": s, "entry": e})
        scored.sort(key=lambda r: -r["score"])
        return scored[:topk]

    def __len__(self):
        return len(self.entries)


class EngramGate(nn.Module):
    """Binary classifier: can this query be answered from memory alone?"""

    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden).squeeze(-1)

    @torch.no_grad()
    def should_bypass(self, hidden: torch.Tensor, threshold: float = 0.90) -> torch.Tensor:
        return torch.sigmoid(self.forward(hidden)) >= threshold