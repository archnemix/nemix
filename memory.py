"""Hierarchical gist memory -- the KV-cache replacement.

Same design as before, with two changes:
  * write() now returns (wrote, novelty) so the caller can decide whether
    to archive raw ids in the sparse episodic store (NoteBank).
  * retrieve() normalizes scores by hierarchy level so deep, old gists
    are not systematically outranked by shallow, recent ones (which is
    what linear attention's recency bias would otherwise produce).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class GistMemory:
    def __init__(
        self,
        dim: int,
        capacity: int = 512,
        hier_group: int = 4,
        max_levels: int = 4,
        write_threshold: float = 0.90,
        max_salient: int = 8,
    ):
        self.dim = dim
        self.capacity = capacity
        self.hier_group = hier_group
        self.max_levels = max_levels
        self.write_threshold = write_threshold
        self.max_salient = max_salient
        self.levels: list[list[dict]] = [[]]
        self.writes = 0
        self.skips = 0

    def _iter(self):
        for li, lvl in enumerate(self.levels):
            for gi, g in enumerate(lvl):
                yield li, gi, g

    def write(
        self,
        vec: torch.Tensor,
        salient: Optional[list] = None,
        pos: int = 0,
        n_tokens: int = 0,
    ) -> tuple[bool, float]:
        vec = vec.detach()
        q = F.normalize(vec.flatten(), dim=0)
        max_sim = 0.0
        for _, _, g in self._iter():
            s = float(torch.dot(q, F.normalize(g["vec"].flatten(), dim=0)))
            if s > max_sim:
                max_sim = s
            if s >= self.write_threshold:
                self.skips += 1
                return False, 1.0 - max_sim
        novelty = 1.0 - max_sim
        entry = {
            "vec": vec.detach().flatten().clone(),
            "salient": list(salient or []),
            "pos": pos,
            "n_tokens": n_tokens,
        }
        self.levels[0].append(entry)
        self.writes += 1
        if len(self.levels[0]) > self.capacity:
            self._consolidate(0)
        return True, novelty

    def _consolidate(self, level: int) -> None:
        src = self.levels[level]
        hg = self.hier_group
        groups = [src[i: i + hg] for i in range(0, len(src), hg)]
        at_top = level + 1 >= self.max_levels
        if not at_top:
            if level + 1 == len(self.levels):
                self.levels.append([])
        else:
            while len(groups) > self.capacity:
                groups = [sum(groups[i: i + 2], []) for i in range(0, len(groups), 2)]
        merged = []
        for grp in groups:
            vec = torch.stack([g["vec"] for g in grp]).mean(dim=0)
            sal: list = []
            for g in grp:
                for s in g["salient"]:
                    if s not in sal:
                        sal.append(s)
            merged.append({
                "vec": vec,
                "salient": sal[: self.max_salient],
                "pos": grp[0]["pos"],
                "n_tokens": sum(g["n_tokens"] for g in grp),
            })
        if at_top:
            self.levels[level] = merged
        else:
            self.levels[level] = []
            self.levels[level + 1].extend(merged)
            if len(self.levels[level + 1]) > self.capacity:
                self._consolidate(level + 1)

    def retrieve(self, query: torch.Tensor, topk: int = 5) -> list[dict]:
        q = F.normalize(query.detach().flatten(), dim=0)
        per_level: dict[int, list[float]] = {}
        scored = []
        for li, gi, g in self._iter():
            s = float(torch.dot(q, F.normalize(g["vec"].flatten(), dim=0)))
            per_level.setdefault(li, []).append(s)
            scored.append({"score": s, "level": li, "index": gi, "gist": g})
        for rec in scored:
            li = rec["level"]
            mu = sum(per_level[li]) / max(1, len(per_level[li]))
            rec["norm_score"] = rec["score"] - mu - 0.01 * li
        scored.sort(key=lambda r: -r["norm_score"])
        return scored[:topk]

    def find_salient(self, token_id: int) -> list[dict]:
        return [g for _, _, g in self._iter() if token_id in g["salient"]]

    def stats(self) -> dict:
        counts = [len(lvl) for lvl in self.levels]
        return {
            "levels": len(self.levels),
            "gists_per_level": counts,
            "gists_total": sum(counts),
            "salient_ids": sum(len(g["salient"]) for _, _, g in self._iter()),
            "writes": self.writes,
            "novelty_skips": self.skips,
        }

    def nbytes(self) -> int:
        vecs = sum(g["vec"].numel() for _, _, g in self._iter())
        sals = sum(len(g["salient"]) for _, _, g in self._iter())
        return vecs * 4 + sals * 8

    def all_vectors(self) -> list:
        return [g["vec"] for _, _, g in self._iter()]