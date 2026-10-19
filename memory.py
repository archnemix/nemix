"""Hierarchical gist memory -- the KV-cache replacement.

Why: a KV cache grows linearly with every message, so each new output
waits on ever more compute and RAM until quality degrades under memory
pressure.  Humans do not retain every token -- they keep *gists*.

Design
------
* One gist per block: a salience-weighted vector of the block's content
  plus the handful of *salient* (high-surprisal) token ids -- the rare,
  information-dense bits (needles are exactly the high-surprisal tokens,
  so recall survives compression).
* Novelty gate: a near-duplicate of stored memory is NOT written --
  redundancy never enters capacity.
* Hierarchy: when a level exceeds capacity it consolidates -- groups of
  ``hier_group`` gists merge into one gist at the next level (mean of
  vectors, union of salient ids).  Deep levels summarise older material
  coarsely, like human memory.  Total storage bounded by capacity *
  n_levels: O(1) in context length.
* Retrieval = cosine dot product against the query gist: content-
  addressed recall at every depth.
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
        # levels[0] = recent/fine ... levels[k] = older/coarser
        self.levels: list[list[dict]] = [[]]
        self.writes = 0
        self.skips = 0

    # ------------------------------------------------------------------
    def _iter(self):
        for li, lvl in enumerate(self.levels):
            for gi, g in enumerate(lvl):
                yield li, gi, g

    # ------------------------------------------------------------------
    def write(
        self,
        vec: torch.Tensor,
        salient: Optional[list] = None,
        pos: int = 0,
        n_tokens: int = 0,
    ) -> bool:
        """Novelty-gated write.  Returns False for near-duplicates."""
        vec = vec.detach()
        q = F.normalize(vec.flatten(), dim=0)
        for _, _, g in self._iter():
            if float(torch.dot(q, F.normalize(g["vec"].flatten(), dim=0))) >= self.write_threshold:
                self.skips += 1
                return False
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
        return True

    # ------------------------------------------------------------------
    def _consolidate(self, level: int) -> None:
        """Merge overflow of ``level`` upward (in-place at the top)."""
        src = self.levels[level]
        hg = self.hier_group
        groups = [src[i : i + hg] for i in range(0, len(src), hg)]
        at_top = level + 1 >= self.max_levels

        if not at_top:
            if level + 1 == len(self.levels):
                self.levels.append([])
        else:
            # top level: merge pairwise until within capacity -- storage
            # stays bounded no matter how long the stream is
            while len(groups) > self.capacity:
                groups = [sum(groups[i : i + 2], []) for i in range(0, len(groups), 2)]

        merged = []
        for grp in groups:
            vec = torch.stack([g["vec"] for g in grp]).mean(dim=0)
            sal: list = []
            for g in grp:                      # union, keep order, cap size
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

    # ------------------------------------------------------------------
    def retrieve(self, query: torch.Tensor, topk: int = 5) -> list[dict]:
        """Content-addressed cosine retrieval across all depths."""
        q = F.normalize(query.detach().flatten(), dim=0)
        scored = []
        for li, gi, g in self._iter():
            s = float(torch.dot(q, F.normalize(g["vec"].flatten(), dim=0)))
            scored.append({"score": s, "level": li, "index": gi, "gist": g})
        scored.sort(key=lambda r: -r["score"])
        return scored[:topk]

    def find_salient(self, token_id: int) -> list[dict]:
        """Cued recall: which gists hard-stored this token id?"""
        return [g for _, _, g in self._iter() if token_id in g["salient"]]

    # ------------------------------------------------------------------
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
