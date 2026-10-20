"""Padding-free packing + variable-length streaming.

Because the backbone is GDN (linear attention), variable-length examples
do not need a rectangular KV cache. For training we pack multiple
examples into one sequence with a block-diagonal attention mask, so
100% of the batch's tokens are useful. For inference we can stream a
4M-token input chunk by chunk, sharing gist memory across chunks.
"""
from __future__ import annotations

from typing import Optional

import torch


def pack_pairs(
    pairs: list[dict],
    max_len: int = 2048,
    separator_id: int = 32258,
) -> list[dict]:
    """Greedily pack (prompt_ids, target_ids) pairs. Each packed block
    includes a `segments` list so the caller can build a block-diagonal
    mask via make_block_diag_attention_mask(segments)."""
    packed = []
    cur_p: list[int] = []
    cur_t: list[int] = []
    cur_segments: list[int] = []
    cur_len = 0
    for p in pairs:
        pl = len(p["prompt_ids"])
        tl = len(p["target_ids"])
        total = pl + tl + 1
        if cur_len + total > max_len and cur_p:
            packed.append({
                "prompt_ids": cur_p, "target_ids": cur_t,
                "segments": cur_segments, "length": cur_len,
            })
            cur_p, cur_t, cur_segments, cur_len = [], [], [], 0
        cur_p.extend(p["prompt_ids"])
        cur_p.append(separator_id)
        cur_t.extend(p["target_ids"])
        cur_segments.append(pl + tl + 1)
        cur_len += total
    if cur_p:
        packed.append({
            "prompt_ids": cur_p, "target_ids": cur_t,
            "segments": cur_segments, "length": cur_len,
        })
    return packed


def make_block_diag_attention_mask(
    segment_lengths: list[int],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """[T, T] float mask: 0 inside each block, -inf outside."""
    T = sum(segment_lengths)
    mask = torch.full((T, T), float("-inf"), device=device)
    off = 0
    for L in segment_lengths:
        mask[off:off + L, off:off + L] = 0.0
        off += L
    return mask


class VariableLengthStreamer:
    """Process a very long input chunk by chunk without padding."""

    def __init__(self, backbone, chunk_size: int = 2048):
        self.backbone = backbone
        self.chunk_size = chunk_size

    @torch.no_grad()
    def feed(self, ids: torch.Tensor) -> dict:
        squeeze = ids.dim() == 1
        if squeeze:
            ids = ids.unsqueeze(0)
        T = ids.shape[1]
        last = None
        for start in range(0, T, self.chunk_size):
            chunk = ids[:, start:start + self.chunk_size]
            last = self.backbone.stream(chunk, write=True)
        return last if last is not None else {"hidden": None, "n_written": 0}