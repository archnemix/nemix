"""Token-reliability head (PRISM) + leftward relaxation (PURE / T2M).

Standard MDM training does not reliably induce self-correction because
models cannot identify unreliable tokens in a complete input. A small
auxiliary head trained to predict "will this token change on a second
pass?" gives the model a signal for WHERE to remask, not just HOW.

PURE: naive correction fails under contextual lock-in. Unlock a window
of positions and relax leftward, because errors depend on earlier tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TokenReliabilityHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden).squeeze(-1)


def reliability_targets(
    draft_a: torch.Tensor,
    draft_b: torch.Tensor,
    pad_mask: torch.Tensor,
) -> torch.Tensor:
    """1 where the two drafts disagree (token is unstable)."""
    disagree = (draft_a != draft_b).float()
    return disagree * (~pad_mask).float()


def leftward_relaxation_mask(unreliable: torch.Tensor, window: int = 3) -> torch.Tensor:
    """Expand a [B, T] boolean mask leftward by up to `window` positions."""
    if window <= 0:
        return unreliable
    B, T = unreliable.shape
    out = unreliable.clone()
    for shift in range(1, window + 1):
        shifted = torch.zeros_like(unreliable)
        shifted[:, shift:] = unreliable[:, :T - shift]
        out = out | shifted
    return out


def select_remask_positions(
    reliability: torch.Tensor,
    committed: torch.Tensor,
    top_frac: float = 0.25,
    window: int = 3,
) -> torch.Tensor:
    """Pick the least-reliable committed positions and expand leftward."""
    B, T = reliability.shape
    n_eligible = committed.sum(dim=1, keepdim=True).clamp_min(1)
    n_pick = (top_frac * n_eligible).long().clamp_min(1)
    masked_rel = torch.where(committed, reliability, torch.full_like(reliability, -1e9))
    pick = torch.zeros_like(committed)
    for b in range(B):
        k = int(n_pick[b].item())
        if k <= 0:
            continue
        idx = torch.topk(masked_rel[b], k).indices
        pick[b, idx] = True
    return leftward_relaxation_mask(pick, window=window) & committed