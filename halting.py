"""Token-level halting with oracle supervision (MoR / LoopFormer / RecurTrace).

The key failure of PonderNet/ACT halting is signal weakness: a scalar
compute penalty is not enough. RecurTrace shows both collapse to a single
loop under matched compute. Fix: oracle supervision. At training time we
literally measure marginal loss reduction from one more loop; a token
should halt when more loops no longer reduce its loss.

Loop memory: a looped block that only passes the latest hidden state
forgets what earlier loops computed. Loop Memory Attention lets each
looped layer attend to its own states along the loop-time axis.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoopMemoryAttention(nn.Module):
    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm = nn.RMSNorm(d)

    def forward(self, x: torch.Tensor, past_hidden: list[torch.Tensor]) -> torch.Tensor:
        if not past_hidden:
            return x
        B, T, d = x.shape
        seq = torch.stack(past_hidden, dim=0)          # [L, B, T, d]
        L = seq.shape[0]
        keys = seq.permute(1, 2, 0, 3).reshape(B * T, L, d)
        q = x.reshape(B * T, 1, d)
        attn_out, _ = self.attn(self.norm(q), keys, keys, need_weights=False)
        return x + attn_out.reshape(B, T, d)


class HaltingPolicy(nn.Module):
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
def compute_oracle_halting_targets(
    state_fn: Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]],
    x0: torch.Tensor,
    target_ids: torch.Tensor,
    pad_mask: torch.Tensor,
    max_loops: int = 4,
    eps: float = 1e-4,
) -> torch.Tensor:
    """state_fn(x, loop_idx) -> (logits, new_x).
    Returns [B, T] int64 halt_at: the first loop at which further loops
    no longer reduce loss. Tokens still improving at max_loops get max_loops."""
    losses = []
    x = x0
    for i in range(max_loops):
        logits, x = state_fn(x, i)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape(target_ids.shape)
        losses.append(ce)
    loss_stack = torch.stack(losses, dim=0)
    reductions = loss_stack[:-1] - loss_stack[1:]
    halt_at = torch.full_like(target_ids, max_loops)
    for i in range(reductions.shape[0] - 1, -1, -1):
        beneficial = reductions[i] > eps
        halt_at = torch.where(beneficial, halt_at, torch.full_like(halt_at, i))
    halt_at = torch.where(pad_mask, torch.zeros_like(halt_at), halt_at)
    return halt_at