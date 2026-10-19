"""NEMIX-v2 full model: GDN backbone + recurrent depth + any-process decoder.

Fixes vs previous version:
  * loss() now implements the MDLM Rao-Blackwellized objective: t ~ U(0, t_max),
    mask probability t, loss ONLY on masked positions, weighted by 1/alpha_t
    clamped to elbo_max_weight.  This is the correct masked-diffusion objective;
    the previous version computed CE on every position and used a fixed
    [0.5, 1.0] mask ratio, which is why the loss bounced instead of decreasing.
  * mask_id logit is set to -inf before CE (MDLM SUBS: zero masking probability).
  * auxiliary insertion/deletion loss is gated by cfg.aux_weight (default 0.0).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import NEMIXBackbone
from .refinement import AnyProcessDecoder
from .megas import MEGaBank


class RecurrentDepth(nn.Module):
    """Weight-tied recurrent core.  With use_act=False (default at 20M) it
    just runs max_loop_iters fixed passes with no halting head and no
    ponder penalty."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=d, nhead=cfg.n_heads,
                dim_feedforward=4 * d, batch_first=True, norm_first=True,
            )
            for _ in range(cfg.n_recurrent)
        )
        self.halt = nn.Linear(d, 1)

    def forward(self, x):
        cfg = self.cfg
        B, L, _ = x.shape
        cumulative = torch.zeros(B, L, device=x.device)
        halted = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        ponder = x.new_zeros(())
        for _ in range(cfg.max_loop_iters):
            for layer in self.blocks:
                x = layer(x)
            if cfg.use_act:
                p = torch.sigmoid(self.halt(x)).squeeze(-1)
                still = ~halted
                p = torch.where(still, p, torch.zeros_like(p))
                cumulative = cumulative + p
                halted = halted | (cumulative >= cfg.act_threshold)
                ponder = ponder + (p * still.float()).sum()
                if halted.all():
                    break
        if cfg.use_act:
            ponder = cfg.act_penalty_weight * ponder / (B * L)
        return x, ponder


class NEMIXModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = NEMIXBackbone(cfg)
        self.prelude = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d_model, batch_first=True, norm_first=True,
        )
        self.rdt = RecurrentDepth(cfg)
        self.coda = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d_model, batch_first=True, norm_first=True,
        )
        self.decoder = AnyProcessDecoder(cfg, self.backbone.emb)
        self.mega = MEGaBank(cfg)

    def _encode_prompt(self, prompt_ids):
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        h = self.backbone.stream(prompt_ids, write=False)["hidden"]
        h = self.prelude(h)
        h, ponder = self.rdt(h)
        h = self.coda(h)
        return h, ponder

    # ------------------------------------------------------------------
    # MDLM Rao-Blackwellized loss
    # ------------------------------------------------------------------
    def loss(self, prompt_ids, target_ids, t_max: float = 1.0):
        """t ~ U(0, t_max); alpha_t = 1 - t; mask probability = t.
        Loss only on masked positions, weighted by 1/alpha_t clamped to
        cfg.elbo_max_weight.  t_max is the curriculum upper bound: start
        at ~0.15 and anneal to 1.0."""
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
            target_ids = target_ids.unsqueeze(0)
        B, T = target_ids.shape
        dev = target_ids.device

        prompt_hidden, ponder = self._encode_prompt(prompt_ids)

        # ---- forward noising process (MDLM) ---------------------------
        t = torch.rand(B, device=dev) * float(t_max)          # [B]
        mask_prob = t                                          # alpha=1-t, mask=1-alpha=t
        rand = torch.rand((B, T), device=dev)
        masked = rand < mask_prob.unsqueeze(1)

        pad_mask = target_ids == self.cfg.pad_id
        masked = masked & ~pad_mask
        # ensure at least one masked token per example (unless all PAD)
        for b in range(B):
            if (not masked[b].any()) and (~pad_mask[b]).any():
                first_valid = (~pad_mask[b]).nonzero(as_tuple=False)[0, 0]
                masked[b, first_valid] = True

        corrupted = target_ids.clone()
        corrupted[masked] = self.cfg.mask_id

        logits, ins, dele = self.decoder(prompt_hidden, corrupted)
        # MDLM SUBS: zero probability on the mask token itself
        logits = logits.clone()
        logits[..., self.cfg.mask_id] = -float("inf")

        ce = F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape(B, T)

        # ---- Rao-Blackwellized weight 1/alpha_t, clamped ---------------
        alpha = (1.0 - t).clamp_min(1.0 / self.cfg.elbo_max_weight)  # [B]
        weight = (1.0 / alpha).unsqueeze(1)                          # [B, 1]

        denom = masked.float().sum().clamp_min(1.0)
        nll = (ce * masked.float() * weight).sum() / denom

        loss = nll + ponder.to(nll.device)

        if self.cfg.aux_weight > 0:
            aux = self.cfg.aux_weight * (
                F.binary_cross_entropy_with_logits(ins, torch.zeros_like(ins))
                + F.binary_cross_entropy_with_logits(dele, torch.ones_like(dele))
            )
            loss = loss + aux

        acc = (logits.argmax(-1) == target_ids).float().mean()
        return {"loss": loss, "acc": acc.item(), "nll": nll.item(),
                "mask_frac": masked.float().mean().item()}

    def generate(self, prompt_ids, out_len, **kwargs):
        squeeze = prompt_ids.dim() == 1
        prompt_hidden, _ = self._encode_prompt(prompt_ids)
        res = self.decoder.refine(prompt_hidden, out_len, **kwargs)
        if squeeze:
            res["tokens"] = res["tokens"][0]
        return res

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def memory_nbytes(self):
        return self.backbone.memory_nbytes()