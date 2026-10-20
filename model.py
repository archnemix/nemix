"""NEMIX-v3 full model: GDN backbone + recurrent depth + any-process decoder
+ edit-consistency loss + token-level halting + reliability head + randomness
consistency + Engram hard-bypass.

New in this revision:
  * `ingest_context(ids)` explicitly populates gist memory, sparse store,
    and hippocampal cache. Nothing else writes to memory, so without a
    call to ingest_context the memory system is empty and Engram bypass /
    sparse replay / novelty reweighting all silently no-op. Call it once
    per new conversation / document before generate().
  * Halting in refinement.refine is now gated by cfg.halting_oracle_trained.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import NEMIXBackbone
from .refinement import AnyProcessDecoder
from .megas import MEGaBank
from .halting import LoopMemoryAttention, HaltingPolicy
from .reliability import leftward_relaxation_mask
from .randomness import (
    should_apply_randomness_loss,
    extract_valid_strings,
    strings_to_token_ids,
    randomness_consistency_loss,
)


class RecurrentDepth(nn.Module):
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
        use_loop_mem = getattr(cfg, "halting_use_loop_memory", True)
        self.loop_memory = LoopMemoryAttention(d, n_heads=cfg.n_heads) if use_loop_mem else None
        # NOTE: this halting_head is constructed but only trained once the
        # oracle wiring in model.loss is added. It is intentionally NOT
        # called in forward() below -- the RecurrentDepth-level halting
        # signal comes from self.halt (ACT), not from HaltingPolicy.
        self.halting_head = HaltingPolicy(d) if getattr(cfg, "use_halting", False) else None

    def forward(self, x, return_all_loops: bool = False):
        cfg = self.cfg
        B, L, _ = x.shape
        cumulative = torch.zeros(B, L, device=x.device)
        halted = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        ponder = x.new_zeros(())
        use_ckpt = (getattr(cfg, "use_gradient_checkpointing", False)
                    and self.training and torch.is_grad_enabled())
        if use_ckpt:
            from torch.utils.checkpoint import checkpoint
        past_hidden: list[torch.Tensor] = []
        all_hidden: list[torch.Tensor] = []
        for _ in range(cfg.max_loop_iters):
            for layer in self.blocks:
                x = checkpoint(layer, x, use_reentrant=False) if use_ckpt else layer(x)
            if self.loop_memory is not None and past_hidden:
                x = self.loop_memory(x, past_hidden)
            past_hidden.append(x)
            if return_all_loops:
                all_hidden.append(x)
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
        if return_all_loops:
            return x, ponder, all_hidden
        return x, ponder


class NEMIXModel(nn.Module):
    def __init__(self, cfg, tokenizer=None):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
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

    def _encode_prompt(self, prompt_ids, write_memory: bool = False):
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        h = self.backbone.stream(prompt_ids, write=write_memory)["hidden"]
        h = self.prelude(h)
        h, ponder = self.rdt(h)
        h = self.coda(h)
        return h, ponder

    @torch.no_grad()
    def ingest_context(self, prompt_ids):
        """Populate gist memory / sparse store / hippocampal cache with a
        new context. Call this once per conversation or document before
        generate(). Without this call, Engram bypass, sparse replay, and
        novelty reweighting are all no-ops because the memory is empty."""
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        out = self.backbone.stream(prompt_ids, write=True)
        return {"n_written": out["n_written"],
                "memory_stats": self.backbone.memory.stats(),
                "sparse_entries": len(self.backbone.sparse_store)}

    def _edit_weight(self, step: Optional[int] = None) -> float:
        cfg = self.cfg
        w0 = getattr(cfg, "edit_weight_start", 0.0)
        w1 = getattr(cfg, "edit_weight_final", 0.5)
        warmup = getattr(cfg, "edit_weight_warmup_steps", 1000)
        if warmup <= 0 or step is None:
            return w1
        frac = min(1.0, step / warmup)
        return w0 + (w1 - w0) * frac

    def loss(self, prompt_ids, target_ids, t_max: float = 1.0,
             elbo_max_weight: Optional[float] = None,
             edit_weight: Optional[float] = None,
             step: Optional[int] = None,
             prompt_text: Optional[str] = None,
             return_dict: bool = True):
        if elbo_max_weight is None:
            elbo_max_weight = self.cfg.elbo_max_weight
        if edit_weight is None:
            edit_weight = self._edit_weight(step)
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
            target_ids = target_ids.unsqueeze(0)
        B, T = target_ids.shape
        dev = target_ids.device

        prompt_hidden, ponder = self._encode_prompt(prompt_ids)

        t = torch.rand(B, device=dev) * float(t_max)
        mask_prob = t
        rand = torch.rand((B, T), device=dev)
        masked = rand < mask_prob.unsqueeze(1)
        pad_mask = target_ids == self.cfg.pad_id
        masked = masked & ~pad_mask
        for b in range(B):
            if (not masked[b].any()) and (~pad_mask[b]).any():
                first_valid = (~pad_mask[b]).nonzero(as_tuple=False)[0, 0]
                masked[b, first_valid] = True
        corrupted = target_ids.clone()
        corrupted[masked] = self.cfg.mask_id

        logits, ins, dele, hidden = self.decoder(prompt_hidden, corrupted, return_hidden=True)
        logits = logits.clone()
        logits[..., self.cfg.mask_id] = -float("inf")

        ce = F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape(B, T)

        alpha = (1.0 - t).clamp_min(1.0 / elbo_max_weight)
        weight = (1.0 / alpha).unsqueeze(1)
        denom = masked.float().sum().clamp_min(1.0)
        nll = (ce * masked.float() * weight).sum() / denom
        loss = nll + ponder.to(nll.device)

        edit_loss_val = torch.tensor(0.0, device=dev)
        if edit_weight > 0:
            draft = logits.argmax(-1)
            wrong = (draft != target_ids) & masked & ~pad_mask
            if wrong.any():
                wrong_relaxed = leftward_relaxation_mask(
                    wrong,
                    window=getattr(self.cfg, "edit_leftward_relaxation", 3),
                ) & ~pad_mask
                corrupted2 = draft.clone()
                corrupted2[wrong_relaxed] = self.cfg.mask_id
                logits2, _, _, _ = self.decoder(prompt_hidden, corrupted2, return_hidden=True)
                logits2 = logits2.clone()
                logits2[..., self.cfg.mask_id] = -float("inf")
                ce2 = F.cross_entropy(
                    logits2.reshape(-1, self.cfg.vocab_size),
                    target_ids.reshape(-1),
                    reduction="none",
                ).reshape(B, T)
                edit_loss_val = (
                    ce2 * wrong_relaxed.float()
                ).sum() / wrong_relaxed.float().sum().clamp_min(1.0)
                loss = loss + edit_weight * edit_loss_val

        rel_loss_val = torch.tensor(0.0, device=dev)
        if (self.decoder.reliability_head is not None
                and getattr(self.cfg, "use_reliability_head", True)
                and getattr(self.cfg, "reliability_head_weight", 0.1) > 0):
            with torch.no_grad():
                target_rel = ((logits.argmax(-1) != target_ids) & ~pad_mask).float()
            rel_logit = self.decoder.reliability_head(hidden)
            rel_loss = F.binary_cross_entropy_with_logits(rel_logit, target_rel)
            loss = loss + getattr(self.cfg, "reliability_head_weight", 0.1) * rel_loss
            rel_loss_val = rel_loss.detach()

        rand_loss_val = torch.tensor(0.0, device=dev)
        if (prompt_text is not None
                and getattr(self.cfg, "randomness_consistency_weight", 0.05) > 0
                and self.tokenizer is not None
                and should_apply_randomness_loss(prompt_text)):
            try:
                strings = extract_valid_strings(prompt_text)
                valid_ids = strings_to_token_ids(strings, self.tokenizer)
                if valid_ids:
                    first_pos = (~pad_mask).float().argmax(dim=1)
                    per_ex = []
                    for b in range(B):
                        p = int(first_pos[b].item())
                        per_ex.append(randomness_consistency_loss(logits[b, p], valid_ids, weight=1.0))
                    if per_ex:
                        rand_loss = torch.stack(per_ex).mean()
                        w = getattr(self.cfg, "randomness_consistency_weight", 0.05)
                        loss = loss + w * rand_loss
                        rand_loss_val = rand_loss.detach()
            except Exception:
                pass

        if self.cfg.aux_weight > 0:
            aux = self.cfg.aux_weight * (
                F.binary_cross_entropy_with_logits(ins, torch.zeros_like(ins))
                + F.binary_cross_entropy_with_logits(dele, torch.ones_like(dele))
            )
            loss = loss + aux

        acc = (logits.argmax(-1) == target_ids).float().mean()
        out = {
            "loss": loss,
            "acc": acc.item(),
            "nll": nll.item(),
            "mask_frac": masked.float().mean().item(),
            "edit_loss": float(edit_loss_val.item()),
            "rel_loss": float(rel_loss_val.item()),
            "rand_loss": float(rand_loss_val.item()),
        }
        return out if return_dict else out["loss"]

    def generate(self, prompt_ids, out_len, prompt_text=None,
                novelty_lam=0.0, **kwargs):
        squeeze = prompt_ids.dim() == 1
        prompt_hidden, _ = self._encode_prompt(prompt_ids)
        res = self.decoder.refine(
            prompt_hidden, out_len, prompt_text=prompt_text,
            backbone=self.backbone, novelty_lam=novelty_lam,
            query_ids=prompt_ids,
            **kwargs,
        )
        if squeeze:
            res["tokens"] = res["tokens"][0]
        return res

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def memory_nbytes(self):
        return self.backbone.memory_nbytes()