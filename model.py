"""NEMIX-v2 full model: GDN backbone + recurrent depth + any-process decoder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import NEMIXBackbone
from .refinement import AnyProcessDecoder
from .megas import MEGaBank


class RecurrentDepth(nn.Module):
    """Weight-tied recurrent core with ACT halting."""

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
        """prompt_ids: [B, T] or [T] -> prompt_hidden: [B, T, D]."""
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        h = self.backbone.stream(prompt_ids, write=False)["hidden"]
        h = self.prelude(h)
        h, ponder = self.rdt(h)
        h = self.coda(h)
        return h, ponder

    def loss(self, prompt_ids, target_ids, generator=None):
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
            target_ids = target_ids.unsqueeze(0)
        prompt_hidden, ponder = self._encode_prompt(prompt_ids)

        B, T_t = target_ids.shape
        if generator is not None:
            r = torch.rand(B, generator=generator) * 0.5 + 0.5
            noise = torch.rand(target_ids.shape, generator=generator)
        else:
            r = torch.rand(B) * 0.5 + 0.5
            noise = torch.rand(target_ids.shape)
        r = r.to(target_ids.device)
        keep = noise.to(target_ids.device) < (1 - r[:, None])
        corrupted = target_ids.clone()
        corrupted[~keep] = self.cfg.mask_id

        logits, ins, dele = self.decoder(prompt_hidden, corrupted)
        nll = F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size),
            target_ids.reshape(-1),
            ignore_index=self.cfg.pad_id,
        )
        # auxiliary heads: predict no insertion, always keep
        ins_target = torch.zeros_like(ins)
        del_target = torch.ones_like(dele)
        aux = 0.01 * F.binary_cross_entropy_with_logits(ins, ins_target)
        aux = aux + 0.01 * F.binary_cross_entropy_with_logits(dele, del_target)
        loss = nll + aux + ponder.to(nll.device)
        acc = (logits.argmax(-1) == target_ids).float().mean()
        return {"loss": loss, "acc": acc.item(), "nll": nll.item()}

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