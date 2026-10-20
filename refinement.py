"""NEMIX-v3 any-process decoder: remask + insert + delete, plus
reliability-guided remasking (PRISM), leftward relaxation (PURE/T2M),
token-level halting (MoR/LoopFormer/RecurTrace), and Engram hard-bypass (AION).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sampler import resolve_sampling_mode
from .creativity import novelty_reweight
from .reliability import (
    TokenReliabilityHead,
    leftward_relaxation_mask,
    select_remask_positions,
)
from .halting import HaltingPolicy
from .episodic import EngramGate


class DecoderLayer(nn.Module):
    def __init__(self, d, nhead, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, y, prompt):
        y_n = self.norm1(y)
        y = y + self.drop(self.self_attn(y_n, y_n, y_n, need_weights=False)[0])
        y_n = self.norm2(y)
        y = y + self.drop(self.cross_attn(y_n, prompt, prompt, need_weights=False)[0])
        y = y + self.drop(self.ffn(self.norm3(y)))
        return y


class AnyProcessDecoder(nn.Module):
    def __init__(self, cfg, emb: nn.Embedding):
        super().__init__()
        self.cfg = cfg
        self.emb = emb
        d = cfg.d_model
        dropout = getattr(cfg, "decoder_dropout", 0.1)
        self.layers = nn.ModuleList(
            DecoderLayer(d, cfg.n_heads, dropout=dropout) for _ in range(cfg.n_dec_layers)
        )
        self.norm = nn.RMSNorm(d)
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        self.lm_head.weight = emb.weight
        self.insert_proj = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        self.delete_proj = nn.Linear(d, 1)
        self.reliability_head = TokenReliabilityHead(d) if getattr(cfg, "use_reliability_head", True) else None
        self.halting_head = HaltingPolicy(d) if getattr(cfg, "use_halting", False) else None
        self.engram_gate = EngramGate(d) if getattr(cfg, "use_engram_bypass", True) else None

    def forward(self, prompt_hidden, y, return_hidden: bool = False):
        if prompt_hidden.dim() == 2:
            prompt_hidden = prompt_hidden.unsqueeze(0)
        x = self.emb(y)
        for layer in self.layers:
            x = layer(x, prompt_hidden)
        h = self.norm(x)
        logits = self.lm_head(h)
        ins = F.softplus(self.insert_proj(h)).squeeze(-1)
        dele = self.delete_proj(h).squeeze(-1)
        if return_hidden:
            return logits, ins, dele, h
        return logits, ins, dele

    @torch.no_grad()
    def refine(self, prompt_hidden, length, steps=None, allow_insert=True,
               allow_delete=True, init_ids=None, verifier=None,
               logit_noise=0.0, seed=None, prompt_text=None, sample_mode=None,
               backbone=None, novelty_lam=0.0, query_ids=None):
        cfg = self.cfg
        if prompt_hidden.dim() == 2:
            prompt_hidden = prompt_hidden.unsqueeze(0)
        B = prompt_hidden.shape[0]
        dev = prompt_hidden.device
        steps = steps or cfg.refine_steps
        gen = torch.Generator(device="cpu").manual_seed(0 if seed is None else seed)
        mode = resolve_sampling_mode(prompt_text, sample_mode)

        # ---- Engram hard-bypass (AION): B==1 only (per-example decision) ----
        if (B == 1 and self.engram_gate is not None and backbone is not None
                and getattr(cfg, "use_engram_bypass", True) and query_ids is not None):
            try:
                q = query_ids[0] if query_ids.dim() > 1 else query_ids
                address = backbone.address_score(q)
                if address >= getattr(cfg, "engram_confidence_threshold", 0.90):
                    hits = backbone.retrieve(q, topk=1)
                    if hits:
                        salient = hits[0]["gist"].get("salient") or []
                        if salient:
                            tokens = torch.tensor(
                                [salient[i % len(salient)] for i in range(length)],
                                device=dev, dtype=torch.long,
                            ).unsqueeze(0)
                            return {"tokens": tokens, "edits": [],
                                    "verify_rounds": 0, "bypassed": True,
                                    "address_score": address}
            except Exception:
                pass

        if init_ids is None:
            y = torch.full((B, length), cfg.mask_id, dtype=torch.long, device=dev)
            committed = torch.zeros(B, length, dtype=torch.bool, device=dev)
        else:
            y = init_ids.view(B, -1).clone().to(dev)
            committed = y != cfg.mask_id

        edits = []
        for s in range(steps):
            logits, ins, dele, hidden = self.forward(prompt_hidden, y, return_hidden=True)
            if logit_noise > 0:
                noise = torch.randn(logits.shape, generator=gen).to(dev)
                logits = logits + logit_noise * noise
            logits = logits.clone()
            logits[..., cfg.mask_id] = -float("inf")

            if backbone is not None and novelty_lam > 0:
                mem_vecs = backbone.memory.all_vectors()
                if mem_vecs:
                    mem_tensor = torch.stack(mem_vecs).to(dev)
                    emb_in_gist_space = backbone.gist_proj(self.emb.weight)
                    logits = novelty_reweight(logits, emb_in_gist_space, mem_tensor, lam=novelty_lam)

            probs = F.softmax(logits, dim=-1)
            conf, amax = probs.max(-1)

            if mode == "uniform":
                V = logits.shape[-1]
                u = torch.rand(logits.shape[:-1], generator=gen).to(dev)
                amax = (u * V).long().clamp_max(V - 1)
                amax = torch.where(amax == cfg.mask_id, (amax + 1) % V, amax)
            elif mode == "semantic_uniform":
                top_k = min(10, logits.shape[-1])
                u = torch.rand(logits.shape, generator=gen).to(dev).clamp_min(1e-6)
                gumbel = -torch.log(-torch.log(u))
                perturbed = logits + 0.1 * gumbel
                top_idx = torch.topk(perturbed, top_k, dim=-1).indices
                pick_pos = torch.randint(0, top_k, logits.shape[:-1], generator=gen).to(dev)
                amax = torch.gather(top_idx, -1, pick_pos.unsqueeze(-1)).squeeze(-1)

            # token-level halting (soft freeze of confident committed positions)
            if self.halting_head is not None and getattr(cfg, "use_halting", False):
                try:
                    halt_prob = torch.sigmoid(self.halting_head(hidden))
                    newly_halted = committed & (halt_prob >= 0.5)
                except Exception:
                    newly_halted = torch.zeros_like(committed)
            else:
                newly_halted = torch.zeros_like(committed)

            # PRISM reliability-guided remask of committed tokens
            if self.reliability_head is not None and getattr(cfg, "use_reliability_head", True):
                try:
                    rel_prob = torch.sigmoid(self.reliability_head(hidden))
                    remask = select_remask_positions(
                        reliability=rel_prob,
                        committed=committed & ~newly_halted,
                        top_frac=getattr(cfg, "reliability_remask_top_frac", 0.25),
                        window=getattr(cfg, "edit_leftward_relaxation", 3),
                    )
                    if remask.any():
                        for b in range(B):
                            for i in remask[b].nonzero().flatten().tolist():
                                edits.append(("remask_rel", b, i, int(y[b, i]), None))
                                y[b, i] = cfg.mask_id
                                committed[b, i] = False
                except Exception:
                    pass

            # revise committed tokens
            cur_p = probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            better = (committed & ~newly_halted
                      & (conf > cur_p + cfg.revise_margin) & (amax != y))
            for b in range(B):
                for i in better[b].nonzero().flatten().tolist():
                    edits.append(("edit", b, i, int(y[b, i]), int(amax[b, i])))
                    y[b, i] = amax[b, i]

            # commit most confident masked positions
            remaining = ~committed
            for b in range(B):
                if not remaining[b].any():
                    continue
                n_rem = int(remaining[b].sum())
                m = min(n_rem, max(1, math.ceil(cfg.commit_per_step * n_rem)))
                if s == steps - 1:
                    pick = remaining[b]
                else:
                    masked_conf = torch.where(
                        remaining[b], conf[b], torch.full_like(conf[b], -1e9)
                    )
                    pick = torch.zeros_like(remaining[b])
                    pick[torch.topk(masked_conf, m).indices] = True
                y[b][pick] = amax[b][pick]
                committed[b] |= pick

            # insertion
            row_y_lists, row_c_lists = None, None
            if allow_insert and s < steps - 1:
                t = (s + 1) / steps
                sched = cfg.insert_alpha * (t ** (cfg.insert_alpha - 1))
                row_y_lists, row_c_lists = [], []
                for b in range(B):
                    expected = sched * ins[b]
                    n_ins = min(int(expected.sum().item()), cfg.max_insertions_per_step)
                    top = set(torch.topk(expected, n_ins).indices.tolist()) if n_ins > 0 else set()
                    y_list, c_list = [], []
                    for j in range(y.shape[1]):
                        y_list.append(y[b, j])
                        c_list.append(committed[b, j])
                        if j in top:
                            y_list.append(torch.tensor(cfg.mask_id, device=dev))
                            c_list.append(torch.tensor(False, device=dev))
                    row_y_lists.append(y_list)
                    row_c_lists.append(c_list)

            # deletion
            if allow_delete and committed.any():
                del_mask = committed & (conf < cfg.delete_floor) & (dele < 0)
                if row_y_lists is None:
                    row_y_lists = [[y[b, j] for j in range(y.shape[1])] for b in range(B)]
                    row_c_lists = [[committed[b, j] for j in range(y.shape[1])] for b in range(B)]
                for b in range(B):
                    drop = set(del_mask[b].nonzero().flatten().tolist())
                    if not drop:
                        continue
                    L = y.shape[1]
                    if len(drop) >= L:
                        drop = set(list(drop)[: L - 1])
                    for i in sorted(drop):
                        edits.append(("delete", b, i, int(y[b, i]), None))
                    keep_idx = [j for j in range(L) if j not in drop]
                    if len(keep_idx) == 0:
                        continue
                    row_y_lists[b] = [row_y_lists[b][j] for j in keep_idx]
                    row_c_lists[b] = [row_c_lists[b][j] for j in keep_idx]

            if row_y_lists is not None:
                max_len = max(len(r) for r in row_y_lists)
                new_y = torch.full((B, max_len), cfg.mask_id, dtype=torch.long, device=dev)
                new_c = torch.zeros((B, max_len), dtype=torch.bool, device=dev)
                for b in range(B):
                    n = len(row_y_lists[b])
                    new_y[b, :n] = torch.stack(row_y_lists[b])
                    new_c[b, :n] = torch.stack(row_c_lists[b])
                y, committed = new_y, new_c

        # verification loop with leftward relaxation
        v_rounds = 0
        if verifier is not None:
            for _ in range(3):
                res = verifier(y[0])
                v_rounds += 1
                if res.get("ok"):
                    break
                bad = res.get("bad") or list(range(y.shape[1]))
                bad_mask = torch.zeros(y.shape[1], dtype=torch.bool, device=dev)
                for i in bad:
                    if i < y.shape[1]:
                        bad_mask[i] = True
                bad_mask = leftward_relaxation_mask(
                    bad_mask.unsqueeze(0),
                    window=getattr(cfg, "edit_leftward_relaxation", 3),
                )[0]
                for i in bad_mask.nonzero().flatten().tolist():
                    y[0, i] = cfg.mask_id
                    committed[0, i] = False
                logits, _, _, _ = self.forward(prompt_hidden, y, return_hidden=True)
                logits[..., cfg.mask_id] = -float("inf")
                y[0, ~committed[0]] = logits[0, ~committed[0]].argmax(-1)

        return {"tokens": y, "edits": edits, "verify_rounds": v_rounds, "bypassed": False}