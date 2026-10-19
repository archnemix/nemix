"""NEMIX-v2 any-process decoder (remask + insert + delete)."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.layers = nn.ModuleList(
            DecoderLayer(d, cfg.n_heads) for _ in range(cfg.n_dec_layers)
        )
        self.norm = nn.RMSNorm(d)
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        self.lm_head.weight = emb.weight
        self.insert_proj = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        self.delete_proj = nn.Linear(d, 1)

    def forward(self, prompt_hidden, y):
        if prompt_hidden.dim() == 2:
            prompt_hidden = prompt_hidden.unsqueeze(0)
        x = self.emb(y)
        for layer in self.layers:
            x = layer(x, prompt_hidden)
        h = self.norm(x)
        logits = self.lm_head(h)
        ins = F.softplus(self.insert_proj(h)).squeeze(-1)
        dele = self.delete_proj(h).squeeze(-1)
        return logits, ins, dele

    @torch.no_grad()
    def refine(self, prompt_hidden, length, steps=None, allow_insert=True,
               allow_delete=True, init_ids=None, verifier=None,
               logit_noise=0.0, seed=None):
        cfg = self.cfg
        if prompt_hidden.dim() == 2:
            prompt_hidden = prompt_hidden.unsqueeze(0)
        B = prompt_hidden.shape[0]
        dev = prompt_hidden.device
        steps = steps or cfg.refine_steps
        gen = torch.Generator(device="cpu").manual_seed(0 if seed is None else seed)

        if init_ids is None:
            y = torch.full((B, length), cfg.mask_id, dtype=torch.long, device=dev)
            committed = torch.zeros(B, length, dtype=torch.bool, device=dev)
        else:
            y = init_ids.view(B, -1).clone().to(dev)
            committed = y != cfg.mask_id

        edits = []
        for s in range(steps):
            logits, ins, dele = self.forward(prompt_hidden, y)
            if logit_noise > 0:
                noise = torch.randn(logits.shape, generator=gen).to(dev)
                logits = logits + logit_noise * noise
            # MDLM SUBS: never predict the mask token
            logits = logits.clone()
            logits[..., cfg.mask_id] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            conf, amax = probs.max(-1)

            # deletion (kept off during base training; can be enabled)
            if allow_delete and committed.any():
                del_mask = committed & (conf < cfg.delete_floor) & (dele < 0)
                # deletions handled per-example are rare; left as no-op here
                # to keep the shape bookkeeping simple.

            # revision of committed tokens
            cur_p = probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            better = committed & (conf > cur_p + cfg.revise_margin) & (amax != y)
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
            if allow_insert and s < steps - 1:
                t = (s + 1) / steps
                sched = cfg.insert_alpha * (t ** (cfg.insert_alpha - 1))
                for b in range(B):
                    expected = sched * ins[b]
                    n_ins = min(int(expected.sum().item()), cfg.max_insertions_per_step)
                    if n_ins <= 0:
                        continue
                    top = torch.topk(expected, n_ins).indices.tolist()
                    y_list, c_list = [], []
                    for j in range(y.shape[1]):
                        y_list.append(y[b, j])
                        c_list.append(committed[b, j])
                        if j in top:
                            y_list.append(torch.tensor(cfg.mask_id, device=dev))
                            c_list.append(torch.tensor(False, device=dev))
                    y[b] = torch.stack(y_list)
                    committed[b] = torch.stack(c_list)

        v_rounds = 0
        if verifier is not None:
            for _ in range(3):
                res = verifier(y[0])
                v_rounds += 1
                if res.get("ok"):
                    break
                bad = res.get("bad") or list(range(y.shape[1]))
                for i in bad:
                    if i < y.shape[1]:
                        y[0, i] = cfg.mask_id
                        committed[0, i] = False
                logits, _, _ = self.forward(prompt_hidden, y)
                logits[..., cfg.mask_id] = -float("inf")
                y[0, ~committed[0]] = logits[0, ~committed[0]].argmax(-1)

        return {"tokens": y, "edits": edits, "verify_rounds": v_rounds}