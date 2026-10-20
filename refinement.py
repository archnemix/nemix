"""NEMIX-v2 any-process decoder (remask + insert + delete)."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sampler import resolve_sampling_mode
from .creativity import novelty_reweight


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
        # Previously DecoderLayer's dropout defaulted to 0.0 and nothing
        # overrode it, so the one built-in regularizer in the decoder was
        # silently disabled -- nothing was pushing the model toward
        # generalizing rather than memorizing exact training sequences.
        dropout = getattr(cfg, "decoder_dropout", 0.1)
        self.layers = nn.ModuleList(
            DecoderLayer(d, cfg.n_heads, dropout=dropout) for _ in range(cfg.n_dec_layers)
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
               logit_noise=0.0, seed=None, prompt_text=None, sample_mode=None,
               backbone=None, novelty_lam=0.0):
        """prompt_text / sample_mode: previously every commit/revise
        decision in this loop used plain argmax over probs -- Sampler's
        uniform/semantic_uniform modes existed in sampler.py but were
        never called from here, so a prompt asking for "a random number"
        still got the model's ordinary (dataset-frequency-biased) most
        confident token every time; logit_noise only ever added Gaussian
        noise before the same argmax, which is not actually sampling from
        a calibrated distribution. sample_mode, or an auto-detected mode
        from prompt_text via resolve_sampling_mode(), now controls how
        the WINNING token at each committed/revised position is chosen:
        'model' (default) keeps the previous argmax-of-probs behavior
        unchanged for anyone not asking for randomness; 'uniform' picks
        flat over the whole vocab; 'semantic_uniform' picks flat among
        the top-k plausible candidates (stays on-topic, defeats frequency
        bias) -- this is what prompts matching detect_randomness_request
        route to automatically."""
        cfg = self.cfg
        if prompt_hidden.dim() == 2:
            prompt_hidden = prompt_hidden.unsqueeze(0)
        B = prompt_hidden.shape[0]
        dev = prompt_hidden.device
        steps = steps or cfg.refine_steps
        gen = torch.Generator(device="cpu").manual_seed(0 if seed is None else seed)
        mode = resolve_sampling_mode(prompt_text, sample_mode)

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

            # creativity: push logits away from vocabulary entries whose
            # embedding resembles anything already stored in gist memory.
            # novelty_reweight (creativity.py) was fully implemented but
            # had no real call site anywhere in the package -- generation
            # never actually became more novel/creative, regardless of
            # what was stored. Only applied when a backbone (for
            # gist_proj + its GistMemory) is actually passed in and
            # novelty_lam > 0, so default behavior (no backbone passed)
            # is unchanged. Projects the token embedding matrix into
            # gist_dim space via the SAME gist_proj the memory vectors
            # were written with, so the two sides of the comparison are
            # in the same space -- novelty_reweight raises loudly if they
            # aren't, which was the dimensional bug (d_model=256 vs
            # gist_dim=128) in the version nothing ever called.
            if backbone is not None and novelty_lam > 0:
                mem_vecs = backbone.memory.all_vectors()
                if mem_vecs:
                    mem_tensor = torch.stack(mem_vecs).to(dev)
                    emb_in_gist_space = backbone.gist_proj(self.emb.weight)
                    logits = novelty_reweight(logits, emb_in_gist_space,
                                              mem_tensor, lam=novelty_lam)

            probs = F.softmax(logits, dim=-1)
            conf, amax = probs.max(-1)

            # anti-bias sampling: replace the argmax winner at every
            # position with a draw from the requested distribution. conf
            # (used for commit ordering / revise-margin gating) stays the
            # MODEL's own confidence either way -- only WHICH token wins
            # at a position changes. This keeps "commit the positions
            # the model is surest about first" intact while fixing WHAT
            # gets written there.
            if mode == "uniform":
                V = logits.shape[-1]
                # one draw per POSITION, not per vocab entry -- drawing
                # logits.shape ([B,T,V]) worth of randomness here would
                # waste V-1/V of the draws and was simply wrong shape for
                # "pick one random token per position".
                u = torch.rand(logits.shape[:-1], generator=gen).to(dev)
                amax = (u * V).long().clamp_max(V - 1)
                amax = torch.where(amax == cfg.mask_id, (amax + 1) % V, amax)
            elif mode == "semantic_uniform":
                top_k = min(10, logits.shape[-1])
                u = torch.rand(logits.shape, generator=gen).to(dev).clamp_min(1e-6)
                gumbel = -torch.log(-torch.log(u))
                perturbed = logits + 0.1 * gumbel
                top_idx = torch.topk(perturbed, top_k, dim=-1).indices  # [B,T,k]
                pick_pos = torch.randint(0, top_k, logits.shape[:-1],
                                         generator=gen).to(dev)  # [B,T]
                amax = torch.gather(top_idx, -1, pick_pos.unsqueeze(-1)).squeeze(-1)

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

            # insertion: build each row as a python list (length varies
            # per example depending on how many slots that example wants
            # inserted), then pad every row to the batch's new max length
            # with fresh mask_id slots before restacking into a tensor.
            # Previously rows were stacked back individually per-b without
            # reconciling lengths across the batch -- with B>1 and
            # per-example insertion counts differing, that produces
            # ragged rows that cannot actually form a single tensor on
            # the next loop iteration (a silent correctness bug that only
            # surfaces once two rows in a batch insert different counts).
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

            # deletion: a committed position is dropped if the model has
            # low confidence in ANY token there (conf < delete_floor) AND
            # the trained delete_proj head agrees (dele < 0). Previously
            # del_mask was computed and never applied -- deletion was dead
            # code despite allow_delete=True being the default and
            # delete_proj being trained via the aux loss. Applied here,
            # after insertion, against this step's own conf/dele (same
            # pattern insertion already used against this step's ins),
            # so neither edit operates on stale indices from a layout
            # that's since shifted. At least one token per example is
            # always kept so a row never collapses to empty.
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

            # restack: pad every row to the batch max length with fresh
            # mask_id slots so y/committed stay rectangular tensors.
            if row_y_lists is not None:
                max_len = max(len(r) for r in row_y_lists)
                new_y = torch.full((B, max_len), cfg.mask_id, dtype=torch.long, device=dev)
                new_c = torch.zeros((B, max_len), dtype=torch.bool, device=dev)
                for b in range(B):
                    n = len(row_y_lists[b])
                    new_y[b, :n] = torch.stack(row_y_lists[b])
                    new_c[b, :n] = torch.stack(row_c_lists[b])
                y, committed = new_y, new_c

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