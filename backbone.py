"""NEMIX-v2 backbone: Gated DeltaNet + gist memory sidecar."""
from __future__ import annotations

import torch
import torch.nn as nn

from .memory import GistMemory

try:
    from fla.layers import GatedDeltaNet
except ImportError:
    GatedDeltaNet = None


class GDNBlock(nn.Module):
    """Gated DeltaNet mixer + SwiGLU FFN.  Falls back to attention if fla missing."""

    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        if GatedDeltaNet is not None:
            kwargs = dict(
                hidden_size=cfg.d_model,
                num_heads=cfg.n_heads,
                mode="chunk",
                expand_v=cfg.expand_v,
                use_gate=cfg.use_gate,
            )
            try:
                self.mixer = GatedDeltaNet(**kwargs, chunk_size=cfg.chunk_size)
            except TypeError:
                # older fla versions do not accept chunk_size
                self.mixer = GatedDeltaNet(**kwargs)
        else:
            # graceful fallback so imports never fail on machines without fla
            self.mixer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model, nhead=cfg.n_heads,
                dim_feedforward=4 * cfg.d_model, batch_first=True, norm_first=True,
            )
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False),
            nn.SiLU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False),
        )

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class NEMIXBackbone(nn.Module):
    def __init__(self, cfg, unigram_log_probs=None):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(GDNBlock(cfg) for _ in range(cfg.n_layers))
        self.norm_f = nn.RMSNorm(cfg.d_model)

        self.gist_proj = nn.Linear(cfg.d_model, cfg.gist_dim, bias=False)
        self.memory = GistMemory(
            dim=cfg.gist_dim,
            capacity=cfg.gist_capacity,
            hier_group=cfg.hier_group,
            max_levels=cfg.hier_max_levels,
            write_threshold=cfg.gist_write_threshold,
            max_salient=cfg.max_salient_tokens,
        )
        if unigram_log_probs is None:
            # uniform log-probability (not -9 constant, which would make every token equally salient)
            unigram_log_probs = torch.full(
                (cfg.vocab_size,), -torch.log(torch.tensor(float(cfg.vocab_size))).item()
            )
        self.register_buffer("unigram_log_probs", unigram_log_probs.float())

    def stream(self, ids, write=True):
        """ids: [B, T] or [T].  Returns dict with 'hidden' [B, T, D]."""
        squeeze = ids.dim() == 1
        if squeeze:
            ids = ids.unsqueeze(0)
        B, T = ids.shape
        x = self.emb(ids)
        for layer in self.layers:
            x = layer(x)
        h = self.norm_f(x)

        n_written = 0
        if write and self.memory is not None:
            step = self.cfg.gist_block_tokens
            for b in range(B):
                for i in range(0, T, step):
                    chunk = ids[b, i:i + step]
                    vec = self.gist_vector(chunk)
                    if self.memory.write(vec, salient=self.salient_ids(chunk), pos=i):
                        n_written += 1
        return {"hidden": h.squeeze(0) if squeeze else h, "n_written": n_written}

    def gist_vector(self, ids):
        sal = (-self.unigram_log_probs[ids]).clamp(0.0, 20.0)
        w = torch.softmax(sal, dim=0)
        vec = (w.unsqueeze(-1) * self.emb(ids)).sum(0)
        return self.gist_proj(vec)

    def salient_ids(self, ids):
        sal = -self.unigram_log_probs[ids]
        th = self.cfg.salience_threshold
        keep = [(float(s), int(t)) for s, t in zip(sal, ids) if float(s) >= th]
        keep.sort(key=lambda p: -p[0])
        return [t for _, t in keep[: self.cfg.max_salient_tokens]]

    def retrieve(self, query_ids, topk=5):
        return self.memory.retrieve(self.gist_vector(query_ids), topk=topk)

    def memory_nbytes(self):
        return self.memory.nbytes()