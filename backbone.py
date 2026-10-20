"""NEMIX-v3 backbone: Gated DeltaNet + gist memory + sparse episodic store
+ hippocampal exact KV cache + optional hybrid full-attention layers.
"""
from __future__ import annotations

import os
import warnings

import torch
import torch.nn as nn

from .memory import GistMemory
from .episodic import SparseEpisodicStore

try:
    from fla.layers import GatedDeltaNet
except ImportError:
    GatedDeltaNet = None

_ALLOW_FALLBACK = os.environ.get("NEMIX_ALLOW_ATTENTION_FALLBACK", "0") == "1"


def _unpack_mixer_output(out):
    if isinstance(out, tuple):
        return out[0]
    return out


class HippocampusCache:
    """Fixed-size LRU exact KV cache for short-range recall."""

    def __init__(self, size: int = 0):
        self.size = size
        self.k: list[torch.Tensor] = []
        self.v: list[torch.Tensor] = []

    def enabled(self) -> bool:
        return self.size > 0

    def write(self, k: torch.Tensor, v: torch.Tensor) -> None:
        if not self.enabled():
            return
        B, T, d = k.shape
        for b in range(B):
            for t in range(T):
                self.k.append(k[b, t].detach().clone())
                self.v.append(v[b, t].detach().clone())
        while len(self.k) > self.size:
            self.k.pop(0)
            self.v.pop(0)

    def retrieve(self, q: torch.Tensor, topk: int = 4):
        if not self.enabled() or not self.k:
            return None, None
        K = torch.stack(self.k)
        V = torch.stack(self.v)
        scores = torch.nn.functional.normalize(K, dim=-1) @ \
                 torch.nn.functional.normalize(q.flatten(), dim=0)
        idx = torch.topk(scores, min(topk, scores.numel())).indices
        return K[idx], V[idx]


class GDNBlock(nn.Module):
    def __init__(self, cfg, use_full_attention: bool = False):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.is_fallback_attention = False
        self.is_full_attention = use_full_attention
        if use_full_attention:
            self.mixer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model, nhead=cfg.n_heads,
                dim_feedforward=4 * cfg.d_model,
                batch_first=True, norm_first=True,
            )
        elif GatedDeltaNet is not None:
            try:
                self.mixer = GatedDeltaNet(
                    hidden_size=cfg.d_model, num_heads=cfg.n_heads,
                    mode="chunk", expand_v=cfg.expand_v,
                    use_gate=cfg.use_gate, chunk_size=cfg.chunk_size,
                )
            except TypeError:
                self.mixer = GatedDeltaNet(
                    hidden_size=cfg.d_model, num_heads=cfg.n_heads,
                    mode="chunk", expand_v=cfg.expand_v,
                    use_gate=cfg.use_gate,
                )
        else:
            if not _ALLOW_FALLBACK:
                raise ImportError(
                    "flash-linear-attention not installed. Install it or set "
                    "NEMIX_ALLOW_ATTENTION_FALLBACK=1 for deliberate debugging."
                )
            warnings.warn(
                "NEMIX_ALLOW_ATTENTION_FALLBACK=1: quadratic attention. "
                "Not a valid test of NEMIX.",
                RuntimeWarning, stacklevel=2,
            )
            self.mixer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model, nhead=cfg.n_heads,
                dim_feedforward=4 * cfg.d_model,
                batch_first=True, norm_first=True,
            )
            self.is_fallback_attention = True
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False),
            nn.SiLU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False),
        )

    def forward(self, x):
        mixed = _unpack_mixer_output(self.mixer(self.norm1(x)))
        x = x + mixed
        x = x + self.ffn(self.norm2(x))
        return x


class NEMIXBackbone(nn.Module):
    def __init__(self, cfg, unigram_log_probs=None):
        super().__init__()
        for name, sid in [("mask", cfg.mask_id), ("pad", cfg.pad_id),
                          ("bos", cfg.bos_id), ("eos", cfg.eos_id)]:
            assert sid < cfg.vocab_size, (
                f"{name}_id {sid} >= vocab_size {cfg.vocab_size}."
            )
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.emb.weight, std=0.02)

        every = int(getattr(cfg, "hybrid_attention_every", 0))
        layers = []
        for i in range(cfg.n_layers):
            use_full = every > 0 and i > 0 and (i % every == 0)
            layers.append(GDNBlock(cfg, use_full_attention=use_full))
        self.layers = nn.ModuleList(layers)
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
        self.sparse_store = SparseEpisodicStore(
            capacity=getattr(cfg, "sparse_store_capacity", 4096),
            path=getattr(cfg, "sparse_store_path", ""),
            min_novelty=getattr(cfg, "sparse_min_novelty", 0.30),
        )
        self.hippocampus = HippocampusCache(
            size=getattr(cfg, "hippocampus_size", 0)
        )

        if unigram_log_probs is None:
            unigram_log_probs = torch.full(
                (cfg.vocab_size,),
                -torch.log(torch.tensor(float(cfg.vocab_size))).item(),
            )
        self.register_buffer("unigram_log_probs", unigram_log_probs.float())

    def stream(self, ids, write=True):
        squeeze = ids.dim() == 1
        if squeeze:
            ids = ids.unsqueeze(0)
        B, T = ids.shape
        x = self.emb(ids)
        use_ckpt = getattr(self.cfg, "use_gradient_checkpointing", False)
        if use_ckpt and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            for layer in self.layers:
                x = checkpoint(layer, x, use_reentrant=False)
        else:
            for layer in self.layers:
                x = layer(x)
        h = self.norm_f(x)

        if write and self.hippocampus.enabled():
            self.hippocampus.write(h, h)

        n_written = 0
        if write and self.memory is not None:
            step = self.cfg.gist_block_tokens
            for b in range(B):
                for i in range(0, T, step):
                    chunk = ids[b, i:i + step]
                    vec = self.gist_vector(chunk)
                    wrote, novelty = self.memory.write(
                        vec, salient=self.salient_ids(chunk), pos=i,
                        n_tokens=len(chunk),
                    )
                    if wrote:
                        n_written += 1
                        self.sparse_store.maybe_write(
                            token_ids=chunk.tolist(),
                            novelty=novelty,
                            salience=float(-self.unigram_log_probs[chunk].mean().item()),
                            vector=vec,
                        )
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

    def retrieve_sparse(self, query_ids, topk=None):
        if topk is None:
            topk = getattr(self.cfg, "sparse_replay_topk", 3)
        return self.sparse_store.retrieve(self.gist_vector(query_ids), topk=topk)

    def retrieve_exact(self, query_hidden, topk=4):
        return self.hippocampus.retrieve(query_hidden, topk=topk)

    def address_score(self, query_ids) -> float:
        hits = self.retrieve(query_ids, topk=1)
        if not hits:
            return 0.0
        return float(hits[0].get("norm_score", hits[0]["score"]))

    def stream_chunked(self, ids, chunk_size: int = 2048, write: bool = True):
        squeeze = ids.dim() == 1
        if squeeze:
            ids = ids.unsqueeze(0)
        T = ids.shape[1]
        last = None
        for start in range(0, T, chunk_size):
            chunk = ids[:, start:start + chunk_size]
            last = self.stream(chunk, write=write)
        return last if last is not None else {"hidden": None, "n_written": 0}

    def memory_nbytes(self):
        return self.memory.nbytes()

    def mixer_kind(self) -> str:
        if any(getattr(layer, "is_fallback_attention", False) for layer in self.layers):
            return "quadratic_attention_fallback"
        n_full = sum(1 for layer in self.layers if getattr(layer, "is_full_attention", False))
        if n_full > 0:
            return f"hybrid_gdn_{len(self.layers) - n_full}_full_{n_full}"
        return "gated_deltanet"