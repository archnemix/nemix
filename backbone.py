"""NEMIX-v2 backbone: Gated DeltaNet + gist memory sidecar.

Fixes vs the previous version on disk:
  * GatedDeltaNet.forward returns (hidden, attn_weights, past_kv), not a
    bare tensor.  Unpack defensively so the same GDNBlock also works with
    the nn.TransformerEncoderLayer fallback (which returns a tensor).
  * nn.Embedding's default N(0, 1) init produces logits of std ~sqrt(d)
    and an initial cross-entropy ~200 when the embedding is weight-tied
    to the LM head.  Force std=0.02 so the initial loss is ~log(V).
  * Assert every reserved special token id is < vocab_size at
    construction time.  Without this, a mismatch (e.g. mask_id=32256
    with vocab_size=32256) crashed the CUDA gather inside nn.Embedding
    after 5 minutes of training with a device-side assert, which is
    impossible to debug from the traceback.  This check fails
    immediately with a clear Python message instead.
"""
from __future__ import annotations

import os
import warnings

import torch
import torch.nn as nn

from .memory import GistMemory

try:
    from fla.layers import GatedDeltaNet
except ImportError:
    GatedDeltaNet = None

# If fla is missing, GDNBlock silently falls back to a standard quadratic
# nn.TransformerEncoderLayer -- which defeats the entire point of this
# architecture (no KV-cache growth, O(1) recurrent state per token). A
# model trained on the fallback is NOT testing Gated DeltaNet at all, and
# nothing about its loss curve or output would reveal that. Default
# behavior is now to raise loudly at backbone construction time; set
# NEMIX_ALLOW_ATTENTION_FALLBACK=1 only for deliberate CPU-only debugging
# where the fla Triton kernels genuinely cannot be installed, and even
# then every constructed GDNBlock prints a visible warning so it can
# never be silently mistaken for the real mixer in a training log.
_ALLOW_FALLBACK = os.environ.get("NEMIX_ALLOW_ATTENTION_FALLBACK", "0") == "1"


def _unpack_mixer_output(out):
    """fla's GatedDeltaNet returns (hidden, attn_weights, past_kv) or similar.
    Return only the hidden tensor; pass tensors through unchanged."""
    if isinstance(out, tuple):
        return out[0]
    return out


class GDNBlock(nn.Module):
    """Gated DeltaNet mixer + SwiGLU FFN."""

    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.is_fallback_attention = False
        if GatedDeltaNet is not None:
            try:
                self.mixer = GatedDeltaNet(
                    hidden_size=cfg.d_model,
                    num_heads=cfg.n_heads,
                    mode="chunk",
                    expand_v=cfg.expand_v,
                    use_gate=cfg.use_gate,
                    chunk_size=cfg.chunk_size,
                )
            except TypeError:
                # older fla versions do not accept chunk_size
                self.mixer = GatedDeltaNet(
                    hidden_size=cfg.d_model,
                    num_heads=cfg.n_heads,
                    mode="chunk",
                    expand_v=cfg.expand_v,
                    use_gate=cfg.use_gate,
                )
        else:
            if not _ALLOW_FALLBACK:
                raise ImportError(
                    "flash-linear-attention (fla) is not installed, so "
                    "GatedDeltaNet is unavailable. NEMIX's whole value "
                    "proposition vs a standard transformer is O(1) "
                    "recurrent state instead of a growing KV cache -- "
                    "silently substituting nn.TransformerEncoderLayer "
                    "here would mean every subsequent benchmark against "
                    "a GPT baseline is comparing two attention "
                    "transformers, not testing this architecture. "
                    "Install fla (`pip install flash-linear-attention`) "
                    "or, for deliberate CPU-only debugging only, set "
                    "NEMIX_ALLOW_ATTENTION_FALLBACK=1 to proceed anyway."
                )
            warnings.warn(
                "NEMIX_ALLOW_ATTENTION_FALLBACK=1: GDNBlock is using a "
                "standard quadratic nn.TransformerEncoderLayer, NOT "
                "GatedDeltaNet. KV-cache growth and quadratic compute "
                "are back. Any run using this fallback is not a valid "
                "test of the NEMIX architecture.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.mixer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=4 * cfg.d_model,
                batch_first=True,
                norm_first=True,
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

        # ---- vocab sanity check: catch special-token/vocab collisions --
        for name, sid in [("mask", cfg.mask_id), ("pad", cfg.pad_id),
                          ("bos", cfg.bos_id), ("eos", cfg.eos_id)]:
            assert sid < cfg.vocab_size, (
                f"{name}_id {sid} >= vocab_size {cfg.vocab_size}. "
                f"Special token ids must be reserved slots past the "
                f"tokenizer range, and vocab_size must be > max of them."
            )

        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Weight tying means this matrix is also the LM head.  Default
        # N(0, 1) init gives std ~sqrt(d) logits and an initial loss ~200.
        # std=0.02 puts the initial loss at ~log(vocab_size) ≈ 10.4.
        nn.init.normal_(self.emb.weight, std=0.02)

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
        # cfg.use_gradient_checkpointing was declared in NEMIXConfig and
        # never referenced anywhere -- a dead flag that did nothing
        # regardless of its value. torch.utils.checkpoint trades extra
        # compute (each checkpointed layer's forward runs twice: once
        # during the forward pass, discarded, and once again during
        # backward to recompute activations) for not storing per-layer
        # activations, which is the actual lever for fitting longer
        # sequences / bigger batches in limited GPU memory. Only engages
        # under grad (training); during no_grad inference (generate/
        # refine) it would recompute for nothing, so it's skipped there.
        use_ckpt = getattr(self.cfg, "use_gradient_checkpointing", False)
        if use_ckpt and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            for layer in self.layers:
                x = checkpoint(layer, x, use_reentrant=False)
        else:
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

    def mixer_kind(self) -> str:
        """Report which mixer is actually running, so training logs never
        leave this ambiguous. 'gated_deltanet' is the real architecture;
        'quadratic_attention_fallback' means fla was unavailable and the
        NEMIX_ALLOW_ATTENTION_FALLBACK escape hatch was used -- any run
        reporting this is not a valid test of the architecture."""
        if any(getattr(layer, "is_fallback_attention", False) for layer in self.layers):
            return "quadratic_attention_fallback"
        return "gated_deltanet"