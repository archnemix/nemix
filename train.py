"""Training: NEMIX-v3 and a same-size GPT baseline.

New:
  * admission-gated training (ContinualMemory.observe per batch, discarded
    examples are dropped from the loss)
  * readout plasticity monitor (adjusts theta_high, partial reset)
  * randomness-consistency routing via prompt_text -> model.loss
  * consolidate_step integration on the gated replay buffer
"""
from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .admission import (
    ContinualMemory,
    AdmissionSignals,
    consolidate_step,
)
from .plasticity import ReadoutMonitor, adjust_theta_high


def resolve_device(cfg):
    if cfg.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.device


def resolve_dtype(cfg, device):
    if cfg.dtype == "fp32":
        return torch.float32
    if cfg.dtype == "fp16":
        return torch.float16
    if cfg.dtype == "bf16":
        return torch.bfloat16
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    return torch.tensor(x, device=device)


def _autocast_context(dtype, device):
    if device == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _make_scaler(dtype, device):
    if device == "cuda" and dtype == torch.float16:
        from torch.amp import GradScaler
        return GradScaler("cuda", growth_interval=200)
    return None


def _warmup_cosine(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class _GPTBaseline(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=4 * d_model, batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        )
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, ids):
        x = self.emb(ids)
        T = ids.shape[1]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=ids.device), diagonal=1)
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        return self.head(self.norm(x))


def _t_max_curriculum(step, cfg):
    if cfg.mask_warmup_steps <= 0:
        return 1.0
    frac = min(1.0, step / cfg.mask_warmup_steps)
    return cfg.mask_warmup_start + (1.0 - cfg.mask_warmup_start) * frac


def _elbo_weight_curriculum(step, cfg):
    warmup = getattr(cfg, "elbo_weight_warmup_steps", 2 * max(1, cfg.mask_warmup_steps))
    start = getattr(cfg, "elbo_min_weight", 1.5)
    final = cfg.elbo_max_weight
    if warmup <= 0:
        return final
    frac = min(1.0, step / warmup)
    return start + (final - start) * frac


def _edit_weight_curriculum(step, cfg):
    w0 = getattr(cfg, "edit_weight_start", 0.0)
    w1 = getattr(cfg, "edit_weight_final", 0.5)
    warmup = getattr(cfg, "edit_weight_warmup_steps", 1000)
    if warmup <= 0:
        return w1
    frac = min(1.0, step / warmup)
    return w0 + (w1 - w0) * frac


def _get_prompt_text(batch, tokenizer):
    if "prompt_text" in batch:
        pt = batch["prompt_text"]
        return pt[0] if isinstance(pt, (list, tuple)) else pt
    if tokenizer is None:
        return None
    try:
        ids = batch["prompt_ids"]
        if ids.dim() > 1:
            ids = ids[0]
        return tokenizer.decode(ids.tolist())
    except Exception:
        return None


def _gist_vectors(model, prompt_ids):
    try:
        with torch.no_grad():
            return [model.backbone.gist_vector(prompt_ids[b]).tolist()
                    for b in range(prompt_ids.shape[0])]
    except Exception:
        return None


@torch.no_grad()
def evaluate_loss(model, val_dataloader, cfg, is_nemix, device, dtype, max_batches=20):
    model.eval()
    total_raw, total_w, n = 0.0, 0.0, 0
    tokenizer = getattr(model, "tokenizer", None)
    for i, batch in enumerate(val_dataloader):
        if i >= max_batches:
            break
        if is_nemix:
            p = _to_device(batch["prompt_ids"], device)
            t = _to_device(batch["target_ids"], device)
            pt = _get_prompt_text(batch, tokenizer)
            with _autocast_context(dtype, device):
                out_raw = model.loss(p, t, t_max=1.0, elbo_max_weight=1.0,
                                     edit_weight=0.0, prompt_text=pt)
                out_w = model.loss(p, t, t_max=1.0,
                                   elbo_max_weight=cfg.elbo_max_weight,
                                   edit_weight=0.0, prompt_text=pt)
            total_raw += float(out_raw["loss"].item())
            total_w += float(out_w["loss"].item())
        else:
            ids = _to_device(batch["input_ids"], device)
            with _autocast_context(dtype, device):
                x, y = ids[:, :-1], ids[:, 1:]
                loss = F.cross_entropy(
                    model(x).reshape(-1, model.head.out_features),
                    y.reshape(-1), ignore_index=cfg.pad_id,
                )
            total_raw += float(loss.item())
            total_w += float(loss.item())
        n += 1
    model.train()
    if n == 0:
        return {"raw": float("nan"), "weighted": float("nan")}
    return {"raw": total_raw / n, "weighted": total_w / n}


def _admission_gate_batch(model, memory, batch, tokenizer, device, utility=0.0):
    if memory is None or not getattr(model, "cfg", None):
        return None
    try:
        prompt_ids = batch["prompt_ids"].to(device)
        vecs = _gist_vectors(model, prompt_ids)
        if vecs is None:
            return None
        keep = []
        for b, v in enumerate(vecs):
            text = None
            if tokenizer is not None:
                try:
                    text = tokenizer.decode(prompt_ids[b].tolist())
                except Exception:
                    text = None
            signals = AdmissionSignals(utility=utility)
            rec = memory.observe(
                str(b),
                payload=prompt_ids[b].detach().cpu().tolist(),
                vector=v, signals=signals, text=text,
            )
            keep.append(rec["decision"] != "discard")
        return torch.tensor(keep, dtype=torch.bool, device=device)
    except Exception:
        return None


def train_any(model, dataloader, cfg, is_nemix=True, save_path=None,
              max_steps=None, val_dataloader=None, eval_every=250,
              tokenizer=None, enable_admission=True,
              enable_plasticity_monitor=True,
              consolidate_every=200):
    device = resolve_device(cfg)
    dtype = resolve_dtype(cfg, device)
    model.to(device)
    model.tokenizer = tokenizer

    if is_nemix and hasattr(model, "backbone"):
        mixer_kind = model.backbone.mixer_kind()
        print(f"NEMIX mixer: {mixer_kind}", flush=True)
        if "fallback" in mixer_kind:
            print("WARNING: not using Gated DeltaNet -- results are not a valid test.",
                  flush=True)

    if cfg.use_compile and hasattr(torch, "compile") and device == "cuda":
        model = torch.compile(model, mode="max-autotune")

    scaler = _make_scaler(dtype, device)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max_steps or (len(dataloader) * 10 // max(1, cfg.grad_accum))
    sched = LambdaLR(opt, lr_lambda=lambda s: _warmup_cosine(s, cfg.warmup_steps, total_steps))

    memory = None
    readout_monitor = None
    if is_nemix and enable_admission:
        memory = ContinualMemory(cfg)
        if hasattr(model, "backbone") and hasattr(model.backbone, "sparse_store"):
            memory.set_sparse_store(model.backbone.sparse_store)
        print(f"Admission enabled (history cap {memory.history.capacity})", flush=True)
    if is_nemix and enable_plasticity_monitor:
        readout_monitor = ReadoutMonitor(
            threshold=getattr(cfg, "readout_rank_threshold", 0.05),
            partial_reset_frac=getattr(cfg, "readout_partial_reset_frac", 0.02),
        )
        print("Plasticity monitor enabled", flush=True)

    # Token usage counter for partial_reset
    vocab_usage = torch.zeros(cfg.vocab_size, device=device)

    step = 0
    micro = 0
    running = 0.0

    for epoch in range(50):
        model.train()
        opt.zero_grad(set_to_none=True)
        for batch in dataloader:
            if step >= (max_steps or 10**9):
                break
            if is_nemix:
                p = _to_device(batch["prompt_ids"], device)
                t = _to_device(batch["target_ids"], device)
                admitted_mask = None
                if memory is not None:
                    admitted_mask = _admission_gate_batch(
                        model, memory, batch, tokenizer, device,
                    )
                    if admitted_mask is not None and admitted_mask.numel() == p.shape[0]:
                        if not admitted_mask.all():
                            if not admitted_mask.any():
                                continue
                            p = p[admitted_mask]
                            t = t[admitted_mask]
                t_max = _t_max_curriculum(step, cfg)
                elbo_w = _elbo_weight_curriculum(step, cfg)
                edit_w = _edit_weight_curriculum(step, cfg)
                prompt_text = _get_prompt_text(batch, tokenizer)
                with _autocast_context(dtype, device):
                    out = model.loss(
                        p, t, t_max=t_max, elbo_max_weight=elbo_w,
                        edit_weight=edit_w, prompt_text=prompt_text,
                    )
                    loss = out["loss"]
                vocab_usage.index_add_(
                    0, t.flatten().clamp_max(cfg.vocab_size - 1),
                    torch.ones_like(t.flatten(), dtype=torch.float),
                )
            else:
                ids = _to_device(batch["input_ids"], device)
                with _autocast_context(dtype, device):
                    x, y = ids[:, :-1], ids[:, 1:]
                    loss = F.cross_entropy(
                        model(x).reshape(-1, model.head.out_features),
                        y.reshape(-1), ignore_index=cfg.pad_id,
                    )

            running += loss.item() / cfg.grad_accum
            loss = loss / cfg.grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            micro += 1
            if micro % cfg.grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1

                if (readout_monitor is not None
                        and step % getattr(cfg, "readout_monitor_every", 500) == 0):
                    try:
                        probe = _to_device(batch["prompt_ids"], device)
                        if probe.shape[0] > 4:
                            probe = probe[:4]
                        result = readout_monitor.measure(model, probe)
                        if result.get("ok"):
                            th = adjust_theta_high(result, cfg)
                            if memory is not None:
                                memory.set_theta_high(th)
                            print(f"  [plasticity] step {step} "
                                  f"erank {result['effective_rank']:.2f} "
                                  f"rayleigh {result['rayleigh']:.4f} "
                                  f"theta_high->{th:.3f}", flush=True)
                            if result.get("below_threshold"):
                                n_reset = readout_monitor.partial_reset(
                                    model, vocab_usage=vocab_usage,
                                )
                                print(f"  [plasticity] partial reset {n_reset} rows",
                                      flush=True)
                    except Exception as e:
                        print(f"  [plasticity] monitor failed: {e}", flush=True)

                if (memory is not None
                        and tokenizer is not None
                        and step % consolidate_every == 0):
                    try:
                        cl = consolidate_step(
                            memory, model, opt, tokenizer,
                            n_replay=8, device=str(device),
                            edit_weight=edit_w,
                        )
                        if cl is not None:
                            print(f"  [admission] consolidate step {step} loss {cl:.4f}",
                                  flush=True)
                    except Exception as e:
                        print(f"  [admission] consolidate failed: {e}", flush=True)

                if step % 50 == 0:
                    if is_nemix:
                        extra = (f" mask {out['mask_frac']:.2f} t_max {t_max:.2f} "
                                 f"elbo_w {elbo_w:.2f} edit_w {edit_w:.2f} "
                                 f"edit_loss {out['edit_loss']:.4f} "
                                 f"rel_loss {out['rel_loss']:.4f} "
                                 f"rand_loss {out['rand_loss']:.4f}")
                    else:
                        extra = ""
                    stats_str = ""
                    if memory is not None:
                        stats = memory.stats()
                        stats_str = (f" adm_e={stats['episodic']} "
                                     f"adm_c={stats['concepts']} "
                                     f"adm_h={stats['history']}")
                    print(f"epoch {epoch} step {step} loss {running:.4f}{extra}{stats_str}",
                          flush=True)

                if val_dataloader is not None and step % eval_every == 0:
                    vl = evaluate_loss(model, val_dataloader, cfg, is_nemix,
                                       device, dtype)
                    if is_nemix:
                        print(f"epoch {epoch} step {step} "
                              f"val_weighted {vl['weighted']:.4f} "
                              f"val_raw_ce {vl['raw']:.4f} "
                              f"gap {vl['weighted'] - running:+.4f}", flush=True)
                    else:
                        print(f"epoch {epoch} step {step} "
                              f"val_loss {vl['raw']:.4f} "
                              f"train_val_gap {vl['raw'] - running:+.4f}", flush=True)
                running = 0.0
                if save_path and step % 500 == 0:
                    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__},
                               save_path)
        if step >= (max_steps or 10**9):
            print(f"reached max_steps={max_steps}", flush=True)
            break

    if save_path:
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, save_path)
    return model


def build_gpt_baseline(cfg):
    return _GPTBaseline(cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads)