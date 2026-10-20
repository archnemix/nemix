"""Training: NEMIX-v2 and a same-size GPT baseline."""
from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


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
    """Mask-ratio upper bound: start at mask_warmup_start, linearly reach 1.0."""
    if cfg.mask_warmup_steps <= 0:
        return 1.0
    frac = min(1.0, step / cfg.mask_warmup_steps)
    return cfg.mask_warmup_start + (1.0 - cfg.mask_warmup_start) * frac


def _elbo_weight_curriculum(step, cfg):
    """SUBS loss-weight ceiling (the 1/alpha_t clamp in NEMIXModel.loss).

    This must NOT reach its final value on the same schedule as t_max.
    Once t_max hits 1.0, ~20% of batches draw t close to 1 and receive the
    full weight ceiling on every masked position -- at the exact step the
    model also starts seeing its hardest masking ratios for the first
    time. Under a 1-2 layer decoder, that combination (hardest masking +
    heaviest per-token weight, both arriving simultaneously with no
    transition) is enough to collapse training into predicting the single
    highest-frequency corpus token everywhere -- observed as all-newline /
    all-underscore output and a loss plateau that never recovers even
    after thousands of further steps.

    Fix: ramp the weight ceiling on its own, slower schedule (default 2x
    the mask warmup) so high-mask-ratio batches are only lightly
    up-weighted until the model already has some denoising signal to
    build on."""
    weight_warmup_steps = getattr(cfg, "elbo_weight_warmup_steps",
                                  2 * max(1, cfg.mask_warmup_steps))
    start = getattr(cfg, "elbo_min_weight", 1.5)
    final = cfg.elbo_max_weight
    if weight_warmup_steps <= 0:
        return final
    frac = min(1.0, step / weight_warmup_steps)
    return start + (final - start) * frac


@torch.no_grad()
def evaluate_loss(model, val_dataloader, cfg, is_nemix, device, dtype,
                  max_batches=20):
    """Held-out loss, unweighted and with masking fixed at t_max=1.0 (the
    hardest, most representative setting) so numbers are comparable across
    checkpoints regardless of where the train-time curriculum happens to
    be. This did not exist before -- there was no way to tell memorization
    from generalization during training short of waiting for the run to
    finish and eyeballing generated samples."""
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(val_dataloader):
        if i >= max_batches:
            break
        if is_nemix:
            p = torch.tensor(batch["prompt_ids"], device=device)
            t = torch.tensor(batch["target_ids"], device=device)
            with _autocast_context(dtype, device):
                out = model.loss(p, t, t_max=1.0, elbo_max_weight=cfg.elbo_max_weight)
                loss = out["loss"]
        else:
            ids = torch.tensor(batch["input_ids"], device=device)
            with _autocast_context(dtype, device):
                x, y = ids[:, :-1], ids[:, 1:]
                loss = F.cross_entropy(
                    model(x).reshape(-1, model.head.out_features),
                    y.reshape(-1), ignore_index=cfg.pad_id,
                )
        total += float(loss.item())
        n += 1
    model.train()
    return total / max(n, 1)


def train_any(model, dataloader, cfg, is_nemix=True, save_path=None,
              max_steps=None, val_dataloader=None, eval_every=250):
    """val_dataloader: optional held-out batches (same schema as
    dataloader). When given, validation loss is computed every
    eval_every steps and the train/val gap is printed -- a growing gap
    is the concrete signal that the model is starting to memorize rather
    than generalize, which previously had no instrumentation at all."""
    device = resolve_device(cfg)
    dtype = resolve_dtype(cfg, device)
    model.to(device)

    if is_nemix and hasattr(model, "backbone"):
        mixer_kind = model.backbone.mixer_kind()
        print(f"NEMIX mixer: {mixer_kind}", flush=True)
        if mixer_kind != "gated_deltanet":
            print("WARNING: this run is NOT using Gated DeltaNet -- results "
                  "are not a valid test of the NEMIX architecture.", flush=True)

    if cfg.use_compile and hasattr(torch, "compile") and device == "cuda":
        model = torch.compile(model, mode="max-autotune")

    scaler = _make_scaler(dtype, device)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max_steps or (len(dataloader) * 10 // max(1, cfg.grad_accum))
    sched = LambdaLR(opt, lr_lambda=lambda s: _warmup_cosine(s, cfg.warmup_steps, total_steps))

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
                p = torch.tensor(batch["prompt_ids"], device=device)
                t = torch.tensor(batch["target_ids"], device=device)
                t_max = _t_max_curriculum(step, cfg)
                elbo_w = _elbo_weight_curriculum(step, cfg)
                with _autocast_context(dtype, device):
                    out = model.loss(p, t, t_max=t_max, elbo_max_weight=elbo_w)
                    loss = out["loss"]
            else:
                ids = torch.tensor(batch["input_ids"], device=device)
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
                if step % 50 == 0:
                    extra = f" mask {out['mask_frac']:.2f} t_max {t_max:.2f} elbo_w {elbo_w:.2f}" if is_nemix else ""
                    print(f"epoch {epoch} step {step} loss {running:.4f}{extra}", flush=True)
                if val_dataloader is not None and step % eval_every == 0:
                    val_loss = evaluate_loss(model, val_dataloader, cfg, is_nemix, device, dtype)
                    gap = val_loss - running
                    print(f"epoch {epoch} step {step} val_loss {val_loss:.4f} "
                         f"train_val_gap {gap:+.4f}", flush=True)
                running = 0.0
                if save_path and step % 500 == 0:
                    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, save_path)
        if step >= (max_steps or 10**9):
            print(f"reached max_steps={max_steps}", flush=True)
            break

    if save_path:
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, save_path)
    return model


def build_gpt_baseline(cfg):
    return _GPTBaseline(cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads)