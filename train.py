"""Training: NEMIX-v2 and a same-size GPT baseline.  All flags read from cfg."""
from __future__ import annotations

import math
import os
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
        from torch.cuda.amp import GradScaler
        return GradScaler(growth_interval=200)
    return None


def _warmup_cosine(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _split_params(model, cfg):
    enc, dec = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "decoder" in n or "mega" in n:
            dec.append(p)
        else:
            enc.append(p)
    return enc, dec


class _GPTBaseline(nn.Module):
    """Decoder-only GPT with a next-token loss, same width as NEMIX."""
    def __init__(self, vocab_size, d_model, n_layers, n_heads):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=4*d_model, batch_first=True, norm_first=True,
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


def _loss_nemix(model, batch, cfg, generator=None):
    p = torch.tensor(batch["prompt_ids"]).unsqueeze(0) if isinstance(batch["prompt_ids"], list) else batch["prompt_ids"]
    t = torch.tensor(batch["target_ids"]).unsqueeze(0) if isinstance(batch["target_ids"], list) else batch["target_ids"]
    return model.loss(p, t, generator=generator)["loss"]


def _loss_gpt(model, ids):
    x, y = ids[:, :-1], ids[:, 1:]
    return F.cross_entropy(
        model(x).reshape(-1, model.head.out_features), y.reshape(-1)
    )


def train_any(model, dataloader, cfg, is_nemix=True, save_path=None):
    device = resolve_device(cfg)
    dtype = resolve_dtype(cfg, device)
    model.to(device)
    if cfg.use_compile and hasattr(torch, "compile") and device == "cuda":
        model = torch.compile(model, mode="max-autotune")

    scaler = _make_scaler(dtype, device)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(dataloader) * 10 // max(1, cfg.grad_accum)
    sched = LambdaLR(opt, lr_lambda=lambda s: _warmup_cosine(s, cfg.warmup_steps, total_steps))

    enc_params, dec_params = _split_params(model, cfg)
    gen = torch.Generator().manual_seed(cfg.seed)
    opt_step = 0
    micro = 0

    for epoch in range(10):
        opt.zero_grad(set_to_none=True)
        for batch in dataloader:
            if is_nemix:
                p = torch.tensor(batch["prompt_ids"], device=device)
                t = torch.tensor(batch["target_ids"], device=device)
                with _autocast_context(dtype, device):
                    loss = model.loss(p, t, generator=gen)["loss"]
            else:
                ids = torch.tensor(batch["input_ids"], device=device)
                with _autocast_context(dtype, device):
                    loss = _loss_gpt(model, ids)
            loss = loss / cfg.grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            micro += 1
            if micro % cfg.grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(enc_params, cfg.grad_clip_encoder)
                torch.nn.utils.clip_grad_norm_(dec_params, cfg.grad_clip_decoder)
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                opt_step += 1
                if opt_step % 50 == 0:
                    print(f"epoch {epoch} step {opt_step} loss {loss.item()*cfg.grad_accum:.4f}")
                if save_path and opt_step % 500 == 0:
                    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, save_path)
    if save_path:
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, save_path)
    return model


def build_gpt_baseline(cfg):
    return _GPTBaseline(cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads)