"""Continual learning: admission formula + OASIS history sketch + readout
monitor hookup.

Fixes applied in this revision:
  * consolidate_step() accepts payloads stored as list[int] (token ids),
    which is what train._admission_gate_batch actually stores. Previously it
    always did str(payload) then tokenizer.encode(...), which tokenized the
    literal characters '[', '1', '5', ',' ... and trained on garbage.
  * ContinualMemory.observe() no longer tries to write to SparseEpisodicStore.
    That branch was writing utf-8 BYTES as if they were token ids, and passed
    vector=None so entries were permanently unretrievable. The sparse store is
    now populated only by backbone.stream(write=True), which uses real chunk
    tokens and a real gist vector.
  * _cos() uses .item() instead of the fragile `torch.linalg.norm(a) or 1.0`
    pattern.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Optional

import torch

from .plasticity import HistorySketch


FRUSTRATION_MARKERS = (
    "stupid", "idiot", "wtf", "ugh", "damn", "hate you", "useless",
    "trash", "garbage", "whatever", "idc", "fuck", "shit", "dumb",
    "pointless", "waste of time",
)
THANKS_ONLY = {
    "thanks", "thank you", "thx", "ok", "okay", "cool", "nice", "great",
    "good", "yes", "no", "yep", "nope", "lol", "k", "kk", "wow",
    "nice!", "ok.", "thanks.", "ty", "appreciate it",
}
DEFLECTION_MARKERS = ("as an ai", "i cannot assist", "as a language model")
_REPEAT_RE = re.compile(r"(.)\1{4,}")
_LAUGHTER_RE = re.compile(r"(ha){3,}|(lol)+|lmao|hehehe")


def text_bloat(text: str) -> float:
    t = (text or "").strip().lower()
    if not t:
        return 1.0
    score = 0.0
    core = t.strip(" .!?,;:")
    if core in THANKS_ONLY:
        return 1.0
    words = core.split()
    if words and all(w.strip(" .!?") in THANKS_ONLY for w in words):
        return 1.0
    if any(m in t for m in FRUSTRATION_MARKERS):
        score += 0.50
    if any(m in t for m in DEFLECTION_MARKERS):
        score += 0.50
    if _REPEAT_RE.search(t) or _LAUGHTER_RE.search(t):
        score += 0.25
    if len(words) <= 2 and not any(ch.isdigit() for ch in t):
        score += 0.35
    if len(words) >= 3:
        if len(set(words)) / len(words) < 0.4:
            score += 0.20
    letters = [c for c in t if c.isalpha()]
    if len(letters) >= 6 and sum(1 for c in letters if c.isupper()) / len(letters) > 0.7:
        score += 0.15
    return min(score, 1.0)


@dataclass
class AdmissionSignals:
    novelty: float = 0.0
    learnability: float = 0.0
    utility: float = 0.0
    redundancy: float = 0.0
    bloat: float = 0.0
    volatility: float = 0.0


def admission_score(s: AdmissionSignals, cfg=None) -> float:
    wn = getattr(cfg, "w_novelty", 0.25)
    wl = getattr(cfg, "w_learnability", 0.25)
    wu = getattr(cfg, "w_utility", 0.20)
    wr = getattr(cfg, "w_redundancy", 0.15)
    wb = getattr(cfg, "w_bloat", 0.30)
    wv = getattr(cfg, "w_volatility", 0.15)
    return (
        wn * s.novelty + wl * s.learnability + wu * s.utility
        - wr * s.redundancy - wb * s.bloat - wv * s.volatility
    )


def admit(s: AdmissionSignals, cfg=None, theta_high: Optional[float] = None) -> str:
    if s.bloat >= 0.5 or s.volatility >= 0.5:
        return "discard"
    th = theta_high if theta_high is not None else getattr(cfg, "theta_high", 0.25)
    tl = getattr(cfg, "theta_low", -0.05)
    a = admission_score(s, cfg)
    if a >= th:
        return "consolidate"
    if a >= tl:
        return "episodic"
    return "discard"


def _cos(a, b) -> float:
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    na = float(torch.linalg.norm(a).item()) or 1.0
    nb = float(torch.linalg.norm(b).item()) or 1.0
    return float(torch.dot(a, b).item() / (na * nb))


@dataclass
class MemoryEntry:
    key: str
    payload: Any
    vector: tuple
    score: float
    hits: int = 1


class ContinualMemory:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.cap_e = getattr(cfg, "capacity_episodic", 64)
        self.cap_c = getattr(cfg, "capacity_concepts", 128)
        self.promote_hits = getattr(cfg, "promote_hits", 2)
        self.episodic: list[MemoryEntry] = []
        self.concepts: list[MemoryEntry] = []
        self.log: list[dict] = []
        dim = getattr(cfg, "gist_dim", 128) if cfg is not None else 128
        self.history = HistorySketch(
            dim=dim,
            capacity=getattr(cfg, "admission_history_size", 512) if cfg else 512,
        )
        self._current_theta_high: Optional[float] = None
        self.archive_hook = None

    def set_archive_hook(self, hook) -> None:
        self.archive_hook = hook

    def set_theta_high(self, th: Optional[float]) -> None:
        self._current_theta_high = th

    def _effective_theta_high(self) -> float:
        if self._current_theta_high is not None:
            return self._current_theta_high
        return getattr(self.cfg, "theta_high", 0.25)

    def _find(self, key: str) -> Optional[MemoryEntry]:
        for e in self.concepts:
            if e.key == key:
                return e
        for e in self.episodic:
            if e.key == key:
                return e
        return None

    def max_similarity(self, vector) -> float:
        best = 0.0
        for e in self.concepts + self.episodic:
            best = max(best, _cos(vector, e.vector))
        return best

    def history_similarity(self, vector) -> float:
        return self.history.max_similarity(torch.as_tensor(vector, dtype=torch.float32))

    def observe(
        self,
        key: str,
        payload: Any,
        vector,
        signals: Optional[AdmissionSignals] = None,
        text: Optional[str] = None,
    ) -> dict:
        if signals is None:
            signals = AdmissionSignals()
        if text is not None:
            signals.bloat = max(signals.bloat, text_bloat(text))

        sim_concept = max((_cos(vector, e.vector) for e in self.concepts), default=0.0)
        sim_history = self.history_similarity(vector)
        effective_redundancy = max(sim_concept, sim_history)
        if signals.redundancy == 0.0:
            signals.redundancy = effective_redundancy
        if signals.novelty == 0.0:
            signals.novelty = 1.0 - max(
                sim_concept, self.max_similarity(vector), sim_history,
            )

        a = admission_score(signals, self.cfg)
        th = self._effective_theta_high()
        decision = admit(signals, self.cfg, theta_high=th)
        rec = {
            "key": key, "score": round(a, 4), "decision": decision,
            "theta_high_used": round(th, 4),
        }
        self.log.append(rec)

        if decision == "discard":
            return rec

        existing = self._find(key)
        if existing is not None:
            existing.hits += 1
            existing.score = max(existing.score, a)
            existing.payload = payload
            rec["action"] = "refresh"
        else:
            entry = MemoryEntry(key, payload, tuple(vector), a, hits=1)
            self.episodic.append(entry)
            rec["action"] = "episodic"

        try:
            self.history.add(torch.as_tensor(vector, dtype=torch.float32).flatten())
        except Exception:
            pass
        if self.archive_hook is not None:
            try:
                self.archive_hook(key, payload, vector, signals.novelty)
            except Exception:
                pass

        self._enforce_capacity()
        return rec

    def _promote(self, entry: MemoryEntry) -> None:
        if entry in self.episodic:
            self.episodic.remove(entry)
        if entry not in self.concepts:
            self.concepts.append(entry)

    def _enforce_capacity(self) -> None:
        th = self._effective_theta_high()
        for e in list(self.episodic):
            if e.hits >= self.promote_hits and e.score >= th:
                self._promote(e)
        if len(self.episodic) > self.cap_e:
            self.episodic.sort(key=lambda e: e.score, reverse=True)
            self.episodic = self.episodic[: self.cap_e]
        if len(self.concepts) > self.cap_c:
            self.concepts.sort(key=lambda e: e.score, reverse=True)
            self.concepts = self.concepts[: self.cap_c]

    def replay(self, n: int, seed: int = 0) -> list:
        pool = self.concepts + self.episodic
        if not pool:
            return []
        rng = random.Random(seed)
        weights = [max(e.score, 0.01) + 0.05 for e in pool]
        return [rng.choices(pool, weights=weights, k=1)[0] for _ in range(n)]

    def stats(self) -> dict:
        return {
            "episodic": len(self.episodic),
            "concepts": len(self.concepts),
            "history": len(self.history),
            "discarded": sum(1 for r in self.log if r["decision"] == "discard"),
            "accepted": sum(1 for r in self.log if r["decision"] != "discard"),
        }


def observe_turn(
    memory: "ContinualMemory",
    model,
    text: str,
    key: Optional[str] = None,
    tokenizer=None,
    utility: float = 0.0,
) -> dict:
    if key is None:
        key = (text or "").strip().lower()[:80]
    vector = [0.0] * max(getattr(model, "cfg", None) and model.cfg.gist_dim or 8, 1)
    if tokenizer is not None and model is not None and hasattr(model, "backbone"):
        try:
            ids = tokenizer.encode(text)
            ids = torch.as_tensor(ids, dtype=torch.long)
            with torch.no_grad():
                vector = model.backbone.gist_vector(ids).tolist()
        except Exception:
            pass
    signals = AdmissionSignals(utility=utility)
    return memory.observe(key, payload=text, vector=vector, signals=signals, text=text)


def _payload_to_ids(payload, tokenizer, device) -> Optional[torch.Tensor]:
    """Normalize a MemoryEntry.payload into a long tensor of token ids.

    Accepts:
      * list[int] / tuple[int]: treated as token ids directly.
      * torch.Tensor 1-D of ints: treated as token ids directly.
      * str: encoded via the tokenizer.
    Returns None if the payload cannot be turned into >= 2 token ids.
    """
    if isinstance(payload, torch.Tensor):
        t = payload.detach().flatten().to(dtype=torch.long)
    elif isinstance(payload, (list, tuple)) and payload and isinstance(payload[0], int):
        t = torch.as_tensor(list(payload), dtype=torch.long)
    elif isinstance(payload, str):
        if tokenizer is None:
            return None
        ids = tokenizer.encode(payload)
        t = torch.as_tensor(ids, dtype=torch.long)
    else:
        return None
    if t.numel() < 2:
        return None
    return t.to(device)


def consolidate_step(
    memory: "ContinualMemory",
    model,
    opt,
    tokenizer,
    n_replay: int = 8,
    device: str = "cpu",
    edit_weight: float = 0.0,
):
    """One small gradient step on the gated replay buffer.

    Trains on whatever payload the entry was stored with:
      * list[int] token ids (the form train._admission_gate_batch stores)
      * str, encoded via the tokenizer
    """
    batch = memory.replay(n_replay)
    if not batch:
        return None
    losses = []
    model.train()
    for entry in batch:
        ids_t = _payload_to_ids(entry.payload, tokenizer, device)
        if ids_t is None:
            continue
        prompt_ids = ids_t[:1]
        target_ids = ids_t[1:]
        out = model.loss(prompt_ids, target_ids, t_max=0.5,
                         edit_weight=edit_weight)
        losses.append(out["loss"])
    if not losses:
        return None
    loss = torch.stack(losses).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.item())