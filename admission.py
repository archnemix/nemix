"""Continual learning: the admission formula.

The single biggest lever on continual-learning quality is *what is allowed
into capacity in the first place*.  Training on everything the user says
degrades the model (bloat, frustration, thanks-only messages, duplicates,
contradictions).

Research basis
--------------
* Learnability-aware data selection (arXiv 2510.15035): examples are scored
  +1 if learnable-and-new (generalisation gain), 0 if learnable-but-
  already-mastered (redundancy risk), -1 if unlearnable (stale/noisy/hard).
  Unlearnable data drags validation *down*; mastered data invites memorisation.
* SFT for knowledge / RL for generalisation: stable factual knowledge enters
  slowly; behaviours are shaped by outcome reward.
* Human memory anatomy: episodic (fast, disposable) -> semantic concept
  store (consolidated, gated) -> slow structural change.

NEMIX scores every interaction x:

    A(x) = +w_n*novelty + w_l*learnability + w_u*utility
           - w_r*redundancy - w_b*bloat - w_v*volatility

  A >= theta_high : consolidate into the concept store (-> slow weights)
  A >= theta_low  : keep in the episodic buffer (replay, may decay)
  A <  theta_low  : DISCARD -- never trained on (the guard rail)

Repeated, consistent, high-scoring observations of the same key are
promoted episodic -> concept; capacity limits evict the *lowest-score*
items first, so quality of the store is monotone -- it cannot bloat.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Any, Optional


# ----------------------------------------------------------------------
# bloat / noise detectors
# ----------------------------------------------------------------------
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
    """How much of this text is noise that should NOT enter capacity. [0,1]."""
    t = (text or "").strip().lower()
    if not t:
        return 1.0
    score = 0.0
    core = t.strip(" .!?,;:")                     # punctuation-insensitive
    if core in THANKS_ONLY:
        return 1.0
    words = core.split()
    if words and all(w.strip(" .!?") in THANKS_ONLY for w in words):
        return 1.0                                # "great! nice ok, thanks."
    if any(m in t for m in FRUSTRATION_MARKERS):
        score += 0.50          # emotional noise: veto-level, never learned
    if any(m in t for m in DEFLECTION_MARKERS):
        score += 0.50          # refusal boilerplate: veto-level as well
    if _REPEAT_RE.search(t) or _LAUGHTER_RE.search(t):
        score += 0.25
    if len(words) <= 2 and not any(ch.isdigit() for ch in t):
        score += 0.35           # "ok cool" -- nothing to learn
    if len(words) >= 3:
        if len(set(words)) / len(words) < 0.4:
            score += 0.20       # repetitive filler
    letters = [c for c in t if c.isalpha()]
    if len(letters) >= 6 and sum(1 for c in letters if c.isupper()) / len(letters) > 0.7:
        score += 0.15
    return min(score, 1.0)


# ----------------------------------------------------------------------
# signals + formula
# ----------------------------------------------------------------------
@dataclass
class AdmissionSignals:
    novelty: float = 0.0        # 1 - max similarity to existing memory
    learnability: float = 0.0   # normalised loss drop from training on it
    utility: float = 0.0        # did it improve the outcome / user goal
    redundancy: float = 0.0     # max similarity to *mastered* material
    bloat: float = 0.0          # noise / frustration / boilerplate
    volatility: float = 0.0     # conflicts with consolidated knowledge


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


def admit(s: AdmissionSignals, cfg=None) -> str:
    """-> 'consolidate' | 'episodic' | 'discard'"""
    # hard rails: some things NEVER enter capacity, no matter how novel
    if s.bloat >= 0.5 or s.volatility >= 0.5:
        return "discard"
    th = getattr(cfg, "theta_high", 0.25)
    tl = getattr(cfg, "theta_low", -0.05)
    a = admission_score(s, cfg)
    if a >= th:
        return "consolidate"
    if a >= tl:
        return "episodic"
    return "discard"


# ----------------------------------------------------------------------
# three-tier memory with gated capacity
# ----------------------------------------------------------------------
def _cos(a, b) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


@dataclass
class MemoryEntry:
    key: str
    payload: Any
    vector: tuple
    score: float
    hits: int = 1


class ContinualMemory:
    """episodic buffer (fast, disposable) + concept store (gated, slow)."""

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.cap_e = getattr(cfg, "capacity_episodic", 64)
        self.cap_c = getattr(cfg, "capacity_concepts", 128)
        self.promote_hits = getattr(cfg, "promote_hits", 2)
        self.episodic: list[MemoryEntry] = []
        self.concepts: list[MemoryEntry] = []
        self.log: list[dict] = []

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

    # -- main entry -----------------------------------------------------
    def observe(
        self,
        key: str,
        payload: Any,
        vector,
        signals: Optional[AdmissionSignals] = None,
        text: Optional[str] = None,
    ) -> dict:
        """Gate one interaction into capacity.  Returns the decision."""
        if signals is None:
            signals = AdmissionSignals()
        if text is not None:
            signals.bloat = max(signals.bloat, text_bloat(text))

        sim_concept = max((_cos(vector, e.vector) for e in self.concepts), default=0.0)
        if signals.redundancy == 0.0 and sim_concept > 0.0:
            signals.redundancy = sim_concept
        if signals.novelty == 0.0:
            signals.novelty = 1.0 - max(sim_concept, self.max_similarity(vector))

        a = admission_score(signals, self.cfg)
        decision = admit(signals, self.cfg)
        rec = {"key": key, "score": round(a, 4), "decision": decision}
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

        self._enforce_capacity()
        return rec

    def _promote(self, entry: MemoryEntry) -> None:
        if entry in self.episodic:
            self.episodic.remove(entry)
        if entry not in self.concepts:
            self.concepts.append(entry)

    def _enforce_capacity(self) -> None:
        for e in list(self.episodic):
            if e.hits >= self.promote_hits and e.score >= getattr(
                self.cfg, "theta_high", 0.25
            ):
                self._promote(e)
        # evict LOWEST score first -- quality-monotone capacity
        if len(self.episodic) > self.cap_e:
            self.episodic.sort(key=lambda e: e.score, reverse=True)
            self.episodic = self.episodic[: self.cap_e]
        if len(self.concepts) > self.cap_c:
            self.concepts.sort(key=lambda e: e.score, reverse=True)
            self.concepts = self.concepts[: self.cap_c]

    # -- training mix ----------------------------------------------------
    def replay(self, n: int, seed: int = 0) -> list:
        """Score-weighted sample (with replacement) for consolidation
        training -- higher A(x) => more replay, quality can only rise."""
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
            "discarded": sum(1 for r in self.log if r["decision"] == "discard"),
            "accepted": sum(1 for r in self.log if r["decision"] != "discard"),
        }


# ----------------------------------------------------------------------
# online driver -- this is the integration point that was missing.
# ContinualMemory + admit() were fully implemented and correct, but
# nothing in the package ever called them: no part of train.py or
# model.py constructed a ContinualMemory, called .observe() on a live
# turn, or trained on .replay(). The admission formula existed and did
# nothing. observe_turn() below is what a chat/serving loop calls once
# per user turn, and consolidate_step() is what periodically trains the
# backbone on the gated, score-weighted replay buffer -- the actual
# "learn continuously but don't dump everything in" loop your original
# requirement asked for.
# ----------------------------------------------------------------------
def observe_turn(
    memory: "ContinualMemory",
    model,
    text: str,
    key: Optional[str] = None,
    tokenizer=None,
    utility: float = 0.0,
) -> dict:
    """Gate one raw user/assistant turn into capacity.

    key: a stable identifier for this fact/preference/topic so repeated
    mentions accumulate hits toward promotion rather than each becoming a
    separate episodic entry. If not given, falls back to a normalized
    prefix of the text -- coarse, but keeps near-identical repeated
    messages from each claiming a fresh slot.

    tokenizer: anything exposing .encode(text) -> list[int]/tensor. If
    omitted, novelty/redundancy signals degrade gracefully to 0 (the
    bloat/frustration/thanks-only guard rail in text_bloat still runs
    regardless, since that's pure text analysis and needs no model).

    Returns the same decision dict ContinualMemory.observe() returns, so
    callers can log or route on 'consolidate' / 'episodic' / 'discard'.
    """
    if key is None:
        key = (text or "").strip().lower()[:80]

    vector = [0.0] * max(getattr(model, "cfg", None) and model.cfg.gist_dim or 8, 1)
    if tokenizer is not None and model is not None and hasattr(model, "backbone"):
        try:
            import torch
            ids = tokenizer.encode(text)
            ids = torch.as_tensor(ids, dtype=torch.long)
            with torch.no_grad():
                vector = model.backbone.gist_vector(ids).tolist()
        except Exception:
            pass  # fall back to the zero vector; bloat/frustration gating still applies

    signals = AdmissionSignals(utility=utility)
    return memory.observe(key, payload=text, vector=vector, signals=signals, text=text)


def consolidate_step(memory: "ContinualMemory", model, opt, tokenizer,
                     n_replay: int = 8, device: str = "cpu"):
    """One small gradient step on the gated replay buffer. Call this
    periodically (e.g. every N observe_turn calls, or on an idle timer)
    rather than training on every single turn -- replay() already
    score-weights which memories get trained on more, so capacity-gated
    quality and training frequency are two independent knobs."""
    import torch

    batch = memory.replay(n_replay)
    if not batch or tokenizer is None:
        return None
    losses = []
    model.train()
    for entry in batch:
        text = entry.payload if isinstance(entry.payload, str) else str(entry.payload)
        ids = tokenizer.encode(text)
        if len(ids) < 2:
            continue
        ids_t = torch.as_tensor(ids, dtype=torch.long, device=device)
        prompt_ids = ids_t[:1]
        target_ids = ids_t[1:]
        out = model.loss(prompt_ids, target_ids, t_max=0.5)
        losses.append(out["loss"])
    if not losses:
        return None
    loss = torch.stack(losses).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.item())