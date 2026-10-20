"""Deliberation: K diverse drafts + external verification (S*-style)."""
from __future__ import annotations


def deliberate(model, prompt_ids, out_len, verifier=None, k=4, budget=8,
               logit_noise=0.6, **gen_kw):
    """K diverse drafts + external verification; return the first verified
    draft immediately, or else the single highest-scoring draft seen.

    Previously the fallback best-tracking used:
        if best is None or (score > 0 and score > history[-2]["score"]
                             if len(history) > 1 else True): ...
    Due to Python's ternary precedence this parses as
        best is None or (score > 0 and (score > history[-2]["score"]
                                        if len(history) > 1 else True))
    which does two wrong things: (1) it compares the CURRENT draft's
    score against the PREVIOUS draft's score instead of against the best
    score seen so far, so a mediocre draft could beat a strong earlier
    one simply by slightly beating whatever happened to run right before
    it; (2) whenever len(history) <= 1 the whole clause collapses to
    `score > 0`, unconditionally overwriting best with the very first
    low-scoring draft. Net effect: when no draft passes verification,
    the function's fallback pick was not reliably the best-scoring one.
    Replaced with an explicit best_score tracked directly against every
    draft's own score -- no history indexing, no ternary ambiguity."""
    best = None
    best_ok = False
    best_score = float("-inf")
    calls = 0
    history = []
    for i in range(k):
        out = model.generate(prompt_ids, out_len=out_len,
                             logit_noise=logit_noise, seed=i, **gen_kw)
        tokens = out["tokens"]
        ok, score = False, 0.0
        if verifier is not None and calls < budget:
            v = verifier(tokens)
            calls += 1
            ok = bool(v.get("ok"))
            score = float(v.get("score", 1.0 if ok else 0.0))
        history.append({"draft": i, "ok": ok, "score": score, "tokens": tokens})
        if ok:
            return {"tokens": tokens, "ok": True, "history": history,
                    "verify_calls": calls}
        if best is None or score > best_score:
            best, best_ok, best_score = tokens, ok, score
    return {"tokens": best, "ok": best_ok, "history": history, "verify_calls": calls}