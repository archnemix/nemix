"""Deliberation: K diverse drafts + external verification (S*-style)."""
from __future__ import annotations


def deliberate(model, prompt_ids, out_len, verifier=None, k=4, budget=8,
               logit_noise=0.6, **gen_kw):
    best = None
    best_ok = False
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
        if best is None or (score > 0 and score > history[-2]["score"] if len(history) > 1 else True):
            best, best_ok = tokens, ok
    return {"tokens": best, "ok": best_ok, "history": history, "verify_calls": calls}