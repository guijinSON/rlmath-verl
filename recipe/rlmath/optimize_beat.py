"""Beat-best bookkeeping for optimize tasks (2026-10-21).

The beat-best grader (graders/optimize_run.py) scores 1.5 when an answer strictly improves on best_known
and writes opt_* flags to verifier/details.json; the remote agent loop adds them to the trial's rewards
dict (next to "reward", which alone is the training reward). This module turns them into
* per-trajectory ``reward_extra_info`` (every trajectory gets the same keys, 0.0 when absent), which
  verl's validation turns into ``val-aux/<data_source>/opt_beat_best/mean@N`` etc.;
* training metrics ``rlmath/optimize/*`` over the valid optimize trajectories of a step.
Witnesses of beats are copied by the remote agent loop (RLMATH_BEAT_DIR, beats.jsonl).
"""
from __future__ import annotations

OPT_FLAGS = ("opt_beat_best", "opt_reach_best", "opt_counterexample")


def optimize_flags(rewards) -> dict[str, float]:
    """reward_extra_info for one trajectory: the opt_* flags of its Harbor rewards dict (0.0 if absent)."""
    rewards = rewards if isinstance(rewards, dict) else {}
    out = {}
    for k in OPT_FLAGS:
        try:
            out[k] = float(rewards.get(k, 0.0) or 0.0)
        except (TypeError, ValueError):
            out[k] = 0.0
    return out


def optimize_metrics(kinds, beat, reach, cex, valid, totals: dict | None = None) -> dict:
    """rlmath/optimize/* for one training step. kinds = task_kind per trajectory ('optimize', ...);
    beat/reach/cex = opt_* flags; valid = not infra-failed. `totals` (mutated) keeps running counts."""
    idx = [i for i in range(len(valid)) if valid[i] and str(kinds[i]) == "optimize"]
    n = len(idx)
    nb = sum(float(beat[i]) > 0 for i in idx)
    nr = sum(float(reach[i]) > 0 for i in idx)
    nc = sum(float(cex[i]) > 0 for i in idx)
    m = {
        "rlmath/optimize/trajs": float(n),
        "rlmath/optimize/reach_best_frac": nr / n if n else 0.0,
        "rlmath/optimize/beat_best_frac": nb / n if n else 0.0,
        "rlmath/optimize/beat_best_count": float(nb),
        "rlmath/optimize/counterexample_count": float(nc),
    }
    if totals is not None:
        totals["beat"] = totals.get("beat", 0) + nb
        totals["cex"] = totals.get("cex", 0) + nc
        m["rlmath/optimize/beat_best_cum"] = float(totals["beat"])
        m["rlmath/optimize/counterexample_cum"] = float(totals["cex"])
    return m
