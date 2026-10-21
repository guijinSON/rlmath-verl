"""rlmath runtime grader. Copied into every task as /tests/run.py next to /tests/check.py.

Deterministic; no LLM in the reward path. Runs with python3 + numpy/scipy/sympy/networkx only.

    python3 /tests/run.py /tests/instance.json /workdir/answer.json

Prints the reward and writes /logs/verifier/details.json when that directory exists.

check.py must define  check(inst, ans) -> (value, extra_dict)  and raise Invalid for a bad answer.
  construct : value is ignored; a returned check means the witness is valid -> reward 1.0
  optimize  : value is the raw objective; reward = 0.1 + 0.9 * progress, where
              progress = clip((s - baseline) / (best_known - baseline), 0, 1)       (maximize)
                         clip((baseline - s) / (baseline - best_known), 0, 1)       (minimize)
              inst must carry "direction", "baseline", "best_known".

Beat-best variant (rlmath, 2026-10-21; drop-in for the stock run.py of optimize tasks):
  - an answer that STRICTLY improves on best_known gets reward 1 + BEAT_BONUS (= 1.5) instead of 1.0.
    Exact objectives (int / Fraction value, int best_known) must beat it outright; float objectives must
    beat it by more than max(FLOAT_ABS_TOL, FLOAT_REL_TOL * |best_known|), so a published value rounded to
    a few digits cannot be "beaten" by its own unrounded form. inst["beat_tol"] overrides the tolerance.
  - optimize tasks also write /logs/verifier/reward.json (Harbor reads it before reward.txt):
      reward, opt_beat_best, opt_reach_best, opt_counterexample (0/1), opt_progress (unclipped),
      opt_raw_score, opt_best_known, opt_improvement (signed, positive = better than best_known).
    Only "reward" is the training reward; the opt_* keys are diagnostics. A trainer that SUMS every
    Harbor reward key (the stock remote agent loop before 2026-10-21) must not use these tasks.
  - on a beat, the submitted answer is copied to /logs/verifier/beat_answer.json for human review.
"""
import json
import math
import numbers
import os
import shutil
import sys
from fractions import Fraction

VALID_FLOOR = 0.1
BEAT_BONUS = 0.5         # reward for strictly improving on best_known = 1.0 + BEAT_BONUS
FLOAT_REL_TOL = 1e-4     # float objectives: required improvement, relative to |best_known| ...
FLOAT_ABS_TOL = 1e-9     # ... and never below this absolute margin


class Invalid(Exception):
    pass


# ------------------------------------------------------------ input helpers

def int_list(v, n=None, lo=None, hi=None, name="list"):
    if not isinstance(v, list):
        raise Invalid(f"{name} must be a list")
    if n is not None and len(v) != n:
        raise Invalid(f"{name} must have length {n}, got {len(v)}")
    for x in v:
        if isinstance(x, bool) or not isinstance(x, int):
            raise Invalid(f"{name} must contain integers only")
        if (lo is not None and x < lo) or (hi is not None and x > hi):
            raise Invalid(f"{name} entry {x} out of range [{lo}, {hi}]")
    return v


def as_int(x, name="value", lo=None, hi=None):
    return int_list([x], 1, lo, hi, name)[0]


def num(x, name="value"):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise Invalid(f"{name} must be a number")
    if not math.isfinite(x):
        raise Invalid(f"{name} is not finite")
    return float(x)


def float_matrix(v, rows=None, cols=None, name="matrix"):
    if not isinstance(v, list) or (rows is not None and len(v) != rows):
        raise Invalid(f"{name} must be a list of {rows} rows")
    out = []
    for r in v:
        if not isinstance(r, list) or (cols is not None and len(r) != cols):
            raise Invalid(f"each row of {name} must have {cols} numbers")
        out.append([num(x, name) for x in r])
    return out


def float_list(v, n=None, name="list"):
    if not isinstance(v, list) or (n is not None and len(v) != n):
        raise Invalid(f"{name} must be a list of length {n}")
    return [num(x, name) for x in v]


# ------------------------------------------------------------------ reward

def optimize_reward(inst, s):
    b, best = inst["baseline"], inst["best_known"]
    if inst.get("progress_scale") == "log":
        # for non-negative losses spanning orders of magnitude (e.g. overlap loss -> 0);
        # progress is measured in log10(value + floor)
        f = inst.get("log_floor", 1e-3)
        b, best, s = (math.log10(max(x, 0.0) + f) for x in (b, best, s))
    if inst["direction"] == "maximize":
        p = (s - b) / (best - b)
    else:
        p = (b - s) / (b - best)
    return VALID_FLOOR + (1 - VALID_FLOOR) * max(0.0, min(1.0, p))


def _exact(x):
    return isinstance(x, numbers.Rational) and not isinstance(x, bool)


def improvement(inst, value):
    """Signed improvement of `value` over best_known in the objective's own units (positive = better)."""
    best = inst["best_known"]
    if _exact(value) and _exact(best):
        d = Fraction(int(value)) if isinstance(value, numbers.Integral) else Fraction(value)
        d -= Fraction(int(best)) if isinstance(best, numbers.Integral) else Fraction(best)
    else:
        d = float(value) - float(best)
    return d if inst["direction"] == "maximize" else -d


def beat_tolerance(inst, value):
    if "beat_tol" in inst:
        return float(inst["beat_tol"])
    if _exact(value) and _exact(inst["best_known"]):
        return 0
    return max(FLOAT_ABS_TOL, FLOAT_REL_TOL * abs(float(inst["best_known"])))


def optimize_progress(inst, s):
    """Unclipped progress (1.0 = best_known), in the same scale optimize_reward uses."""
    b, best = inst["baseline"], inst["best_known"]
    if inst.get("progress_scale") == "log":
        f = inst.get("log_floor", 1e-3)
        b, best, s = (math.log10(max(x, 0.0) + f) for x in (b, best, s))
    if inst["direction"] == "maximize":
        return (s - b) / (best - b)
    return (b - s) / (b - best)


def beat_fields(inst, value):
    """opt_* diagnostics for a valid optimize answer (all finite numbers, as Harbor requires)."""
    s = float(value)
    gain = improvement(inst, value)
    tol = beat_tolerance(inst, value)
    try:
        p = float(optimize_progress(inst, s))
    except (ZeroDivisionError, ValueError, OverflowError):
        p = 0.0
    c = inst.get("counterexample_at")
    if c is None:
        cex = 0
    elif inst["direction"] == "maximize":
        cex = int(s >= float(c))
    else:
        cex = int(s <= float(c))
    return {
        "opt_beat_best": int(gain > tol),
        "opt_reach_best": int(gain >= -tol),
        "opt_counterexample": cex,
        "opt_progress": p if math.isfinite(p) else 0.0,
        "opt_raw_score": s,
        "opt_best_known": float(inst["best_known"]),
        "opt_improvement": float(gain),
    }


def grade(inst, ans, check):
    try:
        if not isinstance(ans, dict):
            raise Invalid("answer must be a JSON object")
        value, extra = check(inst, ans)
        if inst.get("subset") == "optimize":
            s = float(value)
            if not math.isfinite(s):
                raise Invalid("non-finite score")
            try:
                fields = beat_fields(inst, value)
            except Exception:  # noqa: BLE001 - diagnostics must never change the stock reward
                fields = {}
            reward = 1.0 + BEAT_BONUS if fields.get("opt_beat_best") else optimize_reward(inst, s)
            return reward, {"valid": True, "raw_score": s, **fields, **(extra or {})}
        return 1.0, {"valid": True, **(extra or {})}
    except Invalid as e:
        return 0.0, {"valid": False, "error": str(e)}
    except Exception as e:  # any malformed answer
        return 0.0, {"valid": False, "error": f"{type(e).__name__}: {e}"}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    # import ourselves as `run` so check.py's `from run import Invalid` is the same class
    import run as R  # noqa: E402
    import check as family  # noqa: E402

    inst = json.load(open(sys.argv[1]))
    try:
        ans = json.load(open(sys.argv[2]))
    except Exception as e:
        reward, det = 0.0, {"valid": False, "error": f"unreadable answer: {e}"}
    else:
        reward, det = R.grade(inst, ans, family.check)
    det["reward"] = reward
    if os.path.isdir("/logs/verifier"):
        with open("/logs/verifier/details.json", "w") as f:
            json.dump(det, f)
        if inst.get("subset") == "optimize":
            if det.get("opt_beat_best"):
                shutil.copyfile(sys.argv[2], "/logs/verifier/beat_answer.json")
            out = {"reward": reward}
            for k in ("opt_beat_best", "opt_reach_best", "opt_counterexample", "opt_progress",
                      "opt_raw_score", "opt_best_known", "opt_improvement"):
                out[k] = det.get(k, 0)
            with open("/logs/verifier/reward.json.tmp", "w") as f:
                json.dump(out, f)
            os.replace("/logs/verifier/reward.json.tmp", "/logs/verifier/reward.json")
    print(reward)


if __name__ == "__main__":
    main()
