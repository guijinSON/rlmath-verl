"""Tests for the optimize beat-best grader and its bookkeeping (no GPU, no Harbor, no verl import).

    python -m pytest recipe/rlmath/test_optimize_beat.py -q
"""
import json
import sys
import types
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
GRADER_SRC = (HERE / "graders" / "optimize_run.py").read_text()


def _grader(verifier_dir: Path):
    """The grader module with /logs/verifier redirected to a temp dir."""
    mod = types.ModuleType("run")
    exec(compile(GRADER_SRC.replace("/logs/verifier", str(verifier_dir)), "optimize_run.py", "exec"), mod.__dict__)
    return mod


def _inst(**kw):
    base = {"subset": "optimize", "direction": "maximize", "baseline": 2, "best_known": 3}
    base.update(kw)
    return base


def _value_check(inst, ans):
    return ans["v"], {}


@pytest.fixture
def g(tmp_path):
    return _grader(tmp_path)


@pytest.mark.parametrize(
    "inst,value,reward,beat,reach",
    [
        (_inst(), 2, 0.1, 0, 0),  # baseline
        (_inst(), 3, 1.0, 0, 1),  # matches best_known
        (_inst(), 4, 1.5, 1, 1),  # beats it
        (_inst(direction="minimize", baseline=10, best_known=5), 6, 0.1 + 0.9 * 0.8, 0, 0),
        (_inst(direction="minimize", baseline=10, best_known=5), 5, 1.0, 0, 1),
        (_inst(direction="minimize", baseline=10, best_known=5), 4, 1.5, 1, 1),
        (_inst(), Fraction(7, 2), 1.5, 1, 1),  # exact rational beat
        (_inst(), Fraction(3, 1), 1.0, 0, 1),
        (_inst(), np.int64(4), 1.5, 1, 1),  # numpy ints are exact
        # float objective: rounding-level "improvements" do not count
        (_inst(baseline=0.5, best_known=0.6178), 0.61785, 1.0, 0, 1),
        (_inst(baseline=0.5, best_known=0.6178), 0.6185, 1.5, 1, 1),
        (_inst(baseline=0.5, best_known=0.6178, beat_tol=0.0), 0.61785, 1.5, 1, 1),
        # log-scale loss -> 0: a real improvement near zero beats (absolute tolerance only)
        (_inst(direction="minimize", baseline=1.0, best_known=1e-6, progress_scale="log", log_floor=1e-12), 5e-7, 1.5, 1, 1),
    ],
)
def test_grade(g, inst, value, reward, beat, reach):
    r, det = g.grade(inst, {"v": value}, _value_check)
    assert r == pytest.approx(reward)
    assert det["opt_beat_best"] == beat and det["opt_reach_best"] == reach
    assert all(np.isfinite(det[k]) for k in det if k.startswith("opt_"))


def test_counterexample_flag(g):
    r, det = g.grade(_inst(counterexample_at=7), {"v": 7}, _value_check)
    assert r == 1.5 and det["opt_counterexample"] == 1
    _, det = g.grade(_inst(counterexample_at=7), {"v": 6}, _value_check)
    assert det["opt_counterexample"] == 0


def test_invalid_and_construct_unchanged(g):
    def bad(inst, ans):
        raise g.Invalid("nope")

    assert g.grade(_inst(), {"v": 9}, bad)[0] == 0.0
    assert g.grade({"subset": "construct"}, {"v": 1}, _value_check) == (1.0, {"valid": True})


def test_diagnostics_never_change_stock_reward(g):
    inst = _inst()
    del inst["best_known"]  # beat_fields fails -> falls back to optimize_reward, which also needs it
    r, det = g.grade(inst, {"v": 3}, _value_check)
    assert r == 0.0 and det["valid"] is False  # same as the stock grader (KeyError -> invalid)


def test_main_writes_reward_json_and_witness(tmp_path, monkeypatch):
    vdir = tmp_path / "verifier"
    vdir.mkdir()
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "run.py").write_text(GRADER_SRC.replace("/logs/verifier", str(vdir)))
    (tests / "check.py").write_text("def check(inst, ans):\n    return ans['v'], {'n': ans['v']}\n")
    (tests / "instance.json").write_text(json.dumps(_inst(counterexample_at=7)))
    ans = tmp_path / "answer.json"
    ans.write_text(json.dumps({"v": 4}))
    monkeypatch.setattr(sys, "argv", ["run.py", str(tests / "instance.json"), str(ans)])
    monkeypatch.syspath_prepend(str(tests))
    for m in ("run", "check"):
        sys.modules.pop(m, None)
    import run  # noqa: E402

    run.main()
    rj = json.loads((vdir / "reward.json").read_text())
    assert rj["reward"] == 1.5 and rj["opt_beat_best"] == 1 and rj["opt_counterexample"] == 0
    assert set(rj) == {"reward", "opt_beat_best", "opt_reach_best", "opt_counterexample", "opt_progress",
                       "opt_raw_score", "opt_best_known", "opt_improvement"}
    assert json.loads((vdir / "beat_answer.json").read_text()) == {"v": 4}
    for m in ("run", "check"):
        sys.modules.pop(m, None)


def test_optimize_flags_and_metrics():
    from recipe.rlmath.optimize_beat import optimize_flags, optimize_metrics

    assert optimize_flags({"reward": 1.5, "opt_beat_best": 1, "opt_reach_best": 1}) == {
        "opt_beat_best": 1.0, "opt_reach_best": 1.0, "opt_counterexample": 0.0}
    assert optimize_flags({"reward": 1.0}) == {"opt_beat_best": 0.0, "opt_reach_best": 0.0, "opt_counterexample": 0.0}
    assert optimize_flags(None) == {"opt_beat_best": 0.0, "opt_reach_best": 0.0, "opt_counterexample": 0.0}

    kinds = ["optimize", "optimize", "optimize", "construct"]
    totals = {}
    m = optimize_metrics(kinds, [1, 0, 1, 1], [1, 1, 1, 1], [0, 0, 1, 0], [True, True, False, True], totals)
    assert m["rlmath/optimize/trajs"] == 2  # infra-failed and construct rows excluded
    assert m["rlmath/optimize/beat_best_count"] == 1 and m["rlmath/optimize/beat_best_frac"] == 0.5
    assert m["rlmath/optimize/reach_best_frac"] == 1.0 and m["rlmath/optimize/counterexample_count"] == 0
    m = optimize_metrics(kinds, [1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [True] * 4, totals)
    assert m["rlmath/optimize/beat_best_cum"] == 3


def test_group_bins_hold_beat_rewards():
    from recipe.rlmath.tool_metrics import group_metrics

    m = group_metrics(["a", "a", "b", "b"], [1.5, 1.0, 0.0, 0.2], [True] * 4)
    assert sum(v for k, v in m.items() if k.startswith("rlmath/group/mean_bin_")) == pytest.approx(1.0)


def test_harbor_reward_uses_reward_key():
    import ast

    src = (HERE.parent / "agentic" / "agent_loop" / "remote_agent_loop.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_harbor_reward")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "remote_agent_loop.py", "exec"), ns)
    assert ns["_harbor_reward"]({"reward": 1.5, "opt_beat_best": 1, "opt_raw_score": 4.0}) == 1.5
    assert ns["_harbor_reward"]({"a": 0.25, "b": 0.5}) == 0.75  # legacy multi-key dicts still sum
