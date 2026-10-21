"""Per-step diagnostics logged to W&B: what the agent runs in its sandbox, and how saturated the GRPO groups are.

Pure Python + numpy, so it can be unit-tested without verl.

* ``parse_turn_tools(text)`` reads one Terminus-2 reply (JSON with ``commands[].keystrokes``, possibly preceded by a
  ``<think>`` block) and returns the keystrokes it sent. ``summarize_trajectory(texts)`` turns all replies of a
  trajectory into (#commands, libraries imported, executables run) -- the agent loop stores these in extra_fields.
* ``tool_metrics(...)`` -> ``rlmath/tools/*``: per-trajectory usage rates of a watch list of libraries/executables,
  commands per trajectory, and how concentrated usage is (entropy, top-k share) -- a falling entropy / rising top-1
  share means the policy is collapsing onto a few tools.
* ``group_metrics(...)`` -> ``rlmath/group/*``: per-prompt (GRPO group) mean reward distribution, the share of groups
  that are all-correct / all-wrong / zero-variance (no gradient), and the same split per task kind.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict

_THINK_END = "</think>"
_KEYSTROKES = re.compile(r'"keystrokes"\s*:\s*"((?:\\.|[^"\\])*)"', re.S)
# "from pkg.sub import X" -> pkg ; "import a.b as c, d" -> a, d (line start, after ; or a quote as in python3 -c "import x")
_FROM = re.compile(r"(?:^|[\n;\"'])\s*from\s+([A-Za-z_][A-Za-z0-9_]*)(?:\.[A-Za-z0-9_.]+)?\s+import\b")
_IMPORT = re.compile(r"(?:^|[\n;\"'])\s*import\s+([A-Za-z_][A-Za-z0-9_., ]*)")
# words that follow "import"/"from" in prose or shell, not python modules
_NOT_LIBS = {"the", "a", "an", "this", "that", "it", "here", "there", "now", "these", "each", "my", "your", "which"}

WATCH_LIBS = (
    "json", "math", "itertools", "fractions", "numpy", "sympy", "scipy", "networkx", "random", "collections",
    "functools", "decimal", "mpmath", "gmpy2", "flint", "cypari2", "sage", "z3", "pysat", "ortools", "pulp", "cvxpy",
    "highspy", "pyscipopt", "igraph", "numba", "galois",
)
WATCH_EXES = (
    "python3", "python", "cat", "echo", "ls", "bc", "sage", "gp", "gap", "M2", "Singular", "maxima", "z3", "cadical",
    "kissat", "minisat", "glpsol", "scip", "highs", "geng", "nauty-geng", "gcc", "g++", "julia", "node", "pip",
)
_EXE_SET = set(WATCH_EXES)


def parse_turn_tools(text: str | None) -> list[str]:
    """Keystrokes sent in one assistant reply (after the last </think>). Tolerates truncated / invalid JSON."""
    if not text:
        return []
    if _THINK_END in text:
        text = text.rsplit(_THINK_END, 1)[1]
    out = []
    for m in _KEYSTROKES.finditer(text):
        raw = m.group(1)
        try:
            out.append(json.loads(f'"{raw}"'))
        except ValueError:
            out.append(raw.replace("\\n", "\n").replace('\\"', '"'))
    return out


def libs_in(keystrokes: str) -> set[str]:
    found = {m.group(1) for m in _FROM.finditer(keystrokes)}
    for m in _IMPORT.finditer(keystrokes):
        for part in m.group(1).split(","):
            name = part.strip().split(" ")[0].split(".")[0]
            if name:
                found.add(name)
    return found - _NOT_LIBS


def exes_in(keystrokes: str) -> set[str]:
    found = set()
    for line in re.split(r"\n|;|&&|\|\|", keystrokes):
        words = line.strip().split()
        if words and words[0] in _EXE_SET:
            found.add(words[0])
    return found


def summarize_trajectory(texts) -> tuple[int, str, str]:
    """(#commands, comma-joined libraries, comma-joined executables) over all replies of one trajectory."""
    n, libs, exes = 0, set(), set()
    for t in texts or []:
        for ks in parse_turn_tools(t):
            n += 1
            libs |= libs_in(ks)
            exes |= exes_in(ks)
    return n, ",".join(sorted(libs)[:60]), ",".join(sorted(exes))


def _split(s) -> set[str]:
    return {x for x in str(s or "").split(",") if x}


def _entropy_bits(counts: Counter) -> float:
    tot = sum(counts.values())
    return -sum(v / tot * math.log2(v / tot) for v in counts.values() if v) if tot else 0.0


def tool_metrics(n_cmds, libs, exes, rewards, valid) -> dict:
    """Trajectory-level usage rates (share of trajectories that used X) over the valid (not infra-failed) rows."""
    idx = [i for i, v in enumerate(valid) if v]
    if not idx:
        return {}
    n = len(idx)
    cmds = [int(n_cmds[i]) for i in idx]
    lib_sets = [_split(libs[i]) for i in idx]
    exe_sets = [_split(exes[i]) for i in idx]
    lib_c, exe_c = Counter(), Counter()
    for s in lib_sets:
        lib_c.update(s)
    for s in exe_sets:
        exe_c.update(s)
    tot = sum(lib_c.values())
    top = [v for _, v in lib_c.most_common(3)]
    solved = [cmds[k] for k, i in enumerate(idx) if float(rewards[i]) >= 0.999]
    failed = [cmds[k] for k, i in enumerate(idx) if float(rewards[i]) <= 0.0]
    m = {
        "rlmath/tools/cmds_mean": sum(cmds) / n,
        "rlmath/tools/cmds_median": float(sorted(cmds)[n // 2]),
        "rlmath/tools/zero_cmd_frac": sum(c == 0 for c in cmds) / n,
        "rlmath/tools/cmds_mean_solved": sum(solved) / len(solved) if solved else 0.0,
        "rlmath/tools/cmds_mean_failed": sum(failed) / len(failed) if failed else 0.0,
        "rlmath/tools/libs_per_traj": sum(len(s) for s in lib_sets) / n,
        "rlmath/tools/no_import_frac": sum(not s for s in lib_sets) / n,
        "rlmath/tools/distinct_libs_1pct": float(sum(v >= 0.01 * n for v in lib_c.values())),
        "rlmath/tools/lib_entropy_bits": _entropy_bits(lib_c),
        "rlmath/tools/lib_top1_share": top[0] / tot if tot else 0.0,
        "rlmath/tools/lib_top3_share": sum(top) / tot if tot else 0.0,
        "rlmath/tools/exe_entropy_bits": _entropy_bits(exe_c),
    }
    for lib in WATCH_LIBS:
        m[f"rlmath/tools/lib/{lib}"] = lib_c.get(lib, 0) / n
    m["rlmath/tools/lib/other"] = sum(v for k, v in lib_c.items() if k not in WATCH_LIBS) / n
    for exe in WATCH_EXES:
        m[f"rlmath/tools/exe/{exe}"] = exe_c.get(exe, 0) / n
    return m


def group_metrics(uids, rewards, valid, kinds=None) -> dict:
    """GRPO group saturation over valid rows. A zero-variance group (all rewards equal) gives every member advantage 0."""
    groups, gkind = defaultdict(list), {}
    for i, u in enumerate(uids):
        if valid[i]:
            groups[u].append(float(rewards[i]))
            gkind[u] = str(kinds[i]) if kinds is not None else "all"
    if not groups:
        return {}
    vals = list(groups.values())
    means = sorted(sum(v) / len(v) for v in vals)
    ng = len(vals)

    def frac(pred):
        return sum(pred(v) for v in vals) / ng

    zero_var = lambda v: max(v) - min(v) < 1e-6  # noqa: E731
    n_traj = sum(len(v) for v in vals)
    m = {
        "rlmath/group/num_groups": float(ng),
        "rlmath/group/all_correct_frac": frac(lambda v: min(v) >= 0.999),
        "rlmath/group/all_wrong_frac": frac(lambda v: max(v) <= 0.0),
        "rlmath/group/zero_var_frac": frac(zero_var),
        # share of trajectories that can carry gradient (their group has reward variance)
        "rlmath/group/informative_traj_frac": sum(len(v) for v in vals if not zero_var(v)) / n_traj,
        "rlmath/group/mean_score_mean": sum(means) / ng,
    }
    for q in (10, 25, 50, 75, 90):
        m[f"rlmath/group/mean_score_p{q}"] = means[min(ng - 1, int(q / 100 * ng))]
    edges = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        inside = sum(1 for x in means if lo <= x < hi or (hi == 1.0 and x == 1.0))
        m[f"rlmath/group/mean_bin_{lo:.1f}-{hi:.1f}"] = inside / ng
    by_kind = defaultdict(list)
    for u, v in groups.items():
        by_kind[gkind[u]].append(v)
    if len(by_kind) > 1:
        for k, vs in by_kind.items():
            m[f"rlmath/group/{k}/num_groups"] = float(len(vs))
            m[f"rlmath/group/{k}/mean_score"] = sum(sum(v) / len(v) for v in vs) / len(vs)
            m[f"rlmath/group/{k}/zero_var_frac"] = sum(zero_var(v) for v in vs) / len(vs)
            m[f"rlmath/group/{k}/all_correct_frac"] = sum(min(v) >= 0.999 for v in vs) / len(vs)
            m[f"rlmath/group/{k}/all_wrong_frac"] = sum(max(v) <= 0.0 for v in vs) / len(vs)
    return m


def task_kind(instance_id) -> str:
    """'construct-rm-0704-0382-0-k1' -> 'construct' (Harbor task subset prefix)."""
    s = str(instance_id or "")
    return s.split("-", 1)[0] if "-" in s else "unknown"
