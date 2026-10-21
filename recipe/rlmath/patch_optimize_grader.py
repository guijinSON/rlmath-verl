"""Swap the beat-best grader (graders/optimize_run.py) into Harbor optimize task dirs.

    python -m recipe.rlmath.patch_optimize_grader <tasks_root> [--dry-run] [--force]

Patches tests/run.py of every optimize task under <tasks_root> IN PLACE, so run it on a fresh copy of a
dataset (cp -a src dst), never on the task dirs of a run in progress. Only files identical to the stock
aav run.py (STOCK_SHA256) are replaced; anything else is reported and left alone unless --force.
Symlinked task dirs / run.py files are always skipped (they point at shared copies).

The patched grader still writes a single reward (1.5 on a strict beat of best_known), so any trainer can use
the tasks; the opt_* diagnostics in verifier/details.json reach W&B only with the 2026-10-21+ recipe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

STOCK_SHA256 = "d364958443f46cf3ebb71808afc270e23203f62550246eb5008a847a8701bd73"  # aav snapshot-2026-10-20
GRADER = Path(__file__).resolve().parent / "graders" / "optimize_run.py"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _is_optimize(task: Path) -> bool:
    inst = task / "tests" / "instance.json"
    if inst.is_file():
        try:
            return json.loads(inst.read_text()).get("subset") == "optimize"
        except (OSError, ValueError):
            return False
    return task.name.startswith("optimize-")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="also replace run.py files that differ from the stock one")
    a = ap.parse_args(argv)

    new_sha = _sha(GRADER)
    counts = {"patched": 0, "already": 0, "not_optimize": 0, "symlink": 0, "unknown_runpy": 0, "no_runpy": 0}
    unknown = []
    for task in sorted(p for p in a.root.iterdir() if p.is_dir()):
        run_py = task / "tests" / "run.py"
        if task.is_symlink() or run_py.is_symlink():
            counts["symlink"] += 1
            continue
        if not _is_optimize(task):
            counts["not_optimize"] += 1
            continue
        if not run_py.is_file():
            counts["no_runpy"] += 1
            continue
        sha = _sha(run_py)
        if sha == new_sha:
            counts["already"] += 1
            continue
        if sha != STOCK_SHA256 and not a.force:
            counts["unknown_runpy"] += 1
            unknown.append(task.name)
            continue
        if not a.dry_run:
            shutil.copyfile(GRADER, run_py)
        counts["patched"] += 1
    print(json.dumps({"root": str(a.root), "dry_run": a.dry_run, **counts}))
    if unknown:
        print("run.py differs from the stock grader (left alone; --force to replace):", ", ".join(unknown[:20]),
              "..." if len(unknown) > 20 else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
