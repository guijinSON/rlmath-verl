"""Harbor task dirs -> verl parquet, replacing recipe agentic_main._materialize_harbor_datasets.

Why: the recipe caches ``<cache>/<split>-<md5(root.resolve(), split, limit)>.parquet`` and reuses it
while ``data.harbor_overwrite`` is false (recipe/agentic/dataset/local_harbor.py:159-205). The key
ignores the directory *contents*, so if val_tasks/ is rebuilt with other tasks, a stale parquet
keeps serving the old instance_ids and ``local_task_path`` values. RemoteAgentLoop._resolve_task_path
(remote_agent_loop.py:219-249) tries, in order:
  1. local_task_path
  2. <train_harbor_dir>/<id>
  3. <val_harbor_dir>/<id>
  4. the template (default /home/verl/dataset-tasks/{id})
For a stale row whose id is no longer under val_tasks/, and no longer under the train dir
(tasks_train -> tasks_nontrivial), every step fails and it lands on the template:
FileNotFoundError, reward 0.

Here:
* the cache key covers the task list itself (id, real path, task.toml/instruction.md mtimes), so a
  changed dir always produces a new parquet;
* ``local_task_path`` (and ``extra_info.local_path``) are ``os.path.realpath``: val rows point at
  data/tasks/<id> directly, not through the val_tasks/ symlink;
* every row's path is checked at startup; a missing one is a hard error, not a silent 0 reward.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from omegaconf import OmegaConf


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return -1.0


def _arrow_safe(x):
    # task.toml metadata can mix floats with ints > 2**53 (optimize baseline/best_known, 10-18): pyarrow infers
    # double for the column and refuses the inexact int -> store those as float (metadata only; verifiers ignore it)
    if isinstance(x, dict):
        return {k: _arrow_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_arrow_safe(v) for v in x]
    if isinstance(x, int) and not isinstance(x, bool) and abs(x) > 2**53:
        return float(x)
    return x


def build_rows(root: str, split: str, data_source: str, ability: str, limit: int | None) -> list[dict]:
    from recipe.agentic.dataset.local_harbor import build_rows_from_local

    rows = [_arrow_safe(r) for r in build_rows_from_local(root, split=split, data_source=data_source, ability=ability, limit=limit)]
    for r in rows:
        real = os.path.realpath(r["local_task_path"])
        r["local_task_path"] = real
        r["extra_info"]["local_path"] = real
    return rows


def rows_signature(rows: list[dict], split: str, root: str, limit, data_source: str, ability: str) -> str:
    items = [
        (
            r["instance_id"],
            r["local_task_path"],
            _mtime(Path(r["local_task_path"]) / "task.toml"),
            _mtime(Path(r["local_task_path"]) / "instruction.md"),
        )
        for r in rows
    ]
    raw = json.dumps([split, str(Path(root).resolve()), limit, data_source, ability, items], sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def check_rows(rows: list[dict], split: str) -> None:
    missing = [r["instance_id"] for r in rows if not Path(r["local_task_path"]).is_dir()]
    if missing:
        raise FileNotFoundError(f"[rlmath-data] {split}: {len(missing)} task dirs missing, e.g. {missing[:5]}")


def materialize_harbor_datasets(config) -> dict[str, str]:
    data = config.get("data", {}) or {}
    cache_dir = Path(os.path.expanduser(str(data.get("harbor_cache_dir", "~/.cache/verl/agentic/harbor"))))
    data_source = data.get("harbor_data_source", "harbor")
    ability = data.get("harbor_ability", "agent")
    out_paths: dict[str, str] = {}
    for split, root_key, limit_key, files_key in (
        ("train", "train_harbor_dir", "harbor_train_limit", "train_files"),
        ("val", "val_harbor_dir", "harbor_val_limit", "val_files"),
    ):
        root = data.get(root_key)
        if not root:
            continue
        root = os.path.expanduser(str(root))
        limit = data.get(limit_key)
        rows = build_rows(root, split, data_source, ability, limit)
        if split == "val":
            # per-task-type val metrics (10-18): val-core/harbor_<construct|optimize|shortform>/reward/mean@1;
            # data_source only groups metrics -- the Harbor verifier reward ignores it
            for r in rows:
                r["data_source"] = f"{data_source}_{r['instance_id'].split('-')[0]}"
        if not rows:
            raise ValueError(f"[rlmath-data] no harbor tasks under {root}")
        check_rows(rows, split)
        sig = rows_signature(rows, split, root, limit, data_source + ("_bytype" if split == "val" else ""), ability)
        out = cache_dir / f"{split}-rlmath-{sig}.parquet"
        if not out.exists():
            from datasets import Dataset

            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(f".tmp{os.getpid()}")
            Dataset.from_list(rows).to_parquet(str(tmp))
            os.replace(tmp, out)
        via_symlink = sum(1 for r in rows if os.path.islink(os.path.join(root, r["instance_id"])))
        print(
            f"[rlmath-data] {split}: {len(rows)} tasks from {root} ({via_symlink} via symlink, local_task_path=realpath) "
            f"-> {out}",
            flush=True,
        )
        OmegaConf.update(config, f"data.{files_key}", [str(out)], merge=False)
        out_paths[split] = str(out)
    return out_paths
