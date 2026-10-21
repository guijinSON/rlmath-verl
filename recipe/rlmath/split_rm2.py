"""Train/val split of amphora/researchmath2-datav0.1-imagev0.3 held out BY PAPER (source).

val: 128 tasks, one per question (family), from papers with <= 3 tasks, stratified by subset
     (construct 88 / optimize 20 / short_form 20). Every other task of a held-out paper is dropped, so no
     variant of a val question, and nothing else from the same paper, is trained on.
train: all remaining tasks. Task dirs are COPIED (Harbor rejects symlinks that point outside the task).
Agent timeout: train 7200 s, val 3600 s (the dataset's 900 s counts LLM generation time, which at hundreds of
concurrent trials can take minutes per 32k-token turn).
"""
import json, random, re, shutil, sys
from pathlib import Path
import pandas as pd

src = Path(sys.argv[1]); out_train = Path(sys.argv[2]); out_val = Path(sys.argv[3])
TARGET = {"construct": 88, "optimize": 20, "short_form": 20}
rng = random.Random(20261006)
d = pd.read_parquet(src / "data/train.parquet")
per_src = d.groupby("source").size()
small = set(per_src[per_src <= 3].index)
val_rows, held_sources = [], set()
for subset, k in TARGET.items():
    fams = d[(d.subset == subset) & d.source.isin(small - held_sources)].groupby("family")
    fam_list = sorted(fams.groups)
    rng.shuffle(fam_list)
    picked = 0
    for fam in fam_list:
        g = d[d.family == fam]
        if g.source.iloc[0] in held_sources and False:
            continue
        val_rows.append(g.sample(1, random_state=rng.randint(0, 10**6)).iloc[0])
        held_sources.add(g.source.iloc[0])
        picked += 1
        if picked == k:
            break
val = pd.DataFrame(val_rows)
train = d[~d.source.isin(held_sources)]
dropped = d[d.source.isin(held_sources) & ~d.task_id.isin(val.task_id)]
assert not set(val.family) & set(train.family) and not set(val.source) & set(train.source)


def copy(rows, dst, timeout):
    dst.mkdir(parents=True, exist_ok=True)
    for tid in rows.task_id:
        t = dst / tid
        if t.exists():
            shutil.rmtree(t)
        shutil.copytree(src / "tasks" / tid, t, symlinks=False)
        p = t / "task.toml"
        s = p.read_text()
        s2 = re.sub(r"(\[agent\]\s*\ntimeout_sec\s*=\s*)[0-9.]+", rf"\g<1>{timeout:.1f}", s)
        assert s2 != s, f"no [agent] timeout in {p}"
        p.write_text(s2)


copy(train, out_train, 7200)
copy(val, out_val, 3600)
summ = {
    "train_tasks": len(train), "train_families": train.family.nunique(), "train_sources": train.source.nunique(),
    "val_tasks": len(val), "val_by_subset": val.subset.value_counts().to_dict(),
    "dropped_tasks_from_heldout_papers": len(dropped),
    "val_q4b_solved_frac": round(float((val.qwen3_5_4b_reward >= 0.999).mean()), 3),
    "train_q4b_solved_frac": round(float((train.qwen3_5_4b_reward >= 0.999).mean()), 3),
    "train_by_subset": train.subset.value_counts().to_dict(),
}
(out_val.parent / f"{out_val.name}.split.json").write_text(json.dumps({**summ, "val_task_ids": sorted(val.task_id)}, indent=1))
print(json.dumps(summ, indent=1))
