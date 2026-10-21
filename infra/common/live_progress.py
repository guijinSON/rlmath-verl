"""Live rollout progress of an rlmath run -> W&B (project rlmath-compare, run id rollout-<source id>), from its logs only.

Every finished GRPO group is a `[RLMathRollouter][stream]` log line with its rewards and wall time; the nearest preceding
agent-log timestamp dates it. One W&B row per 8 finished groups (step = groups finished so far), with rolling stats
over the last 64 groups. groups_done counts every finished group, including ones discarded by a restart. Appends only new rows, so it can run every few minutes while training goes on.

    python live_progress.py <source run id> <run name> <prompts per step> <log> [<log> ...]   (logs in time order)
"""
import re
import sys
from collections import deque
from datetime import datetime, timezone

import wandb

src, name, per_step = sys.argv[1], sys.argv[2], int(sys.argv[3])
logs = sys.argv[4:]
TS = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+:")
STREAM = re.compile(r"\[RLMathRollouter\]\[stream\].*? wall=(\d+)s rewards=\[([^\]]*)\] versions=\[(\d+),(\d+)\]")

groups = []  # (unix time, rewards, wall s)
versions = []  # (min, max) policy version that generated each group
last_t = None
for path in logs:
    try:
        f = open(path, errors="ignore")
    except OSError:
        continue
    for line in f:
        m = TS.search(line)
        if m:
            last_t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
        s = STREAM.search(line)
        if s and last_t:
            r = [float(x) for x in s.group(2).split(",") if x.strip()]
            if r:
                groups.append((last_t, r, int(s.group(1))))
                versions.append((int(s.group(3)), int(s.group(4))))
if not groups:
    print("no finished groups yet"); sys.exit()

run = wandb.init(project="rlmath-compare", entity="math-problem-gen", id=f"rollout-{src}", name=f"{name} (live rollout progress)",
                 resume="allow", tags=["live-progress"], config={"source_id": src, "prompts_per_step": per_step},
                 settings=wandb.Settings(silent=True))
wandb.define_metric("live/groups_done")
wandb.define_metric("live/*", step_metric="live/groups_done")
done = int(run.summary.get("live/groups_done", 0) or 0)
t0 = groups[0][0]
win = deque(maxlen=64)
new = 0
for i, (t, r, wall) in enumerate(groups, 1):
    win.append((t, r, wall))
    if i % 8 or i <= done:
        continue
    means = [sum(x) / len(x) for _, x, _ in win]
    zv = [max(x) - min(x) < 1e-6 for _, x, _ in win]
    walls = sorted(w for _, _, w in win)
    span_h = max((win[-1][0] - win[0][0]) / 3600, 1e-3)
    wandb.log({
        "live/groups_done": i,
        "live/hours": (t - t0) / 3600,
        "live/mean_reward_last64": sum(means) / len(means),
        "live/zero_var_frac_last64": sum(zv) / len(zv),
        "live/all_wrong_frac_last64": sum(max(x) <= 0 for _, x, _ in win) / len(win),
        "live/all_correct_frac_last64": sum(min(x) >= 0.999 for _, x, _ in win) / len(win),
        "live/group_wall_min_p50_last64": walls[len(walls) // 2] / 60,
        "live/groups_per_hour_last64": (len(win) - 1) / span_h,
    }, step=i)
    new += 1
run.finish()
print(f"{name}: {len(groups)} groups in logs, +{new} new rows")

# step-wise view: one row per policy version (= weights after k training steps), logged once the next version has
# started producing groups (so the version is complete). Groups spanning two versions are left out.
import math
byv = {}
for (vmin, vmax), (_, r, _) in zip(versions, groups):
    if vmin == vmax:
        byv.setdefault(vmin, []).append(r)
latest = max((v for v, _ in versions), default=0)
vrun = wandb.init(project="rlmath-compare", entity="math-problem-gen", id=f"rollout-v-{src}", name=f"{name} (rollout reward by policy version)",
                  resume="allow", tags=["live-progress"], config={"source_id": src}, settings=wandb.Settings(silent=True))
vdone = vrun.summary.get("version/policy_version")
vdone = -1 if vdone is None else int(vdone)
for v in sorted(byv):
    if v <= vdone or v >= latest:
        continue
    means = [sum(x) / len(x) for x in byv[v]]; n = len(means); mu = sum(means) / n
    se = (sum((m - mu) ** 2 for m in means) / max(n - 1, 1)) ** 0.5 / math.sqrt(n)
    wandb.log({"version/policy_version": v, "version/mean_reward": mu, "version/mean_reward_se": se, "version/groups": n,
               "version/zero_var_frac": sum(max(x) - min(x) < 1e-6 for x in byv[v]) / n,
               "version/all_wrong_frac": sum(max(x) <= 0 for x in byv[v]) / n}, step=v)
vrun.finish()
