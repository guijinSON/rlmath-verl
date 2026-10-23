"""Backfill W&B steps lost when a Slurm job is killed at its time limit (offline .wandb ends in a truncated record).
verl prints every logged dict to stdout as `step:N - key:value - key:value ...` (several lines per step); parse the
job's Slurm log, merge per step, and log only steps newer than the run's last _step (W&B steps must increase).
usage: wandb_backfill_from_log.py <run_id> <slurm.out> [--dry]"""
import os, re, sys, wandb
ENT, PROJ = os.environ.get("WANDB_ENTITY", "math-problem-gen"), os.environ.get("WANDB_PROJECT", "rlmath")
rid, log = sys.argv[1], sys.argv[2]; dry = "--dry" in sys.argv
ansi = re.compile(r"\x1b\[[0-9;]*m")
num = re.compile(r"^(?:np\.\w+\()?(-?[0-9.]+(?:e[-+]?[0-9]+)?|nan|inf)\)?$")
steps = {}
for line in open(log, errors="replace"):
    line = ansi.sub("", line)
    i = line.find("step:")
    if i < 0 or " - " not in line[i:]: continue
    parts = line[i:].strip().split(" - ")
    m = re.fullmatch(r"step:(\d+)", parts[0].strip())
    if not m: continue
    d = steps.setdefault(int(m.group(1)), {})
    for p in parts[1:]:
        if ":" not in p: continue
        k, v = p.rsplit(":", 1)
        mv = num.match(v.strip())
        if mv: d[k.strip()] = float(mv.group(1))
NAME = sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else None
GROUP = sys.argv[sys.argv.index("--group") + 1] if "--group" in sys.argv else None
api = wandb.Api()
try:
    r = api.run(f"{ENT}/{PROJ}/{rid}"); last = r.summary.get("_step"); last = -1 if last is None else int(last); exists = True
except Exception:   # offline writer died before anything reached W&B (e.g. EDQUOT at wandb.init) -> create it
    last, exists = -1, False
todo = sorted(s for s in steps if s > last and steps[s])
print(f"{rid}: W&B last step {last}; log has steps {sorted(steps)[:3]}..{sorted(steps)[-3:] if steps else []}; to backfill {todo}")
for s in todo: print(f"  step {s}: {len(steps[s])} keys;", {k: round(v, 4) for k, v in steps[s].items() if k in ("critic/score/mean", "val-core/harbor/reward/mean@1", "rlmath/mismatch_kept_k3", "response_length/mean")})
if dry or not todo: sys.exit(0)
kw = {} if exists else {"name": NAME, "group": GROUP}
run = wandb.init(entity=ENT, project=PROJ, id=rid, resume="must" if exists else "allow", settings=wandb.Settings(silent=True), **kw)
for s in todo: run.log(steps[s], step=s)
run.finish(); print("backfilled", todo)
