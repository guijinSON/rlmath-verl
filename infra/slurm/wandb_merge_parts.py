"""Stitch the per-job W&B runs of one chained training run (group G, names G-c0, G-c1, ...) into ONE continuous run.
For a step logged by several parts (a resumed job redoes steps after the checkpoint it loaded), the LATER part wins
per metric:
that is the lineage the training actually continued from. W&B history can't be rewritten, so the merged run is
rebuilt from scratch whenever any part gained steps (previous merged run deleted). State: <state>/merged.json.
usage: wandb_merge_parts.py <group> <state_dir> [--force]"""
import json, os, re, sys, wandb
G, state = sys.argv[1], sys.argv[2]; force = "--force" in sys.argv
ENT, PROJ = os.environ.get("WANDB_ENTITY", "math-problem-gen"), os.environ.get("WANDB_PROJECT", "rlmath")
api = wandb.Api(timeout=60)
parts = []
for r in api.runs(f"{ENT}/{PROJ}", filters={"group": G}):
    m = re.fullmatch(re.escape(G) + r"(?:-c(\d+))?", r.name or "")   # first job = bare name = part 0
    if m: parts.append((int(m.group(1) or 0), r))
parts.sort(key=lambda x: x[0])
if not parts: print(f"{G}: no parts yet"); sys.exit(0)
sig = [[n, r.id, r.summary.get("_step")] for n, r in parts]
sf = os.path.join(state, "merged.json"); old = json.load(open(sf)) if os.path.exists(sf) else {}
if not force and old.get("sig") == sig: print("merged run up to date"); sys.exit(0)
steps = {}
for n, r in parts:
    for row in r.scan_history(page_size=1000):
        s = row.get("_step")
        if s is None: continue
        d = {k: v for k, v in row.items() if not k.startswith("_") and isinstance(v, (int, float)) and v is not None}
        if not d: continue
        # per-metric overlay, later part wins: a resumed job first logs only rollouter metrics at its checkpoint
        # step, which must not wipe the earlier part's full row (score, val, ...) for that step
        cur = steps.setdefault(int(s), {"__owner": n})
        cur.update(d); cur["__owner"] = max(cur["__owner"], n)
if not steps: print(f"{G}: parts have no logged steps yet"); sys.exit(0)
last = old.get("last_step", -1)
if old.get("run_id") and not force:
    # append-only (keeps the merged run's URL stable): new steps beyond what the merged run already has
    todo = [k for k in sorted(steps) if k > last]
    if todo:
        run = wandb.init(entity=ENT, project=PROJ, id=old["run_id"], resume="must", settings=wandb.Settings(silent=True))
        for k in todo:
            d = dict(steps[k]); d["merge/part"] = d.pop("__owner"); run.log(d, step=k)
        run.finish()
    rid = old["run_id"]; print(f"merged run {rid}: appended steps {todo}")
else:
    if old.get("run_id"):
        try: api.run(f"{ENT}/{PROJ}/{old['run_id']}").delete()
        except Exception as e: print("old merged delete failed:", e)
    run = wandb.init(entity=ENT, project=PROJ, name=f"{G}-ALL", group=f"{G}-merged", tags=["merged"],
                     notes=f"continuous view stitched from the per-job parts {G}-cN; later parts win per metric on redone steps",
                     settings=wandb.Settings(silent=True))
    for k in sorted(steps):
        d = dict(steps[k]); d["merge/part"] = d.pop("__owner"); run.log(d, step=k)
    rid = run.id; run.finish(); print(f"merged run {rid} rebuilt: {len(parts)} parts, steps {min(steps)}..{max(steps)}")
json.dump({"sig": sig, "run_id": rid, "last_step": max(steps)}, open(sf, "w"))
