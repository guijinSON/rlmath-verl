"""Mirror EVERY run of W&B project `rlmath` into `rlmath-compare` with compute-normalized x-axes.

Step k (k optimizer steps done) maps to
  x/trajectories = k * prompts_per_step * n   (rollouts x batch size: the fair "data seen" axis, default x)
  x/prompts      = k * prompts_per_step
  x/updates      = k * updates_per_step
  x/train_step   = k
Train metrics of step k = rows with training/global_step == k; validation after k steps = row at _step == k
(the fully-async trainer logs val one row before that step's train metrics). Mirrors are cmp-<source id>; a re-run
only appends steps that are new, so live runs can be refreshed. New source runs are picked up automatically.
"""
import math, re, wandb
ENT = 'math-problem-gen'; SRC = 'rlmath'; DST = 'rlmath-compare'
CFG_DONOR = {'j47ohesj': 'owbedbx0'}                       # JUPITER -ALL merged run has no config -> chain c1
MODEL_OVERRIDE = {'tq8p8fb1': 'Qwen3.5-4B', 'imt3jo1g': 'Qwen3-4B-Thinking', '8z7wyey1': 'Qwen3-4B-Thinking',
                  'j47ohesj': 'Qwen3.5-4B'}               # runs started from merged ckpts
V2_SPLIT = {'flqgwse1', 'm0swnzz3', '2n6epbpv'}            # data/tasks_train_v2 + val_heldout_v2 (Oct 2+)
RENAME = {'xm5o7eoc': 'rlmath_30b_mlxp_v1-resume'}         # 2-step continuation of zoftbk4v, same name
# metrics that never reached W&B, parsed from the run log by bin/log_backfill.py ({k: {metric: value}})
BACKFILL = {'uvd2fzc3': 'uvd2fzc3_backfill.json'}
# verl fully-async resume labels the first new step global_step+2 (ckpt 40 -> logged 42): labels >= S are k = label-1
RESUME_SKIP = {'uvd2fzc3': 42}
import json, os
api = wandb.Api(timeout=120)

def dig(d, path):
    for p in path.split('.'):
        if not isinstance(d, dict) or p not in d: return None
        d = d[p]
    return d

def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

def model_of(rid, cfg):
    if rid in MODEL_OVERRIDE: return MODEL_OVERRIDE[rid]
    p = dig(cfg, 'actor_rollout_ref.model.path') or ''
    m = re.search(r'(Qwen3[.\d]*-[\w.]+?)(?:/|$)', p.replace('models--Qwen--', 'Qwen/').replace('/snapshots', '/'))
    name = m.group(1) if m else p.rsplit('/', 1)[-1]
    return name.replace('-2507', '').replace('-A3B-Thinking', '-A3B-Thinking')

def dataset_of(rid, name):
    if 'rm2' in name: return 'researchmath2'
    if rid in V2_SPLIT: return 'rlmath-v2'
    return 'rlmath-v1'

skipped = []
for src in api.runs(f'{ENT}/{SRC}', order='created_at'):
    rid = src.id
    cfg = api.run(f'{ENT}/{SRC}/{CFG_DONOR[rid]}').config if rid in CFG_DONOR else src.config
    n = dig(cfg, 'actor_rollout_ref.rollout.n'); mb = dig(cfg, 'actor_rollout_ref.actor.ppo_mini_batch_size')
    if not n or not mb: skipped.append((src.name, 'no batch config')); continue
    rb = dig(cfg, 'async_training.require_batches') or 1; trig = dig(cfg, 'async_training.trigger_parameter_sync_step') or 1
    ppe = dig(cfg, 'actor_rollout_ref.actor.ppo_epochs') or 1
    prompts_per_step = mb * rb * trig; traj_per_step = prompts_per_step * n; updates_per_step = rb * trig * ppe
    K = {}
    for row in src.scan_history(page_size=1000):
        gs = row.get('training/global_step')
        for key, v in row.items():
            if key.startswith('_') or not num(v): continue
            if key.startswith('val-'): k = row['_step']
            elif gs is not None: k = int(gs)
            else: continue
            if rid in RESUME_SKIP and k >= RESUME_SKIP[rid]: k -= 1
            K.setdefault(k, {})[key] = v
    if rid in BACKFILL and os.path.exists(BACKFILL[rid]):
        for k, m in json.load(open(BACKFILL[rid])).items():
            for key, v in m.items(): K.setdefault(int(k), {}).setdefault(key, v)
    train_ks = [k for k, m in K.items() if 'critic/score/mean' in m]
    if not train_ks: skipped.append((src.name, 'no training steps logged')); continue
    last_train = max(train_ks)
    model = model_of(rid, cfg); dataset = dataset_of(rid, src.name); name = RENAME.get(rid, src.name)
    run = wandb.init(entity=ENT, project=DST, id=f'cmp-{rid}', name=name, resume='allow', group=model,
                     tags=[model, dataset, src.state],
                     config=dict(source_run=src.url, source_id=rid, source_state=src.state, model=model, dataset=dataset,
                                 rollout_n=n, prompts_per_step=prompts_per_step, trajectories_per_step=traj_per_step,
                                 updates_per_step=updates_per_step, lr=dig(cfg, 'actor_rollout_ref.actor.optim.lr')),
                     settings=wandb.Settings(silent=True))
    for ax in ['x/trajectories', 'x/prompts', 'x/updates', 'x/train_step']:
        wandb.define_metric(ax)
    wandb.define_metric('*', step_metric='x/trajectories')
    done = run.summary.get('x/train_step'); done = -1 if done is None else int(done)
    new = 0
    for k in sorted(K):
        if k <= done or k > last_train: continue
        m = dict(K[k]); m.update({'x/trajectories': k * traj_per_step, 'x/prompts': k * prompts_per_step,
                                  'x/updates': k * updates_per_step, 'x/train_step': k})
        wandb.log(m, step=k); new += 1
    run.finish()
    print(f'{rid} {name:36s} {model:22s} {dataset:13s} {src.state:8s} traj/step={traj_per_step:5d} '
          f'steps {min(train_ks)}..{last_train} (+{new} new)', flush=True)
for s in skipped: print('SKIPPED', *s)
