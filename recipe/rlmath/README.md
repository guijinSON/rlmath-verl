# rlmath: fully-async agentic GRPO on research-math Harbor tasks

The policy (Qwen3.5-4B/9B; earlier Qwen3-4B, Qwen3-30B-A3B) solves math tasks inside Harbor sandboxes through a
Terminus-2 agent: it thinks, runs Python/sympy/solvers in a tmux shell, and writes an answer file that the task's
verifier scores. Training is verl's fully-async policy (streaming rollouts plus partial rollout across weight syncs).
The agent loop, LLM proxy and Harbor plumbing come from the vendored `recipe/agentic`.

```
           ┌───────────── GPU node (4-8 H200/GH200) ────────────────┐       ┌── sandbox hosts ──┐
 parquet → │ Rollouter ─▶ vLLM servers ◀── LLM proxy ◀── Terminus-2 ◀┼──────▶│ Harbor trials:     │
 (Harbor   │    │            (rollout GPUs)                  agents   │ docker│ task image + egress│
  tasks)   │    ▼ trajectories (queue, staleness ≤ 1)                 │  /    │ sidecar per trial  │
           │ Trainer (FSDP or Megatron, trainer GPUs) ─ NCCL weight sync┼ apptainer          │
           └────────────────────────────────────────────────────────────┘       └───────────────────┘
```

## Files

| File | Role |
|---|---|
| `async_main.py` | Hydra entrypoint (`python3 -m recipe.rlmath.async_main`): preflight checks, Ray init, task runner |
| `async_components.py` | Ray actors `RLMathAsyncTaskRunner` / `RLMathRollouter` / `RLMathTrainer` (subclass verl fully-async actors) |
| `async_proxy.py` | LLM proxy whose vLLM provider resumes generations aborted by a weight sync and tracks per-turn weight versions |
| `async_agent_loop.py` | Agent loop: weight-version stamping, logprob alignment, infra-failure detection, circuit breaker on sandbox outages |
| `rl_terminus.py` | Terminus-2 adapted to the remote agent loop (no summarization, sampling params passed through) |
| `tito.py` | Token-in/token-out consistency: trains on exactly the tokens the rollout produced |
| `remote_docker.py` | Harbor Docker environment for remote daemons, with weighted multi-VM routing (`RLMATH_DOCKER_HOSTS_FILE`) |
| `limits.py` | Live-tunable concurrency (`RLMATH_LIMITS_FILE`, re-read every 30 s) |
| `overlong.py` | Overlong filtering/penalty and infra-failure classification (excluded from GRPO groups) |
| `harbor_data.py` | Harbor task dirs → parquet, content-keyed cache, per-task-type validation `data_source` |
| `tool_metrics.py` | W&B diagnostics: tool/library use per trajectory, GRPO group saturation |
| `patch_q35_packing.py`, `check_q35_packing.py` | Qwen3.5 Gated-DeltaNet packed-sequence fix and its numerical check |
| `split_rm2.py`, `make_val.py` | Train/val splits held out by paper |
| `config/` | `rlmath_async.yaml` (FSDP), `rlmath_async_megatron.yaml` (Megatron), agent-loop registry |
| `run_rlmath_async.sh` | Launch script; every knob is an env var (see its header). `DRY=1` / `SMOKE=1` for checks |
| `ASYNC_NOTES.md` | Design notes: how verl's fully-async mode works and every decision made on top of it |

## Running

```bash
cd <this checkout>
export W=$HOME/rlmath-rl            # workspace: venv, data, logs, checkpoints
export MODEL_PATH=Qwen/Qwen3.5-9B
export TRAIN_DIR=$W/data/train_tasks VAL_DIR=$W/data/val_tasks
export RLMATH_DOCKER_HOSTS_FILE=$W/docker_hosts      # see infra/sandbox-vm/
SMOKE=1 bash recipe/rlmath/run_rlmath_async.sh       # 3 tiny steps
bash recipe/rlmath/run_rlmath_async.sh               # full run
```

Cluster-specific launchers, watchdogs, sandbox tunnels and Slurm jobs live in [`infra/`](../../infra/README.md).
