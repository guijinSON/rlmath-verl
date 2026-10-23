# Fixed recipes

A recipe file holds **only the algorithm**: batch geometry, lengths, objective, validation and checkpoint cadence,
backend and optimizer. Everything about the machine (GPU layout, vLLM memory share, sandbox backend, concurrency,
paths, keys) lives in a site profile, `infra/sites/<site>.env`. A run is recipe + site + data:

```bash
# single GPU machine (Kubernetes pod / SSH box)
SITE=infra/sites/<site>.env RECIPE=recipe/rlmath/recipes/q35_9b_64x16_lr2e-6.env \
EXP=rlmath_q35_9b_<data>_<site>_v1 TRAIN_DIR=<tasks> VAL_DIR=<val tasks> bash infra/run_single_node.sh

# Slurm GH200 cluster (multi-node)
SITE=infra/sites/<site>.env RECIPE=recipe/rlmath/recipes/q35_9b_64x16_lr2e-6.env \
EXP=rlmath_q35_9b_<data>_<site>_v1 TRAIN_DIR=<tasks> VAL_DIR=<val tasks> \
  sbatch -A <account> -p <partition> -N 8 -J <name> -o <log dir>/%x-%j.out infra/slurm/rlmath.sbatch
```

| recipe | what |
|---|---|
| `q35_9b_64x16.env` | the shared core (sourced by the two below) |
| `q35_9b_64x16_lr2e-6.env` | lr 2e-6: every Qwen3.5-9B run so far |
| `q35_9b_64x16_lr1e-6.env` | lr 1e-6 (MLXP co-hard lr1e6 test) |

**Core (fixed):**

| setting | value |
|---|---|
| model | Qwen3.5-9B base |
| batch | 64 prompts × 16 rollouts per step = 2 updates × 512 trajectories (TRAIN_BS 32 × REQUIRE_BATCHES 2) |
| steps | 160 |
| episode lengths | 12 turns, 8k prompt, 131k trajectory budget, 32k tokens per call |
| objective | GRPO (no KL, clip 0.2/0.28, entropy 0); token-level rollout IS (threshold 2.0) |
| trajectory filtering | truncated trajectories filtered; infra-failed trials dropped |
| prompting | token-in/token-out proxy |
| micro-batches | 1 sequence (Qwen3.5 GatedDeltaNet must not see packed neighbours) |
| validation | T=1.0, top-p 0.95, 1 attempt, no stop-after-truncation; every 4 steps, with a checkpoint |
| memory | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

**Options:** set before the command; every value lands in the W&B config through the Hydra overrides.

| option | default | values |
|---|---|---|
| `STALENESS` | 1.0 | async staleness threshold; 2.0 is untested in isolation |
| `BACKEND` | fsdp | `fsdp` / `megatron` (verl Megatron engine, TP=`MCORE_TP` + sequence parallel; needs the mcore venv) |
| `FSDP_OPTIM` | torchao_bf16_sr | `torchao_bf16_sr` (bf16 params, torchao AdamW with bf16 stochastic rounding) / `torch_fp32` |
| `MCORE_OPTIM` | fp32 | `fp32` (distributed Adam, fp32 main params + moments) / `bf16` (precision-aware TE FusedAdam: bf16 moments, fp32-exact master via remainders) |
| `MCORE_TP` | 4 | |
| `TOTAL_STEPS`, `VAL_BEFORE` | 160, true | |
| `LR` | per recipe file | |

FSDP and Megatron cannot be made numerically identical: mcore has no bf16 stochastic-rounding optimizer. In a
10-step on-policy test the two engines' curves matched; one step at the same lr moved log-probs ~2x more on Megatron.

`PRINT_ARGS=1 ... bash infra/run_single_node.sh` prints the resolved Hydra overrides and stops (no GPUs, venv or
sandbox side effects), which is how recipe changes are diffed against past runs.
