#!/bin/bash
# rlmath_q35_9b_rm2_v4_64x16 (10-16): 64 prompts x 16 rollouts = 1,024 traj/step, 2 updates x 512 (TRAIN_BS 32 x REQUIRE_BATCHES 2),
# lr 2e-6 (each update = v1 recipe), 160 steps (= v1 trajectory budget), val+ckpt every 4 steps. Pair: KT v4_32x32.
# rlmath_q35_9b_rm2_v2 (10-14): bigger batch + higher lr vs v1. 256 prompts x 16 rollouts per step = TRAIN_BS 64 (ppo mini-batch, 1024 traj/update) x
# REQUIRE_BATCHES 4 -> 4 optimizer updates/step; lr 3e-6; 40 steps = 10,240 prompts (same data budget as v1 at 80 steps, ~4 passes);
# val + ckpt every 2 steps (= every 512 prompts, same spacing in data as v1), no val before train (base val 0.369 known).
# 10-14 23:40 restart: rollout concurrency 16 -> 20 prompts (320 trials; trials/worker 32 -> 40), vLLM stats logging on (disable_log_stats=false).
# Qwen3.5-4B agentic GRPO on researchmath2-datav0.1-imagev0.3, MLXP node1 (8 GPUs: 6 vLLM + 2 FSDP trainers).
# rlmath_q35_9b_rm2_v1: base Qwen3.5-9B. 128 prompts x 16 rollouts per step = TRAIN_BS 32 (ppo mini-batch) x REQUIRE_BATCHES 4
# -> 4 optimizer updates/step; lr 2e-6; NO length penalty: truncated trajectories dropped from the loss (filter, with
# zero-advantage guard token so no mini-batch is empty); val at T=1.0, val trials not stopped after a truncated call;
# 4 vLLM + 4 trainers, no packing, entropy 0, infra exclusion + breaker.
#   SMOKE=1 bash launch_q35rm2.sh   -> 3 tiny steps (no val/ckpt), log logs/<EXP>_smoke.log
set -u
command -v g++ > /dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc g++ libc6-dev > /dev/null 2>&1)   # 10-17: fresh pods lack compilers; Triton JIT needs gcc, flashinfer GDN JIT (nvcc host) needs g++
export FLASHINFER_WORKSPACE_BASE=$W/.fi_cache   # JIT cache on nvme (survives pod changes)
[ -s /root/.wandb_key ] && export WANDB_API_KEY="$(cat /root/.wandb_key)"   # key file 0600, per pod
W=${W:-$HOME/rlmath-rl}
EXP=${EXP:-rlmath_q35_9b_rm2_v4_64x16}
LOG=$W/logs/${EXP}$([ "${SMOKE:-0}" = 1 ] && echo _smoke).log
echo "drivers alive: $(pgrep -fc '^python3 -m recipe.rlmath.async_main')"
export DOCKER_HOST=unix:///tmp/sbx-docker.sock DOCKER_API_VERSION=1.49 DOCKER_CONFIG=$W/.docker   # compose plugin lives on nvme (10-17: fresh pods have no /root/.docker)
$W/bin/docker compose version > /dev/null 2>&1 || { echo "ABORT: docker compose plugin missing (every trial would fail)"; exit 1; }
export RLMATH_LIMITS_FILE=${RLMATH_LIMITS_FILE:-$W/rlmath_limits}   # live concurrency (limits.py)
export RLMATH_DOCKER_HOSTS_FILE=${RLMATH_DOCKER_HOSTS_FILE:-$W/docker_hosts}   # multi-VM sandbox routing (remote_docker.py)
for h in $(grep -v "^#" $RLMATH_DOCKER_HOSTS_FILE 2>/dev/null | awk '$2 > 0 {print $1}'); do $W/bin/docker -H $h ps -aq 2>/dev/null | xargs -r $W/bin/docker -H $h rm -f >/dev/null 2>&1; done   # only hosts this run owns (weight > 0)
cd $W/verl && rm -rf trials
export VENV=$W/venv-q36 RLMATH_TEXT_ONLY=1 FILTER_OVERLONG_PROMPTS=false VLLM_USE_FLASHINFER_MOE_FP16=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
# vLLM 0.20: the bf16 model needs no DeepGEMM (engine init aborts on 'DeepGEMM backend is not available');
# flashinfer JITs the GDN prefill kernel and needs nvcc -> the nvidia-cuda-nvcc 13.0 wheel in the venv (no system CUDA here)
export VLLM_USE_DEEP_GEMM=0 CUDA_HOME=$W/venv-q36/lib/python3.12/site-packages/nvidia/cu13
export EXP MODEL=${MODEL:-Qwen/Qwen3.5-9B}
NG=$(nvidia-smi -L | grep -c "^GPU")   # 10-17: reservations may give 6 or 8 GPUs; trainers stay 4 (ckpt world size 4)
export HF_HOME_OVERRIDE=$W/hf GPU_LIST=$(seq -s, 0 $((NG-1))) ROLLOUT_GPUS=$((NG-4))
export TRAIN_DIR=$W/data/rm2_train VAL_DIR=$W/data/rm2_val_heldout
export TRAIN_BS=32 REQUIRE_BATCHES=2 ROLLOUT_N=16 LR=2e-6 VAL_TEMP=1.0 VAL_TOP_P=0.95 MAX_TURNS=12 MAX_PROMPT=8192 MAX_RESPONSE=131072 TURN_TOKENS=32768 TOTAL_STEPS=${TOTAL_STEPS:-160}
export DYNAMIC_BSZ=false MICRO_BS=1 OPTIM_OFFLOAD=false PARAM_OFFLOAD=false
export STALENESS=1.0 MAX_CONCURRENT_PROMPTS=${MAX_CONCURRENT_PROMPTS:-null} RLMATH_TRIALS_PER_WORKER=${RLMATH_TRIALS_PER_WORKER:-56} RLMATH_ENV_LIFECYCLE_PER_WORKER=3
export SAVE_FREQ=4 TEST_FREQ=4 VAL_BEFORE=false ENTROPY_COEFF=0 OVERLONG_MODE=filter DROP_INFRA_FAILED=true RLMATH_TITO=1
if [ "${SMOKE:-0}" = 1 ]; then export SMOKE=1 SMOKE_MAX_RESPONSE=16384 TURN_TOKENS=8192; fi
setsid nohup bash recipe/rlmath/run_rlmath_async.sh \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 actor_rollout_ref.actor.optim.optimizer_impl=torchao.optim \
  actor_rollout_ref.actor.optim.optimizer=_AdamW 'actor_rollout_ref.actor.optim.override_optimizer_config={bf16_stochastic_round:true}' \
  algorithm.rollout_correction.rollout_is=token algorithm.rollout_correction.rollout_is_threshold=2.0 \
  actor_rollout_ref.rollout.disable_log_stats=false \
  > $LOG 2>&1 < /dev/null &
echo "launched -> $LOG"
