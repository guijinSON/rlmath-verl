#!/bin/bash
# FULLY-ASYNC agentic GRPO on rlmath Harbor tasks: Qwen3-4B-Thinking-2507, verl@9f18abe5
# fully_async_policy + Harbor terminus via the recipe proxy, node1 GPUs 0-3.
# Rollout (standalone vLLM) and training (FSDP) run on disjoint GPUs; trajectories stream
# into a queue; weights are NCCL-synced every SYNC_EVERY steps while in-flight agent turns
# are paused and resumed (partial rollout). See recipe/rlmath/ASYNC_NOTES.md.
#
#   SMOKE=1 bash recipe/rlmath/run_rlmath_async.sh        # 3 tiny steps, no ckpt/val
#   DRY=1 bash recipe/rlmath/run_rlmath_async.sh          # imports + composed config only
#   bash recipe/rlmath/run_rlmath_async.sh                # full run
set -euo pipefail
W=${W:-$HOME/rlmath-rl}                                    # workspace root (see infra/README.md)
VENV=${VENV:-$W/venv}                                         # python env (KT Qwen3.6 runs: $W/venv-q36)
: "${GPU_LIST:=0,1,2,3}"                                     # physical GPUs for this job (MLXP: 0-3; KT: 1-7, GPU 0 is dead)
NGPUS=$(echo "$GPU_LIST" | tr ',' '\n' | grep -c .)
EXP=${EXP:-rlmath_q4bthink_grpo_async_v1}     # different from the sync EXP: resume_mode=auto must not pick up sync ckpts
: "${TRAIN_BS:=16}" "${ROLLOUT_N:=8}" "${MAX_TURNS:=12}" "${TOTAL_STEPS:=300}"
: "${MAX_PROMPT:=8192}" "${MAX_RESPONSE:=32768}" "${TURN_TOKENS:=8192}" "${VAL_DIR:=$W/data/val_tasks}"
# FILTER_OVERLONG_PROMPTS=false: verl's prompt-length filter only templates the placeholder prompt (the task's
# instance_id; the real prompt is built by the agent), and Qwen3.5/3.6's chat template raises "No user query found"
# on it. Real prompt length is already bounded by tasks_train (instruction <= 6000 tokens).
# DYNAMIC_BSZ=false MICRO_BS=1: one sequence per micro-batch. Required for Qwen3.5/3.6 (qwen3_5_moe): transformers'
# Gated-DeltaNet ignores packed-sequence boundaries (conv seq_idx=None, no cu_seqlens), so dynamic-bsz packing would
# leak recurrent state across trajectories (verl's own run_qwen3_5_35b_fsdp.sh uses the same setting).
# MAX_TOKEN_LEN_PER_GPU (default MAX_PROMPT+MAX_RESPONSE): dynamic-bsz token budget per GPU; with Ulysses
# sequence parallel (actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=k) verl packs
# k * this per micro-batch, so set it to (MAX_PROMPT+MAX_RESPONSE)/k
# ---- async knobs
: "${ROLLOUT_GPUS:=3}"                           # vLLM replicas (TP=1 each); trainer gets the rest of the 4
TRAIN_GPUS=$((NGPUS - ROLLOUT_GPUS))
: "${STALENESS:=0.5}"                            # up to (1+STALENESS)*TRAIN_BS prompts started per weight version
: "${SYNC_EVERY:=1}"                             # trigger_parameter_sync_step
: "${REQUIRE_BATCHES:=1}"
: "${PARTIAL:=true}"                             # resume in-flight turns across weight syncs
: "${ROLLOUT_MEM:=0.85}"                         # standalone vLLM owns its GPUs (sync run shared them at 0.6)
: "${MAX_CONCURRENT_PROMPTS:=null}"              # e.g. 16 => at most 16*ROLLOUT_N concurrent Harbor trials
: "${SAVE_FREQ:=25}" "${TEST_FREQ:=20}" "${VAL_BEFORE:=true}" "${TRAIN_LIMIT:=null}"
# ---- model / objective
: "${MODEL:=Qwen/Qwen3-4B-Thinking-2507}"          # HF id or local HF dir (e.g. $W/ckpt/merged/async_v1_step25);
                                                  # feeds actor, vLLM rollout, trainer+proxy tokenizers (all read model.path)
: "${ENTROPY_COEFF:=0.001}"                       # actor_rollout_ref.actor.entropy_coeff (0 = no entropy bonus)
: "${OVERLONG_FILTER:=true}"                      # drop truncated trajectories from the policy loss (overlong.py)
case "$OVERLONG_FILTER" in true|false) ;; *) echo "OVERLONG_FILTER must be true|false"; exit 1;; esac
# OVERLONG_MODE overrides OVERLONG_FILTER: filter (drop truncated trajectories from the loss) | penalize (reward 0,
# kept in the loss, before the GRPO advantage) | off. Unset = filter/off from OVERLONG_FILTER (old behaviour).
: "${OVERLONG_MODE:=$([ "$OVERLONG_FILTER" = true ] && echo filter || echo off)}"
case "$OVERLONG_MODE" in filter|penalize|off) ;; *) echo "OVERLONG_MODE must be filter|penalize|off"; exit 1;; esac
: "${DROP_INFRA_FAILED:=true}"                    # leave sandbox/infra-failed trials out of their GRPO group

if [ "${SMOKE:-0}" = 1 ]; then
  EXP=${EXP}_smoke
  TRAIN_BS=2 ROLLOUT_N=2 MAX_TURNS=4 TOTAL_STEPS=3 MAX_RESPONSE=${SMOKE_MAX_RESPONSE:-16384}
  SAVE_FREQ=-1 TEST_FREQ=-1 VAL_BEFORE=false TRAIN_LIMIT=${TRAIN_LIMIT/null/16}
fi
# trainer.resume_mode=auto loads $W/ckpt/$EXP/global_step_* OVER the MODEL weights. When starting from a
# merged checkpoint (MODEL=<dir>), use a fresh EXP, or set ALLOW_RESUME=1 to resume that EXP on purpose.
if [ "$MODEL" != Qwen/Qwen3-4B-Thinking-2507 ] && [ -f "$W/ckpt/$EXP/latest_checkpointed_iteration.txt" ] && [ "${ALLOW_RESUME:-0}" != 1 ]; then
  echo "refusing: MODEL=$MODEL but $W/ckpt/$EXP already has checkpoints (step $(cat $W/ckpt/$EXP/latest_checkpointed_iteration.txt)) that would be auto-resumed; set a new EXP or ALLOW_RESUME=1"; exit 1
fi
# the rollouter feeds exactly this many prompts; trainer steps = this / (TRAIN_BS*REQUIRE_BATCHES)
TOTAL_ROLLOUT_PROMPTS=$((TOTAL_STEPS * TRAIN_BS * REQUIRE_BATCHES * SYNC_EVERY))
# attention: fa2 = flash-attn varlen + remove-padding + fused log-prob kernels (needs the flash_attn wheel);
# sdpa = no remove-padding and NO fused kernels (verl@9f18abe5's fused no-rmpad path trips the
# no_padding_2_padding assertion). auto picks fa2 when flash_attn imports.
: "${ATTN:=auto}"
if [ "$ATTN" = auto ]; then $VENV/bin/python -c "import flash_attn" 2>/dev/null && ATTN=fa2 || ATTN=sdpa; fi
if [ "$ATTN" = fa2 ]; then ATTN_IMPL=flash_attention_2 RMPAD=true FUSED=true; else ATTN_IMPL=sdpa RMPAD=false FUSED=false; fi
# fused log-prob kernels are incompatible with FSDP2 (lm_head weight stays a DTensor): FUSED_KERNELS=false there
FUSED=${FUSED_KERNELS:-$FUSED}
echo "[rlmath-async] attention=$ATTN (impl=$ATTN_IMPL remove_padding=$RMPAD fused_kernels=$FUSED) max_response=$MAX_RESPONSE"
echo "[rlmath-async] model=$MODEL entropy_coeff=$ENTROPY_COEFF overlong_mode=$OVERLONG_MODE drop_infra_failed=$DROP_INFRA_FAILED"
ARGS=(
  actor_rollout_ref.model.path=$MODEL
  +actor_rollout_ref.model.override_config.attn_implementation=$ATTN_IMPL
  actor_rollout_ref.model.use_remove_padding=$RMPAD
  actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.model.use_fused_kernels=$FUSED
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=true
  actor_rollout_ref.actor.entropy_checkpointing=true
  data.train_harbor_dir=${TRAIN_DIR:-$W/data/tasks_train}
  data.val_harbor_dir=$VAL_DIR
  data.harbor_train_limit=$TRAIN_LIMIT
  data.return_raw_chat=true
  data.train_batch_size=0
  data.gen_batch_size=1
  data.max_prompt_length=$MAX_PROMPT
  data.max_response_length=$MAX_RESPONSE
  data.filter_overlong_prompts=${FILTER_OVERLONG_PROMPTS:-true}
  data.truncation=error
  data.prompt_key=instance_id
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=false
  algorithm.rollout_correction.bypass_mode=false
  actor_rollout_ref.hybrid_engine=false
  actor_rollout_ref.actor.use_kl_loss=false
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.clip_ratio_high=0.28
  actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF
  actor_rollout_ref.actor.optim.lr=${LR:-1e-6}
  actor_rollout_ref.actor.use_dynamic_bsz=${DYNAMIC_BSZ:-true}
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BS:-$TRAIN_BS}
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT + MAX_RESPONSE))}
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT + MAX_RESPONSE))}
  actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD:-false}
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIM_OFFLOAD:-false}
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_MEM
  actor_rollout_ref.rollout.n=$ROLLOUT_N
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.calculate_log_probs=true
  actor_rollout_ref.rollout.multi_turn.enable=true
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
  actor_rollout_ref.rollout.agent.num_workers=8
  actor_rollout_ref.rollout.agent.default_agent_loop=rlmath_remote_agent_async
  actor_rollout_ref.rollout.agent.agent_loop_config_path=recipe/rlmath/config/async-agent.yaml
  actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMP:-0.6}
  actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-0.95}
  actor_rollout_ref.rollout.val_kwargs.n=1
  trainer.logger='["console","wandb"]'
  trainer.project_name=rlmath
  trainer.experiment_name=$EXP
  trainer.n_gpus_per_node=$TRAIN_GPUS
  trainer.nnodes=1
  rollout.nnodes=1
  rollout.n_gpus_per_node=$ROLLOUT_GPUS
  rollout.total_rollout_steps=$TOTAL_ROLLOUT_PROMPTS
  async_training.staleness_threshold=$STALENESS
  async_training.trigger_parameter_sync_step=$SYNC_EVERY
  async_training.require_batches=$REQUIRE_BATCHES
  async_training.partial_rollout=$PARTIAL
  rlmath_async.max_concurrent_prompts=$MAX_CONCURRENT_PROMPTS
  rlmath_async.overlong_filter=$OVERLONG_FILTER
  rlmath_async.overlong_mode=$OVERLONG_MODE
  rlmath_async.drop_infra_failed=$DROP_INFRA_FAILED
  trainer.val_before_train=$VAL_BEFORE
  trainer.test_freq=$TEST_FREQ
  trainer.save_freq=$SAVE_FREQ
  trainer.total_epochs=100
  trainer.default_local_dir=$W/ckpt/$EXP
  remote_agent.agent_name=null
  remote_agent.agent_import_path=recipe.rlmath.rl_terminus:RLTerminus2
  remote_agent.model_name=openai/policy
  remote_agent.environment_import_path=recipe.rlmath.remote_docker:RemoteDockerEnvironment
)

if [ "${DYNAMIC_BSZ:-true}" = false ]; then
  ARGS+=(actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BS:-1}
         actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BS:-1}
         actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BS:-1})
fi

# DRY=1: no Ray, no Docker -- import our modules against the real verl/vLLM stack and print the
# composed Hydra config (hydra --cfg job exits before main()), then stop.
if [ "${DRY:-0}" = 1 ]; then
  source $VENV/bin/activate; export PYTHONPATH=$W/verl:${PYTHONPATH:-}; cd $W/verl
  python3 -c "import vllm, ray, torch; print(f'vllm={vllm.__version__} ray={ray.__version__} torch={torch.__version__}')"
  python3 -c "import recipe.rlmath.async_components, recipe.rlmath.async_agent_loop, recipe.rlmath.async_proxy as p; p.install_provider_patch(); print('[DRY] imports + provider patch OK')"
  python3 -m recipe.rlmath.async_main --cfg job "${ARGS[@]}" "$@" > /tmp/rlmath_async_cfg.yaml
  echo "[DRY] composed config -> /tmp/rlmath_async_cfg.yaml"; grep -nE "^async_training:|^rollout:|^rlmath_async:" -A6 /tmp/rlmath_async_cfg.yaml
  exit 0
fi

source $VENV/bin/activate
export PATH=$W/bin:$PATH
export CUDA_VISIBLE_DEVICES=$GPU_LIST            # MLXP: GPUs 4-7 belong to the running bigroll3 job
# Docker traffic goes through one persistent SSH unix-socket tunnel to the VM (infra/sandbox-vm/sbx-tunnel.sh);
# ssh:// DOCKER_HOST spawns an ssh per docker call and floods the VM's sshd at 100+ concurrent trials.
# Tunnels are started beforehand, one per sandbox VM: infra/sandbox-vm/sbx-tunnel.sh <alias> (see infra/README.md).
RLMATH_DEFAULT_SOCK=${RLMATH_DEFAULT_SOCK:-/tmp/sbx-docker.sock}   # 10-21: default sandbox socket overridable (VM5 down after vast outage)
for i in $(seq 1 30); do [ -S $RLMATH_DEFAULT_SOCK ] && break; sleep 1; done
export DOCKER_HOST=unix://$RLMATH_DEFAULT_SOCK
export DOCKER_API_VERSION=1.49                 # VM daemon (28.1.1) max API; harbor's platform probe does not negotiate
docker version --format "docker: client {{.Client.Version}} -> server {{.Server.Version}}" || { echo "docker tunnel down"; exit 1; }
export HF_HOME=${HF_HOME_OVERRIDE:-$W/hf_cache}
export PYTHONPATH=$W/verl:${PYTHONPATH:-}
export LLM_PROXY_IP=127.0.0.1                  # the agent runs in this pod
export REMOTE_AGENT_USE_LOCAL_TRIAL=true
export REMOTE_AGENT_IMPORT_PATH=recipe.rlmath.rl_terminus:RLTerminus2
export REMOTE_MODEL_NAME=openai/policy
# the recipe proxy defaults max_tokens to 2048 per call when the request omits it, which cuts
# Qwen3-Thinking off mid-reasoning (no </think>, no command) -> always send max_tokens.
export REMOTE_AGENT_KWARGS="{\"max_turns\": $MAX_TURNS, \"llm_call_kwargs\": {\"max_tokens\": $TURN_TOKENS}, \"model_info\": {\"max_input_tokens\": $((MAX_PROMPT + MAX_RESPONSE)), \"max_output_tokens\": $TURN_TOKENS, \"input_cost_per_token\": 0, \"output_cost_per_token\": 0}}"
# generation-time response budget + dead-session short-circuit (async_proxy.py): no tokens past the point
# where the trajectory is certain to be cut / filtered
export RLMATH_RESPONSE_BUDGET=${RLMATH_RESPONSE_BUDGET:-$MAX_RESPONSE}
# filter and penalize both end a session after its first truncated call: the trajectory is dropped (filter) or
# scored 0 (penalize) either way, so further tokens are wasted compute
export RLMATH_STOP_AFTER_TRUNC=${RLMATH_STOP_AFTER_TRUNC:-$([ "$OVERLONG_MODE" != off ] && echo 1 || echo 0)}
# validation trials are exempt from the stop-after-truncation (async_proxy._NO_STOP_TASKS); VAL_NO_STOP=0 restores it
[ "${VAL_NO_STOP:-1}" = 1 ] && export RLMATH_NO_STOP_TASKS_DIR=$VAL_DIR
# circuit breaker in async_agent_loop.py: after RLMATH_BREAKER_FAILS infra failures within RLMATH_BREAKER_WINDOW s
# (per agent-loop worker), hold new trials until `docker info` answers again
export RLMATH_BREAKER_FAILS=${RLMATH_BREAKER_FAILS:-6} RLMATH_BREAKER_WINDOW=${RLMATH_BREAKER_WINDOW:-120} RLMATH_BREAKER_POLL=${RLMATH_BREAKER_POLL:-30}
export REMOTE_AGENT_ENVIRONMENT_IMPORT_PATH=recipe.rlmath.remote_docker:RemoteDockerEnvironment
export REMOTE_AGENT_ENVIRONMENT_KWARGS='{}'
export REMOTE_AGENT_MAX_RETRIES=2
# last-resort task lookup (after local_task_path, <train_harbor_dir>/<id>, <val_harbor_dir>/<id>);
# the recipe default /home/verl/dataset-tasks/{instance_id} does not exist here.
TASK_TEMPLATE_DEFAULT="$W/data/tasks/{instance_id}"
export REMOTE_AGENT_TASK_PATH_TEMPLATE=${REMOTE_AGENT_TASK_PATH_TEMPLATE:-$TASK_TEMPLATE_DEFAULT}
export REMOTE_JOB_ID=$EXP
export RAY_TMPDIR=/tmp/ray_rlmath
unset PROXY_SERVER_URL                         # async variant only supports the in-cluster proxy actor
# own Ray cluster on non-default ports so we never attach to bigroll3's cluster
unset RAY_ADDRESS
# NEVER `ray stop` here: it kills every Ray process on the machine, including bigroll3.
# Clean up only a previous rlmath cluster (processes referencing our temp dir).
# (also clears a stale sync-run "llm_proxy_server" actor, which would point at the wrong load balancer)
pkill -f -- "--temp-dir=$RAY_TMPDIR" 2>/dev/null || true; pkill -f "$RAY_TMPDIR/session" 2>/dev/null || true; sleep 2
ray start --head --port 6391 --dashboard-port 8291 --temp-dir $RAY_TMPDIR --num-gpus $NGPUS ${RAY_NUM_CPUS:+--num-cpus $RAY_NUM_CPUS} \
  --ray-client-server-port 10091 --min-worker-port 30000 --max-worker-port 39999 --disable-usage-stats
export RAY_ADDRESS=127.0.0.1:6391
cd $W/verl
python3 -c "import vllm, ray, torch; print(f'[RLMATH ASYNC] vllm={vllm.__version__} ray={ray.__version__} torch={torch.__version__}')"
python3 -m recipe.rlmath.async_main "${ARGS[@]}" "$@"
