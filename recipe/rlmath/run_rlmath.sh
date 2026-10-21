#!/bin/bash
# Agentic GRPO on rlmath Harbor tasks: Qwen3-4B-Thinking-2507, verl + Harbor terminus, node1 GPUs 0-3.
# Sandboxes run on the vast.ai Docker VM (DOCKER_HOST=ssh://sbx), network_mode = no-network.
set -euo pipefail
W=${W:-$HOME/rlmath-rl}
EXP=${EXP:-rlmath_q4bthink_grpo_v1}
: "${TRAIN_BS:=16}" "${ROLLOUT_N:=8}" "${MAX_TURNS:=12}" "${TOTAL_STEPS:=300}"
: "${MAX_PROMPT:=8192}" "${MAX_RESPONSE:=32768}" "${TURN_TOKENS:=8192}" "${VAL_DIR:=$W/data/val_tasks}"
source $W/venv/bin/activate
export PATH=$W/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3            # GPUs 4-7 belong to the running bigroll3 job
# Docker traffic goes through one persistent SSH unix-socket tunnel to the VM (bin/sbx-tunnel.sh);
# ssh:// DOCKER_HOST spawns an ssh per docker call and floods the VM's sshd at 100+ concurrent trials.
pgrep -f sbx-tunnel.sh >/dev/null || { setsid nohup $W/bin/sbx-tunnel.sh >/dev/null 2>&1 < /dev/null & }
for i in $(seq 1 30); do [ -S /tmp/sbx-docker.sock ] && break; sleep 1; done
export DOCKER_HOST=unix:///tmp/sbx-docker.sock
export DOCKER_API_VERSION=1.49                 # VM daemon (28.1.1) max API; harbor's platform probe does not negotiate
docker version --format "docker: client {{.Client.Version}} -> server {{.Server.Version}}" || { echo "docker tunnel down"; exit 1; }
export HF_HOME=$W/hf_cache
export PYTHONPATH=$W/verl:${PYTHONPATH:-}
export LLM_PROXY_IP=127.0.0.1                  # the agent runs in this pod
export REMOTE_AGENT_USE_LOCAL_TRIAL=true
export REMOTE_AGENT_IMPORT_PATH=recipe.rlmath.rl_terminus:RLTerminus2
export REMOTE_MODEL_NAME=openai/policy
# the recipe proxy defaults max_tokens to 2048 per call when the request omits it, which cuts
# Qwen3-Thinking off mid-reasoning (no </think>, no command) -> always send max_tokens.
export REMOTE_AGENT_KWARGS="{\"max_turns\": $MAX_TURNS, \"llm_call_kwargs\": {\"max_tokens\": $TURN_TOKENS}, \"model_info\": {\"max_input_tokens\": $((MAX_PROMPT + MAX_RESPONSE)), \"max_output_tokens\": 16384, \"input_cost_per_token\": 0, \"output_cost_per_token\": 0}}"
export REMOTE_AGENT_ENVIRONMENT_IMPORT_PATH=recipe.rlmath.remote_docker:RemoteDockerEnvironment
export REMOTE_AGENT_ENVIRONMENT_KWARGS='{}'
export REMOTE_AGENT_MAX_RETRIES=2
export REMOTE_JOB_ID=$EXP
export RAY_TMPDIR=/tmp/ray_rlmath
# own Ray cluster on non-default ports so we never attach to bigroll3's cluster
unset RAY_ADDRESS
# NEVER `ray stop` here: it kills every Ray process on the machine, including bigroll3.
# Clean up only a previous rlmath cluster (processes referencing our temp dir).
pkill -f -- "--temp-dir=$RAY_TMPDIR" 2>/dev/null || true; pkill -f "$RAY_TMPDIR/session" 2>/dev/null || true; sleep 2
ray start --head --port 6391 --dashboard-port 8291 --temp-dir $RAY_TMPDIR --num-gpus 4 \
  --ray-client-server-port 10091 --min-worker-port 30000 --max-worker-port 39999 --disable-usage-stats
export RAY_ADDRESS=127.0.0.1:6391
cd $W/verl
python3 -m recipe.agentic.agentic_main \
  actor_rollout_ref.model.path=Qwen/Qwen3-4B-Thinking-2507 \
  +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_fused_kernels=true \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
  actor_rollout_ref.actor.entropy_checkpointing=true \
  data.train_harbor_dir=${TRAIN_DIR:-$W/data/tasks_train} \
  data.val_harbor_dir=$VAL_DIR \
  data.return_raw_chat=true \
  data.train_batch_size=$TRAIN_BS \
  data.max_prompt_length=$MAX_PROMPT \
  data.max_response_length=$MAX_RESPONSE \
  data.filter_overlong_prompts=true \
  data.truncation=error \
  data.prompt_key=instance_id \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BS \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT + MAX_RESPONSE)) \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((MAX_PROMPT + MAX_RESPONSE)) \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=$ROLLOUT_N \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.agent.num_workers=8 \
  actor_rollout_ref.rollout.agent.default_agent_loop=remote_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=recipe/agentic/remote-agent.yaml \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=rlmath \
  trainer.experiment_name=$EXP \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.val_before_train=${VAL_BEFORE:-true} \
  trainer.test_freq=20 \
  trainer.save_freq=25 \
  trainer.total_training_steps=$TOTAL_STEPS \
  trainer.total_epochs=100 \
  trainer.default_local_dir=$W/ckpt/$EXP \
  remote_agent.agent_name=null \
  remote_agent.agent_import_path=recipe.rlmath.rl_terminus:RLTerminus2 \
  remote_agent.model_name=openai/policy \
  remote_agent.environment_import_path=recipe.rlmath.remote_docker:RemoteDockerEnvironment \
  "$@"
