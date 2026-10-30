"""Fully-async (streaming + partial rollout) agentic GRPO for rlmath Harbor tasks.

    cd <verl checkout> && python3 -m recipe.rlmath.async_main <hydra overrides>

Flow (verl@9f18abe5 verl/experimental/fully_async_policy + recipe agentic proxy):

  RLMathAsyncTaskRunner.run
    ├─ materialize Harbor task dirs -> parquet (recipe agentic_main._materialize_harbor_datasets)
    ├─ RLMathTrainer   on trainer_pool  (trainer.n_gpus_per_node GPUs, FSDP actor, no ref)
    ├─ RLMathRollouter on its own pool  (rollout.n_gpus_per_node GPUs, standalone vLLM
    │                                    replicas + GlobalRequestLoadBalancer + AgentLoopWorkers)
    ├─ AsyncProxyServerActor (named "llm_proxy_server") -> rollouter's load balancer,
    │     VLLMRayProvider patched to resume generations aborted by weight syncs
    ├─ MessageQueue; initial NCCL weight sync (version 0); optional val_before_train
    └─ rollouter.fit() || trainer.fit()
         rollouter: per prompt -> AgentLoopWorker -> n x AsyncRemoteAgentLoop (Harbor Trial
                    whose agent calls the proxy) -> RolloutSample -> MessageQueue
         trainer:   pulls ppo_mini_batch_size*require_batches prompts -> GRPO update ->
                    every trigger_parameter_sync_step steps: abort in-flight requests,
                    NCCL-broadcast weights, resume (proxy re-submits the partial turns)

Implementation lives in async_components.py / async_proxy.py / async_agent_loop.py.
"""

import logging
import os

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.device import auto_set_device


def _preflight(config) -> None:
    """Fail fast on settings that would silently turn this into something else."""
    r = config.actor_rollout_ref.rollout
    errors = []
    if os.environ.get("PROXY_SERVER_URL"):
        errors.append("PROXY_SERVER_URL is set: standalone proxy mode is not supported by the async variant")
    if not r.multi_turn.enable:
        errors.append(
            "actor_rollout_ref.rollout.multi_turn.enable must be true, otherwise fully-async forces "
            "agent_name=single_turn_agent (detach_utils.py:66-67) and the remote agent never runs"
        )
    if r.agent.default_agent_loop != "rlmath_remote_agent_async":
        errors.append(f"default_agent_loop={r.agent.default_agent_loop!r}, expected 'rlmath_remote_agent_async'")
    if config.actor_rollout_ref.hybrid_engine:
        errors.append("actor_rollout_ref.hybrid_engine must be False")
    if config.data.train_batch_size != 0 or config.data.gen_batch_size != 1:
        errors.append("data.train_batch_size must be 0 and data.gen_batch_size 1 (use actor.ppo_mini_batch_size)")
    if not r.calculate_log_probs:
        errors.append("actor_rollout_ref.rollout.calculate_log_probs must be true")
    if config.algorithm.adv_estimator == "grpo" and r.n < 2:
        errors.append("GRPO needs actor_rollout_ref.rollout.n >= 2")
    ngpu_train = config.trainer.n_gpus_per_node * config.trainer.nnodes
    ngpu_roll = config.rollout.n_gpus_per_node * config.rollout.nnodes
    tp = r.tensor_model_parallel_size
    if ngpu_roll % tp:
        errors.append(f"rollout GPUs ({ngpu_roll}) not divisible by rollout TP ({tp})")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    # RAY_EXTERNAL=1 (multi-node sbatch): GPUs come from every node of the joined cluster, not this one's mask
    if visible and os.environ.get("RAY_EXTERNAL") != "1" and ngpu_train + ngpu_roll > len(visible.split(",")):
        errors.append(f"trainer({ngpu_train}) + rollout({ngpu_roll}) GPUs > CUDA_VISIBLE_DEVICES={visible}")
    from recipe.rlmath.overlong import resolve_overlong_mode

    mode = resolve_overlong_mode(
        OmegaConf.select(config, "rlmath_async.overlong_mode", default=None),
        OmegaConf.select(config, "rlmath_async.overlong_filter", default=True),
    )
    # require_batches > 1 (several mini-batches per step) is safe in every mode: _rlmath_no_empty_rows and the filter
    # in _fit_update_actor keep >= 1 zero-advantage loss token per row, so no mini-batch can have a 0/0 loss.
    if errors:
        raise ValueError("rlmath async preflight failed:\n  - " + "\n  - ".join(errors))

    at = config.async_training
    req = config.actor_rollout_ref.actor.ppo_mini_batch_size * at.require_batches
    max_req = int(req * (1 + at.staleness_threshold) * at.trigger_parameter_sync_step)
    steps = config.rollout.total_rollout_steps // (req * at.trigger_parameter_sync_step)
    print(
        "[RLMATH ASYNC] plan: "
        f"train_gpus={ngpu_train} rollout_gpus={ngpu_roll} (tp={tp}, replicas={ngpu_roll // tp}) | "
        f"required_samples={req} prompts x n={r.n} per step | max_in_flight_per_version={max_req} prompts | "
        f"param sync every {at.trigger_parameter_sync_step} step(s) | partial_rollout={at.partial_rollout} | "
        f"total_rollout_steps={config.rollout.total_rollout_steps} prompts -> ~{steps} param versions | "
        f"bypass_mode={config.algorithm.rollout_correction.bypass_mode} | overlong_mode={mode} | "
        f"entropy_coeff={config.actor_rollout_ref.actor.entropy_coeff} | model={config.actor_rollout_ref.model.path}",
        flush=True,
    )


def _ray_init(config) -> None:
    """Same as recipe agentic_main.run_ppo's ray.init: yaml remote_agent/* -> env vars
    (shell wins), plus verl's default PPO runtime env. Must happen before verl's run_ppo,
    which only calls ray.init when Ray is not initialized yet."""
    if ray.is_initialized():
        return
    from recipe.agentic.agentic_main import collect_yaml_env_overrides

    default_runtime_env = get_ppo_ray_runtime_env()
    ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
    runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
    if config.transfer_queue.enable:
        env_vars = runtime_env_kwargs.get("env_vars", {})
        env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
        runtime_env_kwargs["env_vars"] = env_vars
    yaml_env = collect_yaml_env_overrides(config)
    if yaml_env:
        env_vars = dict(runtime_env_kwargs.get("env_vars", {}))
        for k, v in yaml_env.items():
            env_vars.setdefault(k, v)
            os.environ.setdefault(k, v)
        runtime_env_kwargs["env_vars"] = env_vars
    runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
    ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
    print(f"ray init kwargs: {ray_init_kwargs}", flush=True)
    ray.init(**OmegaConf.to_container(ray_init_kwargs))


@hydra.main(config_path="config", config_name="rlmath_async", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    from recipe.rlmath.async_components import RLMathAsyncTaskRunner

    auto_set_device(config)
    # == fully_async_main.main (fully_async_main.py:223-228)
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    config = migrate_legacy_reward_impl(config)
    _preflight(config)
    _ray_init(config)
    run_ppo(config, task_runner_class=RLMathAsyncTaskRunner)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    main()
