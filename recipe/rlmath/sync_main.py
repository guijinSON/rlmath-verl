"""Colocated (hybrid-engine) agentic GRPO for rlmath, for benchmarking against the fully-async recipe.

verl's synchronous RayPPOTrainer (recipe.agentic.agentic_main.run_ppo) with vLLM and the FSDP trainer sharing the
same GPUs: generate the whole batch, put vLLM to sleep, train, reload weights, repeat. Two pieces of agentic_main
are swapped so the rollouts are produced exactly like in the async recipe:

* Harbor dataset materialization -> recipe.rlmath.harbor_data (cache keyed on the task list, real task paths)
* the LLM proxy -> recipe.rlmath.async_proxy.start_async_proxy_server (token-in/token-out prompting, per-session
  versions, response budget / dead-session reply). partial_rollout=False: no weight sync happens mid-generation.

The agent loop is the async recipe's (rlmath_remote_agent_async: trial semaphore, infra flags, breaker).
Not included: the async trainer's hooks (overlong filter, infra exclusion, mismatch metrics); this entry point is
for step-time measurements, launched with COLOCATED=1 run_rlmath_async.sh.
"""

from __future__ import annotations

import hydra


def _start_proxy(load_balancer, model_path, host="0.0.0.0", port=0, tool_format="hermes", debug=False, session_dump_dir=None):
    from recipe.rlmath.async_proxy import start_async_proxy_server

    url = start_async_proxy_server(
        load_balancer=load_balancer,
        model_path=model_path,
        host=host,
        port=port,
        tool_format=tool_format,
        debug=debug,
        session_dump_dir=session_dump_dir,
        partial_rollout=False,
    )
    print(f"[RLMATH SYNC] rlmath async proxy (partial_rollout=False) at {url}", flush=True)
    return url


@hydra.main(config_path="../agentic/config", config_name="agentic_trainer", version_base=None)
def main(config):
    import recipe.agentic.proxyserver.ray_actor as ray_actor
    from recipe.agentic import agentic_main as am
    from recipe.rlmath.harbor_data import materialize_harbor_datasets

    from verl.experimental.reward_loop import migrate_legacy_reward_impl
    from verl.utils.device import auto_set_device

    # agentic_main.TaskRunner.run looks both up at call time (module attribute / function-local import)
    am._materialize_harbor_datasets = materialize_harbor_datasets
    ray_actor.start_proxy_server = _start_proxy

    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    am.run_ppo(config)


if __name__ == "__main__":
    main()
