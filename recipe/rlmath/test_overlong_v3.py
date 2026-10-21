import numpy as np, torch
from recipe.rlmath.overlong import grpo_advantages_excluding, is_infra_failure, resolve_overlong_mode
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

torch.manual_seed(0)
B, L = 12, 5
uid = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 3 + ["d"])
scores = torch.tensor([1, 0, 0, 1, 0.5, 0.1, 0, 1, 0, 0, 0, 1.0])
tlr = torch.zeros(B, L); tlr[:, -1] = scores
mask = torch.ones(B, L)
adv_v, _ = compute_grpo_outcome_advantage(tlr, mask, uid)
ours = grpo_advantages_excluding(scores, uid, [True] * B)
assert torch.allclose(adv_v[:, -1], ours, atol=1e-6), (adv_v[:, -1], ours)
valid = [True, True, False, True] + [True] * 4 + [False, False, False] + [True]
ex = grpo_advantages_excluding(scores, uid, valid)
ref_a = compute_grpo_outcome_advantage(tlr[[0, 1, 3]], mask[[0, 1, 3]], uid[[0, 1, 3]])[0][:, -1]
assert torch.allclose(ex[[0, 1, 3]], ref_a, atol=1e-6) and ex[2] == 0 and (ex[8:11] == 0).all()
print("grpo exclusion OK", ex.tolist())
assert is_infra_failure("error", "All 2 retries exhausted: Agent run error: RuntimeError: Failed to detect Docker platform: Cannot connect to the Docker daemon")
assert is_infra_failure("error", "All 2 retries exhausted: Agent run error: EnvironmentStartTimeoutError: Environment start timed out after 600.0 seconds")
assert is_infra_failure("error", "AddTestsDirError: Failed to add tests directory to environment.")
assert not is_infra_failure("error", "All 2 retries exhausted: Agent run error: AgentTimeoutError: Agent execution timed out after 7200 seconds")
assert not is_infra_failure("completed", "docker something")
assert resolve_overlong_mode(None, True) == "filter" and resolve_overlong_mode("penalize", True) == "penalize" and resolve_overlong_mode(None, False) == "off"
print("infra classifier + mode OK")
from recipe.rlmath.overlong import is_llm_server_failure
e500 = "All 2 retries exhausted: Agent run error: InternalServerError: litellm.InternalServerError: InternalServerError: OpenAIException - Error code: 500 - {'error': 'litellm.InternalServerError: Verl-vllmException - ... vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.'}"
assert is_infra_failure("error", e500) and is_llm_server_failure("error", e500)
assert not is_llm_server_failure("error", "AddTestsDirError: Failed to add tests directory to environment.")
assert not is_infra_failure("error", "All 2 retries exhausted: Agent run error: AgentTimeoutError: Agent execution timed out after 7200 seconds")
print("llm-server classifier OK")
