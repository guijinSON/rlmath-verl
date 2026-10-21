# rlmath fully-async GRPO: design notes

This is the streaming variant of the working sync run, with partial rollout. Files are
installed as `recipe/rlmath/` in the checkout:

| file | role |
|---|---|
| `async_main.py` | Hydra entrypoint (`python3 -m recipe.rlmath.async_main`): preflight checks, `ray.init` with the recipe's env injection, verl `run_ppo` with our task runner |
| `async_components.py` | Ray actors: `RLMathAsyncTaskRunner`, `RLMathRollouter`, `RLMathTrainer`. Each subclasses the verl fully-async actor |
| `async_proxy.py` | patched proxy actor (`llm_proxy_server`) whose vLLM provider resumes generations that a weight sync aborts, and keeps per-trajectory weight versions |
| `async_agent_loop.py` | `AsyncRemoteAgentLoop(RemoteAgentLoop)`: fills `min/max_global_steps`, aligns logprobs, sets real timing metrics |
| `overlong.py` | DAPO-style overlong-filter decision, pure python (§9) |
| `harbor_data.py` | Harbor dirs → parquet with a content-keyed cache and real task paths; replaces the recipe's materializer (§9) |
| `config/rlmath_async.yaml` | fully-async trainer config plus the recipe's `data.harbor_*`, `proxy_server` and `remote_agent` sections, plus `rlmath_async.*` |
| `config/async-agent.yaml` | agent-loop registry entry `rlmath_remote_agent_async` |
| `run_rlmath_async.sh` | launch script. Same env, Ray isolation, docker tunnel and model overrides as `run_rlmath.sh`, plus the async knobs. Supports `DRY=1` and `SMOKE=1` |

No file of the Alibaba recipe (`recipe/agentic/*`) or of verl is modified. The one
monkeypatch, `VLLMRayProvider._generate` and `release_session`, is applied from
`async_proxy.py` inside the proxy actor process only. It also checks that the method
signature it replaces has not changed.

All references below are to **verl @ 9f18abe5** (`verl/...`) and to the recipe clone
**alibaba/verl-recipe @ 40dd1924** (`recipe/agentic/...`).

---------------------------------------------------------------------------------------
## 1. How verl's fully-async mode works (what we rely on)

* **Two actors, one queue.** `FullyAsyncTaskRunner._initialize_components` builds things in
  this order (`fully_async_main.py:51-115`):
  1. the trainer (`_create_trainer`, 138-157, trainer pool only, no `Role.Rollout`)
  2. the rollouter (`_create_rollouter`, 117-136: `init_workers` then `set_max_required_samples`)
  3. the `MessageQueue`
  4. `load_checkpoint`
  5. the **initial weight sync** (`_fit_update_weights`, which gives version 0)
  6. `val_before_train`

  After that, `rollouter.fit()` and `trainer.fit()` run concurrently (176-209).
* **Resource split.** The trainer pool is `trainer.n_gpus_per_node × nnodes`
  (`separation/utils.py:36-47`). The rollouter builds its own standalone vLLM replicas on
  `rollout.n_gpus_per_node` GPUs: `main()` copies `rollout.*` into
  `actor_rollout_ref.rollout.*` (`fully_async_main.py:225-226`), and the replica count is
  `world_size // (tp·dp·pp)` (`workers/rollout/llm_server.py:276-312`). Both are separate Ray
  placement groups on the same node, so trainer GPUs + rollout GPUs must be ≤ 4.
* **Server manager / LB.** The rollouter creates `FullyAsyncLLMServerManager` →
  `GlobalRequestLoadBalancer` (`fully_async_rollouter.py:806-815`, `llm_server.py:337-341`).
  AgentLoopWorkers get a `FullyAsyncLLMServerClient` that re-submits an aborted request with
  `prompt_ids + tokens_so_far` and the remaining `max_tokens` (`fully_async_rollouter.py:52-151`).
  It records `min/max_global_steps` from `extra_fields["global_steps"]`, which vLLM sets at
  `vllm_async_server.py:563`. The value is updated after every weight load at
  `vllm_rollout.py:193-197`.
* **Streaming unit = one prompt.** `_feed_samples` repeats each dataset row `rollout.n` times
  (`detach_utils.py:70`). `_process_single_sample_streaming` sends the whole group to one
  AgentLoopWorker and puts it in the queue when *all n* trajectories of that prompt are done
  (`fully_async_rollouter.py:936-958`). So stragglers no longer block a *step*, but a GRPO
  group still waits for its slowest member.
* **Freshness control.**
  - `required_samples = ppo_mini_batch_size × require_batches` prompts per trainer step.
  - `max_required_samples = required × (1+staleness_threshold) × trigger_parameter_sync_step`
    (`fully_async_rollouter.py:531-545`).
  - The rollouter stops *starting* prompts once `staleness_samples` (prompts started since the
    last sync) reaches `max_required_samples`, or once the queue is full (1085-1107).
  - After each sync, `reset_staleness` recounts it as in-flight + queued (566-597).
  - In-flight concurrency is `min(num_replicas × 16, max_required_samples)` (543-544).
* **Trainer loop.** `fit_step` (`fully_async_trainer.py:412-452`) runs these stages in order:
  1. pull `required_samples` RolloutSamples (275-332)
  2. reward: `rm_scores` from the agent loop
  3. old_log_prob
  4. advantage (GRPO grouped by `uid = uid_sample_<epoch>_<n>`, `fully_async_rollouter.py:940-948`)
  5. update
  6. `_fit_update_local_step`: every `trigger_parameter_sync_step` steps it bumps `current_param_version` (487-499)
  7. `_fit_update_weights` (501-535)
* **Weight sync always aborts.** `CheckpointEngineManager.update_weights` (`checkpoint_engine/base.py:469-514`) runs:
  1. `abort_replicas()` (483). On vLLM ≥ 0.12 this is `pause_generation(wait_for_inflight_requests=False)` (`vllm_async_server.py:710-728`)
  2. release kv
  3. NCCL broadcast
  4. `resume_kv_cache`
  5. `resume_generation_replicas()` (514)

  `async_training.partial_rollout` only decides whether *verl's* client re-submits
  (`fully_async_rollouter.py:143`). The abort happens either way.
* **Batch assembly needs versions.** `assemble_batch_from_rollout_samples` computes
  `abs(max_global_steps - min_global_steps)` per trajectory (`detach_utils.py:151-161`). The
  trainer computes `current_param_version - v` (`fully_async_trainer.py:756`). Both must be
  ints. `AgentLoopWorker._postprocess` always emits these keys, as `None` if the loop did not
  set them (`agent_loop.py:993-1007`).
* **Agent selection.** `prepare_single_generation_data` overwrites
  `agent_name="single_turn_agent"` unless `actor_rollout_ref.rollout.multi_turn.enable`
  (`detach_utils.py:66-67`). `multi_turn.enable` has no other consumer at this commit (its only
  use, `trainer/ppo/ray_trainer.py:1295`, writes a meta_info key nobody reads). So we set it to
  true, and `_preflight` enforces it.

## 2. How the recipe proxy fits in

**Where the proxy starts.** `RLMathAsyncTaskRunner._create_rollouter` does the same as verl,
then asks the rollouter actor for its LB handle (`RLMathRollouter.get_global_load_balancer`)
and starts the proxy as the named actor `llm_proxy_server`, which is the recipe's
`PROXY_ACTOR_NAME` (`recipe/agentic/proxyserver/ray_actor.py:36`). This happens before the
initial weight sync and `val_before_train`. The stock `RemoteAgentLoop._get_proxy_url`
(`remote_agent_loop.py:255-276`) therefore finds it unchanged.

The sync entrypoint gives the proxy `trainer.llm_server_manager.global_load_balancer`
(`agentic_main.py:272`). In fully-async mode that object lives inside the rollouter actor,
so we fetch it over RPC. Every agent LLM call then goes proxy → rollouter LB → a standalone
vLLM replica on the rollout GPUs.

**Is partial rollout safe with the stock proxy? No.** `VLLMRayProvider._generate` calls
`server.generate.remote(...)` directly on the vLLM actor (`vllm_provider.py:195-202`). That
bypasses `FullyAsyncLLMServerClient`. At every weight sync, every in-flight agent turn gets
`stop_reason="aborted"` with a *truncated* completion (`vllm_async_server.py:552-581`). The
provider does not recognise it (`vllm_provider.py:394-396`) and returns it to Terminus as a
normal answer. With Qwen3-Thinking that is usually cut mid-`<think>`: no JSON, a wasted turn,
and a corrupted trajectory. With a sync every ~3-5 min and 1-8 min trials, most trajectories
would be hit.

**Fix: true partial rollout inside the proxy** (`async_proxy.install_provider_patch`):

* The provider loops on `aborted`, and also on `stop_reason None`, which means the request
  ended with no finish reason. On each pass it re-acquires the sticky server and re-submits
  `prompt_ids + tokens_so_far` with `max_tokens - len(tokens_so_far)`. This mirrors
  `FullyAsyncLLMServerClient.generate`.
* The agent's HTTP request just stays open a few seconds longer. It gets **one** complete
  completion.
* The recorded trajectory stays consistent:
  - The resume is token-exact: token ids, not re-tokenized text.
  - The recorded `completion_token_ids` are the concatenation of the chunks.
  - The `completion_logprobs` are the concatenation too, each logprob from the weights that
    sampled that token.
  - The proxy's recorder and `RemoteAgentLoop._reconstruct_output` see exactly what the agent
    saw (`ray_actor.py:205-245`, `remote_agent_loop.py:973-987`).
* Per session (= trial_id) it records min/max `global_steps` over all generated tokens.
  `AsyncRemoteAgentLoop` fetches this with the Ray RPC `pop_session_versions` and writes it
  into `extra_fields["min_global_steps"/"max_global_steps"]`, so verl's partial and staleness
  metrics are exact.
* `async_training.partial_rollout=False` switches the provider to **restart** mode. The
  interrupted turn is regenerated from its original prompt on the new weights, so each turn is
  single-version. It is still transparent to the agent. This is the fallback if continuing a
  turn across versions turns out to hurt.
* Other fixes that come with the patch:
  - It awaits the ObjectRef directly. The recipe used `asyncio.to_thread(ray.get)` at
    `vllm_provider.py:202`, which capped concurrent LLM calls at the default thread pool,
    min(32, ncpu+4).
  - It recognises verl@9f18abe5's prompt-too-long message: "leaves no room to generate"
    (`vllm_async_server.py:479-483`). The recipe only matches "exceeds", so the sync run
    returns HTTP 500 there.
  - It retries transient generate errors for about 60 s.
  - It keeps logprobs the same length as token ids.

**Trajectory staleness.** A trajectory can contain tokens from several versions:
- across turns, which is inherent to streaming;
- within a turn, if a sync lands mid-generation (partial).

The staleness budget bounds how many prompts are started per version. It does not bound how
old a straggler can get: an 8-minute trial spans at most about `8 min / step_time` versions.
With lr = 1e-6 and trigger = 1 this is mild. `rlmath_async/version_lag_max` logs it.

**old_log_prob: decoupled mode, not bypass.** verl's fully-async yaml sets
`rollout_correction.bypass_mode: True`, i.e. old_log_prob = rollout logprobs
(`rollout_corr_helper.py:1102-1137`). We set **False**, which is what the sync run uses
(default `algorithm/rollout_correction.yaml`). The reason is that
`_reconstruct_output` builds the training sequence as prompt + all completions + re-tokenized
tool/user messages. Qwen3-Thinking's chat template removes earlier assistant turns'
`<think>` content when rendering the prompt the model actually saw (the reasoning of messages
before the last user message is dropped). So for turns ≥ 2, rollout logprobs are conditioned
on a different context than the training forward pass. In bypass mode the PPO ratio
π_θ/π_rollout would then be wrong. In decoupled mode it is π_θ/π_old, computed on the same
context, exactly as in sync. This context mismatch predates this work (the sync run has it
too); fixing it means rebuilding the trajectory from the exact per-turn prompt ids.

Cost of decoupled mode: one extra forward pass per step on the trainer GPU. We skip verl's
per-step CPU snapshot of the whole model (`fully_async_trainer.py:465-485`) when
`trigger_parameter_sync_step == 1`, since the current weights are π_old
(`RLMathTrainer._compute_old_log_prob`).

## 3. GPU split: 3 rollout + 1 train (default), 2 + 2 as fallback

* The 4B model is rollout-bound (the sync profile shows ~99% of step time in rollout). The
  scarce resource is decode throughput and KV-cache for about 128-192 concurrent long
  multi-turn sessions.
* KV capacity: 3 standalone replicas at `gpu_memory_utilization=0.85` give 3 × (~120 GB − 8
  GB weights) ≈ 336 GB of KV. The sync run had 4 × (0.6 × 141 − 8) ≈ 308 GB, shared with the
  trainer's resident states. So KV capacity is about the same, with ¾ of the decode compute
  but no idle tail and no training pauses.
* Trainer on 1 H200: full-FT states for 4B (fp32 params, grads and Adam ≈ 64 GB) plus
  checkpointed activations at `ppo_max_token_len_per_gpu=40960`. That fits in 141 GB without
  offload, and there is no ref model (no KL). Training throughput is about ¼ of the sync run's,
  but training was ~1% of the sync step.
* Decide with the metrics, not by guessing:
  - `fully_async/trainer/idle_ratio` is the fraction of each trainer step spent waiting for
    samples (`detach_utils.py:325-327`).
  - `fully_async/rollouter/idle_ratio` is the fraction of each version the rollouter sat
    paused (`fully_async_rollouter.py:579-588`).

  If the trainer idle ratio stays near 0 while the rollouter idle ratio is high, the trainer
  is the bottleneck: use `ROLLOUT_GPUS=2`. If the trainer OOMs, first try
  `OPTIM_OFFLOAD=true`, then 2 + 2.

## 4. Default async knobs (run_rlmath_async.sh)

| knob | default | why |
|---|---|---|
| `ROLLOUT_GPUS` / train | 3 / 1 | §3 |
| `TRAIN_BS` → `ppo_mini_batch_size` | 16 prompts × n=8 | same 128 trajectories and 1 optimizer step per step as sync. `data.train_batch_size` must be 0 |
| `STALENESS` | 0.5 | up to 24 prompts started per version. Concurrency above `required` is what hides stragglers |
| `SYNC_EVERY` | 1 | a 4B NCCL sync takes seconds. Keeps lag low and allows skipping the CPU snapshot. Each sync resets the prefix cache of all live sessions (`vllm_rollout.py:193-195`), so every in-flight session re-prefills once per step. Try 2 if that shows up |
| `REQUIRE_BATCHES` | 1 | streaming |
| `PARTIAL` | true | proxy resume (§2) |
| `MAX_CONCURRENT_PROMPTS` | null → min(3×16, 24) = 24 prompts = **192 concurrent Harbor trials** | the sync peak was 128. Set 16 if the sandbox VM or tunnel struggles |
| `TOTAL_STEPS` | 300 | `rollout.total_rollout_steps = TOTAL_STEPS × TRAIN_BS × REQUIRE_BATCHES × SYNC_EVERY` prompts |
| `ROLLOUT_MEM` | 0.85 | vLLM owns its GPUs |
| `SAVE_FREQ` / `TEST_FREQ` | 25 / 20 | in parameter versions |

Everything else is identical to the sync run: env block, Ray isolation, docker tunnel, sdpa,
no remove-padding, fused kernels, entropy chunking, lengths, lr and clip, remote_agent
overrides, wandb. `async_check/diff_sync_async.py` composes both configs and prints only the
intended differences.

## 5. Assumptions and known risks

1. **vLLM ≥ 0.12 pause semantics.**
   - What we assume: `abort_all_requests` pauses the engine and `generate()` calls made
     during the sync wait until `resume_generation` (`vllm_async_server.py:710-728, 763-772`).
   - If the node's vLLM is < 0.12: there is no pause, and a resubmitted request can start
     while NCCL is overwriting weights. verl's own client has the same issue.
   - If new requests are *rejected* while paused: the patch retries for about 60 s.
   - `DRY=1` prints the vLLM version.
2. **Aborts return partial tokens or nothing.** Both are handled. Empty output (never
   scheduled, 552-561) means the chunk is simply retried.
3. **Validation stalls the rollouter.**
   - `do_validate` is a sync method on the async rollouter actor. `_validate` goes through
     `auto_await` case 3, which blocks the actor's event loop in a thread pool
     (`utils/ray_utils.py:132-137`).
   - While validation runs (Harbor trials, minutes), no new prompts start and finished groups
     are not queued. In-flight trials keep running in the AgentLoopWorkers.
   - The trainer waits on validation anyway (`fully_async_trainer.py:537-555`).
   - Keep `TEST_FREQ` coarse and the val set small.
4. **Checkpoint resume loses in-flight prompts.** verl's own warning
   (`fully_async_rollouter.py:614-628`): the dataloader state is saved, but queued and
   in-flight samples are not.
5. **Queue overflow drops the oldest sample** (`message_queue.py:65-72`). This should not
   happen, because queue size = `max_required_samples` and the rollouter pauses first. It shows
   as `fully_async/count/dropped_stale_samples`.
6. **Group-level stragglers remain**, since a prompt's group finishes with its slowest trial.
   This is inherent to GRPO in verl's fully-async mode.
7. **Sandbox load.** 192 concurrent trials by default, against 128 at the sync peak. Watch the
   docker tunnel and the VM. The knob is `MAX_CONCURRENT_PROMPTS=16`.
8. **Trainer-side memory on 1 GPU.** This is untested. The fallbacks are `OPTIM_OFFLOAD=true`
   and `ROLLOUT_GPUS=2`.
9. **Prefix-cache reset per sync.** See §4 (`SYNC_EVERY`).
10. **Recipe drift.** `install_provider_patch` raises if `VLLMRayProvider._generate`'s
    signature changes. `logic_check.py` also checks our overrides against the verl source.
11. **Stale proxy actor.** If an actor named `llm_proxy_server` already exists (for example a
    sync run in the same Ray cluster), `start_async_proxy_server` refuses to reuse it, because
    it would point at the wrong LB. The launch script kills the previous *rlmath* Ray cluster,
    and **this also kills a running sync rlmath job**. Never run both at the same time on
    GPUs 0-3.

## 6. Static validation done (no GPU)

Everything is under `rl_stack/async_check/`:

* `py_compile` of all files, and `bash -n run_rlmath_async.sh`.
* `compose_check.py` composes `rlmath_async.yaml` against the verl@9f18abe5 config tree with
  the exact script overrides (full and `SMOKE=1`). It checks that every override key exists
  in struct mode, that interpolations resolve, and the async invariants, including the
  `engine_workers.py:557` dynamic-bsz assert.
* `diff_sync_async.py` diffs the fully composed sync config against the async one. The only
  differences are the async switches, the n_gpus split, `gpu_memory_utilization`,
  `checkpoint_engine` naive→nccl, the agent-loop name/path, EXP/ckpt dir, and
  `train_batch_size` 16→0. The output is in `sync_vs_async_diff.txt`.
* `logic_check.py` runs our code against the *real* recipe modules (`vllm_provider`,
  `remote_agent_loop`, `models`, `serversdk.client`) and the real verl pydantic
  `AgentLoopOutput`/`AgentLoopMetrics`. Only httpx, litellm, aiohttp, transformers, ray and
  torch are stubbed. It covers:
  - abort→resume token and logprob concatenation, resume prompt, max_tokens budget, LB release;
  - empty abort, `None` stop reason, prompt-too-long, transient errors, the resume bound,
    restart mode;
  - version bookkeeping;
  - the agent-loop outputs: versions, aligned logprobs, reward, no-turn trials;
  - AST checks that every overridden verl method exists with the same signature and
    async-ness at 9f18abe5;
  - import of `async_components`/`async_main`, and `_preflight` on the composed configs.

## 7. Smoke test on the GPU node (tiny batch, 3 steps)

The prerequisite is that nothing else rlmath is running on GPUs 0-3, because the script
kills the previous rlmath Ray cluster.

```bash
# 0. install (from wherever these files were copied to)
cp -r rlmath_recipe/{async_main.py,async_components.py,async_proxy.py,async_agent_loop.py,run_rlmath_async.sh,config,ASYNC_NOTES.md} \
      $W/verl/recipe/rlmath/
cd $W/verl

# 1. DRY: real-stack imports + provider-patch signature check + composed config. No Ray, no docker.
DRY=1 bash recipe/rlmath/run_rlmath_async.sh
#   expect: "vllm=0.1x..." (want >= 0.12), "[DRY] imports + provider patch OK",
#           composed async_training/rollout/rlmath_async blocks printed

# 2. SMOKE: TRAIN_BS=2 prompts x n=2, MAX_TURNS=4, 3 steps, first 16 train tasks, no val/ckpt
SMOKE=1 bash recipe/rlmath/run_rlmath_async.sh 2>&1 | tee /tmp/rlmath_async_smoke.log
```

`SMOKE=1` gives `required_samples=2` and `max_required_samples=3`. That means 3 prompts
(6 trials) in flight and 2 consumed per step, so the third prompt is usually still in flight at
the first sync. That is what exercises staleness and partial rollout.

Then run:

```bash
grep -E "RLMATH ASYNC\] plan|required_samples :|standalone replicas|proxy actor|Param sync before fit|_fit_update_weights|Start streaming|\[stream\]|ShouldPause|reset_staleness|\[partial\]|Loop collection completed|\[consume\]|global_steps: .* local_trigger_step|Training completed" /tmp/rlmath_async_smoke.log
```

**Startup (all must appear, in this order):**
- `[RLMATH ASYNC] plan: train_gpus=1 rollout_gpus=3 (tp=1, replicas=3) | required_samples=2 prompts x n=2 ... max_in_flight_per_version=3 ...`
- `[FullyAsyncLLMServerManager] Created: 3 standalone replicas (rank 0+), 0 hybrid replicas ...`
- `[FullyAsyncRollouter] required_samples : 2 max_required_samples: 3 max_queue_size: 3 total_train_steps: 3 total_rollout_steps: 6 max_concurrent_samples: 3`
- `[rlmath-async] proxy actor 'llm_proxy_server' at http://<ip>:<port> -> rollouter load balancer (partial_rollout=True, max_resumes=32)`
- `[ASYNC MAIN] Param sync before fit..` followed by `[FullyAsyncTrainer] _fit_update_weights, timing_s/param_sync: <s> seconds self.current_param_version: 0`
- `[FullyAsyncRollouter] Start streaming mode, maximum concurrent samples: 3`

**(a) Trajectories stream:**
- `[RLMathRollouter][stream] sample_0_<k> n=2 wall=<s>s rewards=[..] versions=[..] ...` lines appear one at a time, at different wall-clock times, each as soon as *its* group finishes.
- The trainer logs `[FullyAsyncTrainer] Loop collection completed: 2/2 samples, total wait time ...` after the 2nd stream line, while a 3rd prompt is still active (`active=1` in the stream line).
- `[FullyAsyncRollouter][ShouldPause] due to staleness_samples 3 >= max_required_samples 3` shows the freshness gate.

**(b) Param sync happens:**
- `_fit_update_weights, timing_s/param_sync: ... self.current_param_version: 1`, then `: 2` and `: 3`. There is one after every `[FullyAsyncTrainer] global_steps: N local_trigger_step: 1 trigger_parameter_sync_step: 1`.
- Each is followed by `[FullyAsyncRollouter][Public][reset_staleness] reset staleness_samples to: <in-flight+queued>`.

**(c) Stale and partial trajectories are accepted:**
- `[rlmath-async][partial] session=<instance>-<uuid> turn resumed after 1 abort(s) mode=continue tokens=<n> versions=0->1 ...` means a turn survived a weight sync. The agent's HTTP call was *not* failed: the trial continues, and there is no `Traceback` or HTTP 500 for that session.
- `[RLMathTrainer][consume] global_step=2|3 param_version=1|2 trajs=4 traj_versions=[0,1] lag_max=1 stale>=1 ...` means the trainer trained on trajectories generated by an older version. `multi_version>0` means some trajectory mixed versions.
- On wandb: `fully_async/count/stale_trajectory_processed > 0`, `fully_async/partial/total_partial_num > 0`, `rlmath_async/stale_frac > 0`.
- If no `[partial]` line shows up (no turn happened to be decoding at the sync instant), rerun with `SMOKE=1 STALENESS=2` so that more prompts are in flight.

**(d) Rewards flow:**
- The `[stream]` lines show `rewards=[r1, r2]`: Harbor verifier rewards, not all `None`. They also show `harbor={'completed': 2}`.
- `[consume] ... score_mean=<mean of those>` shows the trainer received them.
- On wandb: `critic/score/mean` and `critic/rewards/mean`. They are nonzero whenever any smoke trial solved its task. All zeros is still a pass for the pipeline, provided `harbor_completed=1.00`.

**Clean end:**
- `[FullyAsyncRollouter][Feed] Maximum count has been reached ... 6 >= 6`, then `[FullyAsyncTrainer] Training stopped by queue termination signal`, then `[ASYNC MAIN] Training completed or interrupted`.

**Failure signatures:**
- `rlmath async preflight failed` means a config switch was lost; the message says which.
- `Agent loop rlmath_remote_agent_async not registered` means the agent_loop_config_path does not resolve from cwd `$W/verl`.
- `TypeError: unsupported operand type(s) for -: 'NoneType'` in `assemble_batch_from_rollout_samples` means versions were not stamped (the proxy actor is not the patched one).
- A trainer OOM means `OPTIM_OFFLOAD=true`, or `ROLLOUT_GPUS=2`.

## 8. After the smoke test

Run a short full-size run first, for example `TOTAL_STEPS=10 VAL_BEFORE=false`. Compare with
the sync run on wandb:
- `perf/time_per_step` against sync `timing_s/step`
- `fully_async/trainer/idle_ratio` and `fully_async/rollouter/idle_ratio` (§3)
- `rlmath_async/version_lag_max`
- `fully_async/processing_time/*`, which is trial wall time, now populated

Then set `ROLLOUT_GPUS` and `MAX_CONCURRENT_PROMPTS` accordingly and launch the full run.

---------------------------------------------------------------------------------------
## 9. v2 fixes after the 67-step production run

This round changes: overlong filtering, the entropy bonus, `MODEL`, and task-path resolution.

**What happened.** Truncated trajectories were scored 0, which strongly penalised length.
Mean response length went from 16.7k to about 1k tokens and entropy from 0.46 to 0.2, while
the solve rate stayed flat at about 19%. The fix is to restart from the merged step-25
checkpoint with the changes below.

### Overlong filter (`OVERLONG_FILTER=true`, default) → `rlmath_async.overlong_filter`

A trajectory is **truncated** if either condition holds:

* **(a) its response hit `max_response_length`.** This is exactly verl's
  `response_length/clip_ratio` test: the response length from the attention mask equals
  `responses.shape[-1]` (`trainer/ppo/metric_utils.py:75-81, 236`, called through
  `_compute_response_info`). `RemoteAgentLoop` cuts responses to `response_length`, so an
  overflow lands exactly on the limit.
* **(b) any LLM call in it hit the per-call `max_tokens`** (`TURN_TOKENS`).
  - vLLM reports both a normal stop and a length stop as `stop_reason="completed"`
    (`vllm_async_server.py:582-583`). So `async_proxy.classify_turn` decides "length" when the
    call's tokens reach `max_tokens` and the last token is not an EOS
    (`<|im_end|>`/`<|endoftext|>`).
  - Tokens from aborted (partial) chunks count toward the budget. Only the *final* generation
    of a resumed turn can exhaust it, so a resumed turn is truncated only if that final
    generation hit the budget.
  - A few other cases also count as truncated: the synthetic prompt-too-long reply (the agent
    gets `finish_reason=length`), and a turn given up after `max_resumes`.
  - Turns that end below the budget with no EOS are logged as `no_eos_turns` for diagnosis
    only. They are not filtered.
  - Counts are kept per session in the proxy and carried as
    `extra_fields["truncated_turns"/"llm_turns"/"no_eos_turns"]`.
  - When the recipe retries a trial, the proxy's per-session record is reset, so the failed
    attempt's turns never count against the retry.

**What the filter does to the loss.**

1. `RLMathTrainer._fit_update_actor` runs after `_fit_compute_advantage`
   (`fully_async_trainer.py:441-443`). The GRPO group statistics and the rewards are untouched.
2. It multiplies `response_mask` by the keep mask. That mask becomes the loss mask
   (`workers/utils/padding.py:71`); `losses.ppo_loss` uses it for the pg and entropy terms,
   and token-mean normalizes by its sum (`engine/fsdp/transformer_impl.py:621-626`).
3. It calls the normal update.
4. It restores the original mask, so the `response_length/*` metrics stay comparable.

**Guards.**
- If no loss tokens remain (every trajectory truncated or empty), the actor update is
  **skipped** for that step. Otherwise the token-mean would be 0/0 and give NaN gradients.
- `_preflight` rejects `overlong_filter` with `require_batches > 1`, because a single mini-batch
  could then be fully masked.
- `OVERLONG_FILTER=false` still logs everything but drops nothing.

**Risk.** With Qwen3-Thinking at `TURN_TOKENS=8192` over up to 8 turns, criterion (b) can hit
a large share of trajectories. Watch `rlmath/overlong_filtered_frac` and
`rlmath/truncated_turn_frac` in the first steps. If most of the batch is filtered, learning
slows down, and every-step skips are logged. Raise `TURN_TOKENS` or reconsider criterion (b)
before running long.

### Entropy bonus (`ENTROPY_COEFF=0.001`, default) → `actor_rollout_ref.actor.entropy_coeff`

* The key exists at 9f18abe5 with default 0 (`trainer/config/actor/actor.yaml:89`,
  `workers/config/actor.py:166`).
* A nonzero value turns on `calculate_entropy` (`trainer/ppo/ray_trainer.py:1302-1304`), and
  `ppo_loss` subtracts `entropy_coeff × token-mean entropy` over the loss mask
  (`workers/utils/losses.py:122-128`).
* **With `use_fused_kernels=true`, entropy comes from the fused kernel output**
  (`transformer_impl.py:1093-1096`). The default backend is `torch`
  (`trainer/config/model/hf_model.yaml:64-67`), and its backward adds the exact entropy
  gradient `-p(log p + H)·dH` (`utils/experimental/torch_functional.py:73-77`). The triton
  backend also takes `dentropy` (`utils/kernel/linear_cross_entropy.py:90-102`). So entropy
  gradients do flow.
* `entropy_from_logits_with_chunking` and `entropy_checkpointing` only apply to the
  **non-fused** path (`transformer_impl.py:1118-1133`), so with fa2 + fused they are inert.
  That is harmless. No fix is needed.
* With the filter on, the bonus only covers the tokens that stay in the loss.

### `MODEL` (default `Qwen/Qwen3-4B-Thinking-2507`)

* It sets `actor_rollout_ref.model.path`, which every consumer reads: the FSDP actor, the vLLM
  replicas, the trainer and rollouter tokenizers, and the proxy tokenizer.
* For the restart, use `MODEL=$W/ckpt/merged/async_v1_step25` with a **new `EXP`**.
  `trainer.resume_mode=auto` would otherwise load `$W/ckpt/$EXP/global_step_*` over the merged
  weights, and the script now refuses that case unless `ALLOW_RESUME=1`.
* The optimizer state starts fresh.

### Task-path resolution (`REMOTE_AGENT_TASK_PATH_TEMPLATE`, validation FileNotFoundError)

* The recipe's rows already carry a top-level `local_task_path` (`local_harbor.py:117-120`),
  and it is checked first (`remote_agent_loop.py:238-241`). Validation keeps every
  non-tensor column (`ray_trainer.py:575-589`).
* So a val trial only falls through to the template when *both* its `local_task_path` and
  `<val_harbor_dir>/<id>` are missing at run time, and `<train_harbor_dir>/<id>` is missing
  too. The last one is new since train moved to `tasks_nontrivial`, which lacks the
  `agentic_trivial` tasks.
* The likely cause is the recipe's parquet cache. Its key is `md5(root.resolve(), split,
  limit)` (`local_harbor.py:159-162`), and it is reused while `harbor_overwrite` is false. So
  after `val_tasks/` was rebuilt, the old val parquet with old ids and paths kept being used.
  Relative symlink targets, which `make_val.py` created when given a relative src, would make
  the same failure mode possible.

To confirm on the pod (read-only):

```bash
python3 - <<'PY'
import glob, os, pandas as pd
for f in glob.glob(os.path.expanduser("~/.cache/verl/agentic/harbor/val-*.parquet")):
    df = pd.read_parquet(f); bad = [p for p in df.local_task_path if not os.path.isdir(p)]
    print(f, len(df), "missing:", len(bad), bad[:3])
PY
find $W/data/val_tasks -maxdepth 1 -xtype l | head   # dangling links
```

**Fix, all from our code:**
- `harbor_data.materialize_harbor_datasets` replaces the recipe materializer:
  - the cache key covers the task list itself (id, real path, and mtimes of `task.toml` and `instruction.md`);
  - `local_task_path` is the realpath (`data/tasks/<id>`, not the symlink);
  - every row's dir is checked at startup, and a missing one is a hard error rather than a silent 0.
- The script exports `REMOTE_AGENT_TASK_PATH_TEMPLATE=${REMOTE_AGENT_TASK_PATH_TEMPLATE:-$W/data/tasks/{instance_id}}` as the last-resort lookup.
- `make_val.py` now links to absolute targets.
- The sync `run_rlmath.sh` still uses the recipe materializer. Pass `data.harbor_overwrite=true` there if you rerun it.

### What proves it in the first steps

**Log lines:**
- `[rlmath-data] val: N tasks from .../val_tasks (N via symlink, local_task_path=realpath) -> ...val-rlmath-<sig>.parquet`
  appears at startup; same for train. There should be no `FileNotFoundError` for
  `/home/verl/dataset-tasks/...` in the val trials.
- `[RLMATH ASYNC] plan: ... overlong_filter=True | entropy_coeff=0.001 | model=<MODEL>`
- `[rlmath-async][trunc] session=<trial> turn truncated reason=length tokens=8192 max_tokens=8192 aborts=<k> (total_truncated_turns=<t>/<all>)`
  (rate-limited: the first 20, then every 100th).
- `[RLMathRollouter][stream] ... truncated_turns=<x>/<y> ...` on each streamed group.
- `[RLMathTrainer][overlong] global_step=<s> param_version=<v> enabled=True trajs=128 truncated=<T> (resp_len=<a>, turn=<b>) filtered=<T> loss_trajs=<128-T-empty> loss_tokens=<n> truncated_turn_frac=<f>`
  on every step. When enabled, `filtered == truncated`.
- `... no loss tokens left after filtering -> SKIPPING actor update this step` appears only in the all-filtered case.

**Wandb metrics (logged per param version):**
- `rlmath/overlong_filtered_frac`
- `rlmath/truncated_turn_frac`
- `rlmath/loss_trajectories`
- `rlmath/loss_tokens`
- `rlmath/truncated_traj_frac`
- `rlmath/truncated_by_response_len_frac`
- `rlmath/truncated_by_turn_frac`
- `rlmath/actor_update_skipped`
- `rlmath/overlong_filter_enabled`
- `actor/entropy_loss`, which now appears because entropy is computed in the update

**Cross-checks:**
- `rlmath/truncated_by_response_len_frac` must equal verl's `response_length/clip_ratio` for the same step (same definition, same batch).
- `rlmath/overlong_filtered_frac` equals `rlmath/truncated_traj_frac` with the filter on, and is 0 with it off.

### Verification done (no GPU)

`async_check/logic_check.py` now has 94 checks, all passing. They add:
- truncation classification through the real patched provider: budget hit, EOS at the budget, resumed turns whose final generation does or does not hit the budget, an aborted chunk that exhausts the budget, prompt-too-long, gave-up, and the reset on retry;
- the agent-loop fields;
- `overlong_decision` cases: mixed, disabled, all-filtered, missing fields;
- the *real* `RLMathTrainer._fit_update_actor` against a numpy stand-in for the batch: the mask is zeroed only during the update and restored afterwards, the disabled path is unchanged, and the all-filtered case skips the update without crashing;
- AST and string checks against 9f18abe5 for every verl fact cited above;
- `harbor_data` on a temp tree with symlinked val tasks: realpath, cache invalidation when the dir changes, and the missing-dir error.

`compose_check.py` and `diff_sync_async.py` still pass. The new config diffs are `entropy_coeff` 0→0.001 and `rlmath_async.overlong_filter`. The attention diffs there only come from `ATTN=auto` resolving to sdpa on the laptop.
