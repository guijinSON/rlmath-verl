"""Ray actors for the rlmath fully-async run (subclasses of verl@9f18abe5 fully_async_policy).

Kept in an importable module (not the ``python -m`` __main__) so Ray/cloudpickle ship
these classes by reference.

* RLMathRollouter   - FullyAsyncRollouter + a getter for its GlobalRequestLoadBalancer
                      (needed by the proxy), an optional in-flight cap, a per-sample log line.
* RLMathTrainer     - FullyAsyncTrainer that skips the CPU weight snapshot when
                      trigger_parameter_sync_step == 1, logs version lag per step, and applies
                      the DAPO-style overlong filter (overlong.py) right before the actor update.
* RLMathAsyncTaskRunner - FullyAsyncTaskRunner that materializes the Harbor parquet,
                      uses the two classes above and starts the patched proxy between
                      rollouter.init_workers() and the first weight sync / validation.
"""

from __future__ import annotations

import time

import numpy as np
import ray
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.experimental.separation.utils import create_resource_pool_manager
from verl.trainer.ppo.utils import Role

# ray.remote(cls) keeps the undecorated class at ActorClass.__ray_actor_class__
# (same trick verl uses in single_controller/ray/base.py:969-970).
_RolloutBase = FullyAsyncRollouter.__ray_actor_class__
_TrainerBase = FullyAsyncTrainer.__ray_actor_class__
_RunnerBase = FullyAsyncTaskRunner.__ray_actor_class__


def _as_int_list(arr) -> list[int]:
    out = []
    for x in arr if arr is not None else []:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            pass
    return out


# --------------------------------------------------------------------------- rollouter
def _kept_mismatch_metrics(batch, resp_mask, keep) -> dict:
    """vLLM-vs-trainer mismatch on the tokens that actually reach the loss (after the overlong
    filter), next to the same numbers on all response tokens. verl's rollout_corr/* is computed
    before the filter, so it also counts trajectories the loss never sees. With r = log pi_old -
    log pi_rollout per token (tokens sampled from pi_rollout): k1 = mean(-r) and k3 = mean(e^r - 1 - r)
    both estimate KL(pi_rollout || pi_old)."""
    try:
        import torch

        bt = batch.batch
        if "rollout_log_probs" not in bt.keys() or "old_log_probs" not in bt.keys():
            return {}
        r = (bt["old_log_probs"] - bt["rollout_log_probs"]).float()
        all_m = resp_mask.float()
        keep_t = torch.tensor(keep, dtype=all_m.dtype, device=all_m.device).unsqueeze(-1)
        out = {}
        for name, m in (("all", all_m), ("kept", all_m * keep_t)):
            denom = m.sum().clamp_min(1.0)
            out[f"rlmath/mismatch_{name}_k1"] = float(((-r) * m).sum() / denom)
            out[f"rlmath/mismatch_{name}_k3"] = float(((torch.exp(r) - 1 - r) * m).sum() / denom)
        return out
    except Exception as e:  # noqa: BLE001 - metrics only
        print(f"[RLMathTrainer][mismatch] metric failed ({e!r})", flush=True)
        return {}


class _RLMathRollouter(_RolloutBase):
    def get_global_load_balancer(self):
        """Handle of the rollouter's GlobalRequestLoadBalancer (created in
        FullyAsyncLLMServerManager.create -> LLMServerManager._init_global_load_balancer,
        llm_server.py:337-341). The proxy routes every agent LLM call through it."""
        return self.llm_server_manager.global_load_balancer

    async def set_max_required_samples(self):
        await super().set_max_required_samples()
        self._rlmath_cap_max = self.max_concurrent_samples  # verl's own cap (replicas x 16, <= max_required_samples)
        cap = OmegaConf.select(self.config, "rlmath_async.max_concurrent_prompts", default=None)
        if cap:
            async with self.lock:
                old = self.max_concurrent_samples
                self.max_concurrent_samples = max(1, min(old, int(cap)))
            print(
                f"[RLMathRollouter] max_concurrent_samples capped {old} -> {self.max_concurrent_samples} prompts "
                f"(x rollout.n={self.config.actor_rollout_ref.rollout.n} trials)",
                flush=True,
            )
        from recipe.rlmath.limits import LIMITS_FILE

        if LIMITS_FILE and getattr(self, "_rlmath_limits_task", None) is None:
            import asyncio

            self._rlmath_limits_task = asyncio.create_task(self._rlmath_watch_limits())

    async def _rlmath_watch_limits(self):
        """Apply `max_concurrent_prompts` from RLMATH_LIMITS_FILE while running (clamped to verl's own cap). verl's
        streaming loop re-reads self.max_concurrent_samples before admitting each new prompt."""
        import asyncio

        from recipe.rlmath.limits import TTL, read_limits

        while True:
            try:
                want = read_limits().get("max_concurrent_prompts")
                if want is not None:
                    new = max(1, min(int(want), self._rlmath_cap_max))
                    if new != self.max_concurrent_samples:
                        # plain attribute write (single event loop); no self.lock: verl's streaming loop holds it while
                        # waiting for a task to finish, which would only delay this
                        old, self.max_concurrent_samples = self.max_concurrent_samples, new
                        print(f"[RLMathRollouter][limits] max_concurrent_samples {old} -> {new} prompts", flush=True)
            except Exception as e:  # noqa: BLE001 - tuning only
                print(f"[RLMathRollouter][limits] watch failed ({e!r})", flush=True)
            await asyncio.sleep(TTL)

    async def _process_single_sample_streaming(self, rollout_sample):
        t0 = time.time()
        await super()._process_single_sample_streaming(rollout_sample)
        try:  # evidence line (a)/(d) for the smoke test; never let logging break rollout
            b = rollout_sample.full_batch
            ntb = b.non_tensor_batch
            vmin = _as_int_list(ntb.get("min_global_steps"))
            vmax = _as_int_list(ntb.get("max_global_steps"))
            rewards = b.batch["rm_scores"].sum(-1).tolist() if "rm_scores" in b.batch.keys() else []
            status = ntb.get("harbor_status")
            status_counts = {}
            for s in status if status is not None else []:
                status_counts[str(s)] = status_counts.get(str(s), 0) + 1
            resumes = sum(_as_int_list(ntb.get("num_resumes")))
            trunc_turns = sum(_as_int_list(ntb.get("truncated_turns")))
            llm_turns = sum(_as_int_list(ntb.get("llm_turns")))
            print(
                f"[RLMathRollouter][stream] {rollout_sample.sample_id} n={len(b)} wall={time.time() - t0:.0f}s "
                f"rewards={[round(r, 3) for r in rewards]} versions=[{min(vmin, default=None)},"
                f"{max(vmax, default=None)}] resumes={resumes} truncated_turns={trunc_turns}/{llm_turns} "
                f"harbor={status_counts} "
                f"active={len(self.active_tasks)} produced={self.total_generated_samples} "
                f"dropped={self.dropped_stale_samples} staleness_samples={self.staleness_samples}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[RLMathRollouter][stream] log failed: {e}", flush=True)


RLMathRollouter = ray.remote(num_cpus=10, max_concurrency=100)(_RLMathRollouter)


# --------------------------------------------------------------------------- trainer
class _RLMathTrainer(_TrainerBase):
    async def load_checkpoint(self):
        out = await super().load_checkpoint()
        # verl sets global_steps = version * trigger + 1 on resume and fit() adds 1 more, so the first resumed step was
        # labelled training/global_step = version + 2 (ckpt 40 -> 42). A fresh start labels step 1 as 1; match that.
        if self.current_param_version > 0 and self.global_steps == self.current_param_version * self.trigger_parameter_sync_step + 1:
            self.global_steps -= 1
            print(f"[RLMathTrainer] resume: global_steps -> {self.global_steps} (first resumed step logs as {self.global_steps + 1})", flush=True)
        return out

    async def fit_step(self, batch_dict=None):
        out = await super().fit_step(batch_dict)
        self._rlmath_flush_step_metrics()
        return out

    def _rlmath_flush_step_metrics(self):
        """verl logs step k's aggregated training metrics only at step k+1's weight sync (_fit_update_weights), and W&B
        keeps a row open until a later step is logged -> W&B trailed training by ~2 steps (hours per step here). After
        fit_step k (its weight sync, validation and checkpoint already logged at step k) log k's metrics at step k and
        commit the row. The next weight sync then finds the aggregator empty and logs only its own timing at k+1."""
        if self.trigger_parameter_sync_step != 1:
            return
        try:
            agg = self.metrics_aggregator.get_aggregated_metrics()
            if agg:
                self.logger.log(data=agg, step=self.current_param_version)
                self.metrics_aggregator.reset()
            wb = getattr(self.logger, "logger", {}).get("wandb")
            if wb is not None:
                wb.log({}, step=self.current_param_version, commit=True)
        except Exception as e:  # noqa: BLE001 - logging only
            print(f"[RLMathTrainer] step-metric flush failed ({e!r})", flush=True)

    def _compute_old_log_prob(self, batch):
        # FullyAsyncTrainer._compute_old_log_prob (fully_async_trainer.py:465-485) snapshots the
        # whole model to CPU every step so that later local steps can recompute with version-1
        # weights. With trigger_parameter_sync_step == 1 there are no later local steps, and the
        # current weights ARE pi_old -> call the plain RayPPOTrainer implementation directly.
        if self.trigger_parameter_sync_step == 1:
            return SeparateRayPPOTrainer._compute_old_log_prob(self, batch)
        return super()._compute_old_log_prob(batch)

    def _rlmath_overlong_mode(self) -> str:
        from recipe.rlmath.overlong import resolve_overlong_mode

        return resolve_overlong_mode(
            OmegaConf.select(self.config, "rlmath_async.overlong_mode", default=None),
            OmegaConf.select(self.config, "rlmath_async.overlong_filter", default=True),
        )

    def _fit_compute_reward(self, batch):
        """fit_step runs _fit_compute_reward -> _fit_compute_advantage -> ... -> _fit_update_actor
        (fully_async_trainer.py:435-446). Before the advantage: flag truncated and infra-failed
        trajectories, and in overlong_mode=penalize zero the reward of truncated ones (self.reward_tensor
        becomes token_level_scores/rewards in SeparateRayPPOTrainer._fit_compute_advantage)."""
        batch = super()._fit_compute_reward(batch)
        self._rlmath_infra = None
        if hasattr(self, "_rlmath_raw"):
            del self._rlmath_raw
        try:
            from recipe.rlmath.overlong import overlong_decision

            from verl.trainer.ppo.metric_utils import _compute_response_info

            mode = self._rlmath_overlong_mode()
            ntb = batch.non_tensor_batch
            n = len(batch)
            keep, _ = overlong_decision(
                response_lengths=_compute_response_info(batch)["response_length"].tolist(),
                max_response_length=int(batch.batch["responses"].shape[-1]),
                truncated_turns=ntb.get("truncated_turns"),
                llm_turns=ntb.get("llm_turns"),
                loss_tokens=batch.batch["response_mask"].sum(-1).tolist(),
                enabled=True,
            )
            truncated = [not k for k in keep]
            infra_raw = ntb.get("infra_failed")
            infra = [bool(int(x)) for x in infra_raw] if infra_raw is not None else [False] * n
            drop_infra = bool(OmegaConf.select(self.config, "rlmath_async.drop_infra_failed", default=True))
            self._rlmath_infra = infra if drop_infra else None

            rt = self.reward_tensor
            raw = rt.sum(-1).float()
            self._rlmath_raw = raw.tolist()
            valid = [not x for x in infra]
            nv = max(sum(valid), 1)
            pen = [t and not f for t, f in zip(truncated, infra, strict=True)]
            lost = float(sum(float(raw[i]) for i in range(n) if pen[i]))
            if mode == "penalize" and any(pen):
                import torch

                rows = torch.tensor(pen, dtype=torch.bool, device=rt.device)
                rt[rows] = 0
            self.metrics.update(
                {
                    "rlmath/overlong_mode_penalize": float(mode == "penalize"),
                    "rlmath/penalized_frac": (sum(pen) / max(n, 1)) if mode == "penalize" else 0.0,
                    "rlmath/penalized_reward_lost": (lost / max(n, 1)) if mode == "penalize" else 0.0,
                    "rlmath/infra_failed_frac": sum(infra) / max(n, 1),
                    "rlmath/infra_failed_count": float(sum(infra)),
                    # reward of the trajectories that really ran, before the length penalty
                    "rlmath/score_mean_valid_raw": float(sum(float(raw[i]) for i in range(n) if valid[i]) / nv),
                }
            )
            print(
                f"[RLMathTrainer][reward] global_step={self.global_steps} mode={mode} trajs={n} "
                f"truncated={sum(truncated)} penalized={sum(pen) if mode == 'penalize' else 0} "
                f"reward_lost={lost:.2f} infra_failed={sum(infra)} (excluded_from_groups={drop_infra})",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 - never lose a step to bookkeeping
            print(f"[RLMathTrainer][reward] penalty/infra bookkeeping failed ({e!r}); rewards unchanged", flush=True)
        try:  # rlmath/group/* (GRPO group saturation) + rlmath/tools/* (sandbox tool usage), on the raw rewards
            from recipe.rlmath.tool_metrics import group_metrics, tool_metrics

            ntb = batch.non_tensor_batch
            raw = self.reward_tensor.sum(-1).float().tolist() if not hasattr(self, "_rlmath_raw") else self._rlmath_raw
            infra = getattr(self, "_rlmath_infra", None) or [False] * len(batch)
            valid = [not x for x in infra]
            self.metrics.update(group_metrics(ntb["uid"], raw, valid, ntb.get("task_kind")))
            if ntb.get("tool_cmds") is not None:
                self.metrics.update(tool_metrics(ntb["tool_cmds"], ntb["tool_libs"], ntb["tool_exes"], raw, valid))
        except Exception as e:  # noqa: BLE001
            print(f"[RLMathTrainer][reward] group/tool metrics failed ({e!r})", flush=True)
        try:  # rlmath/optimize/* from the beat-best grader's flags (recipe/rlmath/optimize_beat.py)
            from recipe.rlmath.optimize_beat import optimize_metrics

            ntb = batch.non_tensor_batch
            if ntb.get("opt_beat_best") is not None and ntb.get("task_kind") is not None:
                infra = getattr(self, "_rlmath_infra", None) or [False] * len(batch)
                if not hasattr(self, "_rlmath_opt_totals"):
                    self._rlmath_opt_totals = {}
                om = optimize_metrics(
                    ntb["task_kind"],
                    ntb["opt_beat_best"],
                    ntb["opt_reach_best"],
                    ntb["opt_counterexample"],
                    [not x for x in infra],
                    self._rlmath_opt_totals,
                )
                self.metrics.update(om)
                if om["rlmath/optimize/beat_best_count"] > 0:
                    print(
                        f"[RLMathTrainer][optimize] global_step={self.global_steps} BEAT best_known in "
                        f"{int(om['rlmath/optimize/beat_best_count'])} trajectories "
                        f"(cum {int(om['rlmath/optimize/beat_best_cum'])}); witnesses in RLMATH_BEAT_DIR",
                        flush=True,
                    )
        except Exception as e:  # noqa: BLE001
            print(f"[RLMathTrainer][reward] optimize metrics failed ({e!r})", flush=True)
        return batch

    def _fit_compute_advantage(self, batch):
        """verl's GRPO advantage, then (if any trajectory is infra-failed) recompute it with those rows left
        out of their group's mean/std. Infra-failed rows are synthetic (no tokens, response_mask [0]) and get
        advantage 0; without this their reward 0 drags the group mean down and inflates the siblings' credit."""
        rm0 = batch.batch["response_mask"]
        if float(rm0.sum()) == 0:
            # every trajectory is empty (e.g. all infra-failed): verl's rollout correction raises "response_mask must
            # contain at least one valid token" and kills the run. Zero advantages instead; _fit_update_actor then
            # skips the update (rlmath/actor_update_skipped).
            import torch

            print(
                f"[RLMathTrainer][adv] global_step={self.global_steps}: no trainable tokens in the batch "
                "(all trajectories empty) -> zero advantages, update will be skipped",
                flush=True,
            )
            batch.batch["token_level_scores"] = self.reward_tensor
            batch.batch["token_level_rewards"] = self.reward_tensor
            z = torch.zeros_like(rm0, dtype=torch.float32)
            batch.batch["advantages"] = z
            batch.batch["returns"] = z
            return batch
        batch = super()._fit_compute_advantage(batch)
        batch = self._rlmath_exclude_infra(batch)
        return self._rlmath_no_empty_rows(batch)

    def _rlmath_no_empty_rows(self, batch):
        """Give every trajectory at least one loss token, with advantage 0.

        verl's token-mean loss divides by the mini-batch's loss-token count. With several mini-batches per step
        (async_training.require_batches > 1) one mini-batch could consist only of empty rows (infra-failed or
        masked trajectories, e.g. during a sandbox outage) -> 0/0 -> NaN gradients. Marking the first response
        token of each empty row as a loss token with advantage 0 contributes nothing to the policy gradient but
        keeps every denominator > 0. Runs after the advantage, so the advantage of those rows is already 0."""
        try:
            rm = batch.batch["response_mask"]
            empty = rm.sum(-1) == 0
            n = int(empty.sum())
            if n and n < rm.shape[0]:
                rm[empty, 0] = 1
                for k in ("advantages", "returns"):
                    if k in batch.batch.keys():
                        batch.batch[k][empty] = 0
                print(f"[RLMathTrainer][adv] global_step={self.global_steps}: {n} empty trajectories get one zero-advantage loss token", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[RLMathTrainer][adv] empty-row guard failed ({e!r})", flush=True)
        return batch

    def _rlmath_exclude_infra(self, batch):
        infra = getattr(self, "_rlmath_infra", None)
        if not infra or not any(infra):
            return batch
        try:
            from recipe.rlmath.overlong import grpo_advantages_excluding

            est = str(getattr(self.config.algorithm.adv_estimator, "value", self.config.algorithm.adv_estimator))
            if est.lower() != "grpo":
                print(f"[RLMathTrainer][adv] adv_estimator={est}: infra exclusion only implemented for grpo", flush=True)
                return batch
            rm = batch.batch["response_mask"]
            adv = grpo_advantages_excluding(
                batch.batch["token_level_rewards"].sum(-1),
                batch.non_tensor_batch["uid"],
                [not x for x in infra],
                norm_by_std=bool(self.config.algorithm.get("norm_adv_by_std_in_grpo", True)),
            )
            a = adv.to(device=rm.device, dtype=batch.batch["advantages"].dtype).unsqueeze(-1) * rm
            batch.batch["advantages"] = a
            batch.batch["returns"] = a
            print(
                f"[RLMathTrainer][adv] global_step={self.global_steps} recomputed GRPO advantages "
                f"excluding {sum(infra)} infra-failed trajectories",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[RLMathTrainer][adv] infra exclusion failed ({e!r}); keeping verl's advantages", flush=True)
        return batch

    def _fit_update_actor(self, batch):
        """Overlong filter. The GRPO advantages were already computed on the full group
        (SeparateRayPPOTrainer._fit_compute_advantage runs before this in fit_step,
        fully_async_trainer.py:435-446). Here the response_mask of every truncated trajectory
        is zeroed for the update only. verl turns response_mask into the loss mask
        (workers/utils/padding.py:71; losses.py ppo_loss uses response_mask for the pg and
        entropy terms), then the original mask is restored so response_length/* metrics
        (_fit_collect_metrics) are unchanged. Only in overlong_mode=filter; penalize already acted
        on the rewards in _fit_compute_reward."""
        from recipe.rlmath.overlong import overlong_decision

        from verl.trainer.ppo.metric_utils import _compute_response_info

        enabled = self._rlmath_overlong_mode() == "filter"
        try:
            resp_mask = batch.batch["response_mask"]
            resp_len = _compute_response_info(batch)["response_length"].tolist()
            max_len = int(batch.batch["responses"].shape[-1])
            ntb = batch.non_tensor_batch
            keep, om = overlong_decision(
                response_lengths=resp_len,
                max_response_length=max_len,
                truncated_turns=ntb.get("truncated_turns"),
                llm_turns=ntb.get("llm_turns"),
                loss_tokens=resp_mask.sum(-1).tolist(),
                enabled=enabled,
            )
        except Exception as e:  # noqa: BLE001 - never lose a step to bookkeeping
            print(f"[RLMathTrainer][overlong] decision failed ({e!r}); updating without filter", flush=True)
            return super()._fit_update_actor(batch)

        self.metrics.update(om)
        self.metrics.update(_kept_mismatch_metrics(batch, resp_mask, keep))
        n = len(keep)
        print(
            f"[RLMathTrainer][overlong] global_step={self.global_steps} param_version={self.current_param_version} "
            f"enabled={enabled} trajs={n} truncated={round(om['rlmath/truncated_traj_frac'] * n)} "
            f"(resp_len={round(om['rlmath/truncated_by_response_len_frac'] * n)}, "
            f"turn={round(om['rlmath/truncated_by_turn_frac'] * n)}) "
            f"filtered={round(om['rlmath/overlong_filtered_frac'] * n)} "
            f"loss_trajs={int(om['rlmath/loss_trajectories'])} loss_tokens={int(om['rlmath/loss_tokens'])} "
            f"truncated_turn_frac={om['rlmath/truncated_turn_frac']:.3f}",
            flush=True,
        )
        if om["rlmath/actor_update_skipped"]:
            print(
                f"[RLMathTrainer][overlong] global_step={self.global_steps}: no loss tokens left after filtering "
                "-> SKIPPING actor update this step (weights unchanged)",
                flush=True,
            )
            return batch
        if not enabled or all(keep):
            return super()._fit_update_actor(batch)

        import torch

        keep_t = torch.tensor(keep, dtype=resp_mask.dtype, device=resp_mask.device).unsqueeze(-1)
        new_mask = resp_mask * keep_t
        # With several mini-batches per step (require_batches > 1) a mini-batch made only of filtered rows would have
        # zero loss tokens -> 0/0 -> NaN. Like _rlmath_no_empty_rows: each filtered row keeps its first response
        # token as a loss token with advantage 0 (no gradient contribution, denominator stays > 0). Restored below.
        dropped = new_mask.sum(-1) == 0
        adv_saved = {}
        if bool(dropped.any()):
            new_mask[dropped, 0] = 1
            for k in ("advantages", "returns"):
                if k in batch.batch.keys():
                    adv_saved[k] = batch.batch[k]
                    z = batch.batch[k].clone()
                    z[dropped] = 0
                    batch.batch[k] = z
        batch.batch["response_mask"] = new_mask
        try:
            return super()._fit_update_actor(batch)
        finally:
            batch.batch["response_mask"] = resp_mask
            for k, v in adv_saved.items():
                batch.batch[k] = v

    def _collect_metrics_from_samples(self, batch, metrics):
        super()._collect_metrics_from_samples(batch, metrics)
        try:  # evidence line (c)/(d)
            ntb = batch.non_tensor_batch
            vmin = np.array(_as_int_list(ntb.get("min_global_steps")))
            vmax = np.array(_as_int_list(ntb.get("max_global_steps")))
            cur = self.current_param_version
            lag = cur - vmin if len(vmin) else np.array([0])
            span = vmax - vmin if len(vmax) == len(vmin) and len(vmin) else np.array([0])
            resumes = np.array(_as_int_list(ntb.get("num_resumes")) or [0])
            status = [str(s) for s in (ntb.get("harbor_status") if ntb.get("harbor_status") is not None else [])]
            scores = batch.batch["rm_scores"].sum(-1) if "rm_scores" in batch.batch.keys() else None
            extra = {
                "rlmath_async/version_lag_max": float(lag.max()),
                "rlmath_async/version_lag_mean": float(lag.mean()),
                "rlmath_async/stale_frac": float((lag >= 1).mean()),
                "rlmath_async/multi_version_frac": float((span > 0).mean()),
                "rlmath_async/resumed_turns_sum": float(resumes.sum()),
                "rlmath_async/harbor_completed_frac": float(np.mean([s == "completed" for s in status]))
                if status
                else 0.0,
            }
            metrics.update(extra)
            print(
                f"[RLMathTrainer][consume] global_step={self.global_steps} param_version={cur} "
                f"trajs={len(batch)} traj_versions=[{vmin.min() if len(vmin) else None},"
                f"{vmax.max() if len(vmax) else None}] lag_max={int(lag.max())} "
                f"stale={int((lag >= 1).sum())} multi_version={int((span > 0).sum())} "
                f"resumed_turns={int(resumes.sum())} "
                f"score_mean={float(scores.mean()) if scores is not None else float('nan'):.4f} "
                f"harbor_completed={extra['rlmath_async/harbor_completed_frac']:.2f}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[RLMathTrainer][consume] log failed: {e}", flush=True)


RLMathTrainer = ray.remote(num_cpus=10)(_RLMathTrainer)


# --------------------------------------------------------------------------- task runner
class _RLMathAsyncTaskRunner(_RunnerBase):
    def run(self, config):
        # Harbor task dirs -> parquet, rewrites data.train_files/val_files BEFORE the
        # rollouter builds its datasets (fully_async_rollouter.py:452-465). Our version keys the
        # cache on the task list and stores real task paths (see harbor_data.py).
        from recipe.rlmath.harbor_data import materialize_harbor_datasets

        materialize_harbor_datasets(config)
        return super().run(config)

    def _create_trainer(self, config) -> None:
        # == FullyAsyncTaskRunner._create_trainer (fully_async_main.py:138-157) with our class
        print("[RLMATH ASYNC] creating RLMathTrainer...", flush=True)
        trainer_role_mapping = {
            role: worker_cls for role, worker_cls in self.components["role_worker_mapping"].items() if role != Role.Rollout
        }
        trainer = RLMathTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            device_name=config.trainer.device,
        )
        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer

    def _create_rollouter(self, config) -> None:
        # == FullyAsyncTaskRunner._create_rollouter (fully_async_main.py:117-136) with our class,
        # then start the proxy on the ROLLOUTER's load balancer. This runs before the initial
        # _fit_update_weights and val_before_train (fully_async_main.py:109-113), both of which
        # may already route agent calls through the proxy.
        print("[RLMATH ASYNC] creating RLMathRollouter...", flush=True)
        rollouter = RLMathRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )
        if "hybrid_worker_group" in self.components:
            ray.get(rollouter.set_hybrid_worker_group.remote(self.components["hybrid_worker_group"]))
        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())
        self.components["rollouter"] = rollouter

        from recipe.rlmath.async_proxy import start_async_proxy_server

        proxy_cfg = config.get("proxy_server", {}) or {}
        load_balancer = ray.get(rollouter.get_global_load_balancer.remote())
        proxy_url = start_async_proxy_server(
            load_balancer=load_balancer,
            model_path=config.actor_rollout_ref.model.path,
            host=proxy_cfg.get("host", "0.0.0.0"),
            port=proxy_cfg.get("port", 0),
            tool_format=proxy_cfg.get("tool_format", "hermes"),
            debug=proxy_cfg.get("debug", False),
            session_dump_dir=proxy_cfg.get("session_dump_dir"),
            partial_rollout=bool(config.async_training.partial_rollout),
            max_resumes=int(OmegaConf.select(config, "rlmath_async.max_resumes_per_turn", default=32)),
        )
        self.components["proxy_url"] = proxy_url
        print(f"[RLMATH ASYNC] rollouter ready, proxy at {proxy_url}", flush=True)


RLMathAsyncTaskRunner = ray.remote(num_cpus=1)(_RLMathAsyncTaskRunner)
