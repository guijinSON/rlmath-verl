"""RemoteAgentLoop adapted for verl's fully-async trainer.

Registered via recipe/rlmath/config/async-agent.yaml under the name
``rlmath_remote_agent_async``. Differences from the recipe's RemoteAgentLoop:

* fills ``extra_fields["min_global_steps"/"max_global_steps"]`` with the weight
  versions that actually generated this trajectory (read from the patched proxy,
  see async_proxy.py). The fully-async batch assembler does arithmetic on these
  (verl detach_utils.py:151-161) and the trainer's staleness metric reads them
  (fully_async_trainer.py:756); the stock loop leaves them None -> TypeError.
* keeps per-turn ``completion_logprobs`` aligned with ``completion_token_ids``
  before the recipe reconstructs the trajectory (a length mismatch would shift
  every later logprob).
* carries per-call truncation from the proxy: ``extra_fields["truncated_turns"]`` (LLM calls that
  hit the per-call max_tokens, async_proxy.classify_turn) and ``extra_fields["llm_turns"]``; the
  trainer's overlong filter (overlong.py) uses them.
* resets the proxy's per-session book when the recipe retries a trial, so a failed attempt's
  turns never count against the retry.
* (tito.py) rebuilds the trained sequence with the SAME gap tokens the patched proxy used to
  prompt turn k>=1, so the trained context equals the sampled context token for token (the stock
  loop trains turn k with the earlier <think> reasoning in context although the chat template had
  stripped it from the rollout prompt).
* optional per-process cap on concurrent Harbor trials (``RLMATH_TRIALS_PER_WORKER``; unset = no cap).
  Needed on hosts with a small cgroup ``pids.max`` (KT: 8192 threads; one trial costs ~50 threads
  across harbor + docker CLI), where validation launches every val task at once and the
  rollouter's max_concurrent_prompts does not apply.
* reports the trial wall time as ``metrics.generate_sequences`` (the recipe puts
  a plain dict there, which becomes all-zero AgentLoopMetrics) and the Harbor
  status as ``extra_fields["harbor_status"]``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from recipe.agentic.agent_loop.remote_agent_loop import RemoteAgentLoop
from recipe.rlmath import tito

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("RLMATH_ASYNC_LOG_LEVEL", "INFO"))

_proxy_actor = None
_TRIALS_PER_WORKER = int(os.getenv("RLMATH_TRIALS_PER_WORKER", "0") or 0)
_trial_sem = None  # created lazily on the worker's event loop

# Circuit breaker (per AgentLoopWorker process): when the sandbox host is down every new trial fails within
# seconds, and those fast failures flood the rollout queue (MLXP 10-05: one batch was mostly empty failures).
# After RLMATH_BREAKER_FAILS infra failures within RLMATH_BREAKER_WINDOW s, new trials wait until
# `docker info` against $DOCKER_HOST succeeds again (polled every RLMATH_BREAKER_POLL s). 0 disables it.
_BREAKER_FAILS = int(os.getenv("RLMATH_BREAKER_FAILS", "6") or 0)
_BREAKER_WINDOW = float(os.getenv("RLMATH_BREAKER_WINDOW", "120"))
_BREAKER_POLL = float(os.getenv("RLMATH_BREAKER_POLL", "30"))
_infra_fail_times: list[float] = []
_breaker_lock = None  # asyncio.Lock, created lazily on the worker's event loop
_breaker_open_since: float | None = None


async def _docker_ok() -> bool:
    import asyncio

    try:
        p = await asyncio.create_subprocess_exec(
            "docker", "info", "--format", "{{.ServerVersion}}",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        return await asyncio.wait_for(p.wait(), timeout=30) == 0
    except Exception:  # noqa: BLE001 - timeout / missing binary = not ok
        try:
            p.kill()  # type: ignore[possibly-undefined]
        except Exception:  # noqa: BLE001
            pass
        return False


# Fail fast when the policy's own inference server is dead (vLLM EngineDeadError -> proxy 500 on every call): the
# run cannot recover from that, and without this it keeps launching trials that fail in seconds until a batch of
# empty trajectories finally crashes the trainer (MLXP 10-06: 3 h lost). Abort when RLMATH_LLM_FAIL_ABORT such
# failures happen within RLMATH_LLM_FAIL_WINDOW s on this worker with no trial completing in that window.
_LLM_FAIL_ABORT = int(os.getenv("RLMATH_LLM_FAIL_ABORT", "20") or 0)
_LLM_FAIL_WINDOW = float(os.getenv("RLMATH_LLM_FAIL_WINDOW", "300"))
_llm_fail_times: list[float] = []
_last_completed_at = time.time()


def _note_llm_failure(error) -> None:
    now = time.time()
    _llm_fail_times.append(now)
    while _llm_fail_times and now - _llm_fail_times[0] > _LLM_FAIL_WINDOW:
        _llm_fail_times.pop(0)
    if _LLM_FAIL_ABORT > 0 and len(_llm_fail_times) >= _LLM_FAIL_ABORT and now - _last_completed_at > _LLM_FAIL_WINDOW:
        msg = (
            f"[rlmath-async] ABORT: {len(_llm_fail_times)} trials failed on the policy inference server within "
            f"{_LLM_FAIL_WINDOW:.0f}s and none completed (vLLM engine dead?). Last error: {str(error)[:400]}"
        )
        logger.error(msg)
        raise RuntimeError(msg)


def _note_infra_failure() -> None:
    now = time.time()
    _infra_fail_times.append(now)
    while _infra_fail_times and now - _infra_fail_times[0] > _BREAKER_WINDOW:
        _infra_fail_times.pop(0)


async def _breaker_wait() -> None:
    """Block new trials while the breaker is open; the first waiter polls docker, the rest wait on the lock."""
    global _breaker_lock, _breaker_open_since
    if _BREAKER_FAILS <= 0:
        return
    import asyncio

    now = time.time()
    recent = [t for t in _infra_fail_times if now - t <= _BREAKER_WINDOW]
    if len(recent) < _BREAKER_FAILS and _breaker_open_since is None:
        return
    if _breaker_lock is None:
        _breaker_lock = asyncio.Lock()
    async with _breaker_lock:
        if _breaker_open_since is None and len([t for t in _infra_fail_times if time.time() - t <= _BREAKER_WINDOW]) < _BREAKER_FAILS:
            return  # another waiter already closed it
        if _breaker_open_since is None:
            _breaker_open_since = time.time()
            logger.error(
                "[rlmath-async][breaker] OPEN: %d infra failures in %.0fs; holding new trials until docker answers",
                len(recent), _BREAKER_WINDOW,
            )
        while not await _docker_ok():
            await asyncio.sleep(_BREAKER_POLL)
        logger.error(
            "[rlmath-async][breaker] CLOSED after %.0fs: docker is answering again", time.time() - _breaker_open_since
        )
        _breaker_open_since = None
        _infra_fail_times.clear()


def _trials_limit() -> int:
    """Per-process trial cap: `trials_per_worker` from RLMATH_LIMITS_FILE (live, see limits.py), else the env value."""
    from recipe.rlmath.limits import read_limits

    return read_limits().get("trials_per_worker", _TRIALS_PER_WORKER)


def _get_trial_sem():
    global _trial_sem
    if _trial_sem is None:
        from recipe.rlmath.limits import LIMITS_FILE, ResizableLimiter

        if _TRIALS_PER_WORKER > 0 or LIMITS_FILE:
            _trial_sem = ResizableLimiter(_trials_limit)
    return _trial_sem


def _get_proxy_actor():
    global _proxy_actor
    if _proxy_actor is None:
        import ray
        from recipe.agentic.proxyserver.ray_actor import PROXY_ACTOR_NAME

        _proxy_actor = ray.get_actor(PROXY_ACTOR_NAME)
    return _proxy_actor


def _align_turn_logprobs(session: Any) -> int:
    fixed = 0
    for turn in getattr(session, "turns", None) or []:
        ids = turn.completion_token_ids or []
        lps = turn.completion_logprobs or []
        if ids and len(lps) != len(ids):
            turn.completion_logprobs = (list(lps) + [0.0] * len(ids))[: len(ids)]
            fixed += 1
    return fixed


class AsyncRemoteAgentLoop(RemoteAgentLoop):
    """One instance per trajectory (AgentLoopWorker._run_agent_loop instantiates per sample)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.proxy_server_url:
            raise RuntimeError("rlmath async: standalone proxy mode (PROXY_SERVER_URL) is not supported")
        self._rlmath_versions: dict | None = None

    def _get_proxy_url(self) -> str:
        url = super()._get_proxy_url()
        # Multi-node: the agent (in this AgentLoopWorker process) may sit on a different node than the proxy
        # actor, so the base class's LLM_PROXY_IP=127.0.0.1 would miss it. The proxy URL already carries the
        # proxy node's Ray IP (ib0 on JUPITER), reachable from every node -> point the agent there.
        if os.environ.get("RLMATH_PROXY_IP_FROM_URL") == "1":
            from urllib.parse import urlparse

            os.environ["LLM_PROXY_IP"] = urlparse(url).hostname
        return url

    async def _get_session_data(self, proxy_url: str, trial_id: str):
        session = await super()._get_session_data(proxy_url, trial_id)
        try:
            self._rlmath_versions = await _get_proxy_actor().pop_session_versions.remote(trial_id)
        except Exception as e:  # noqa: BLE001 - never fail a finished trial on bookkeeping
            logger.warning("[rlmath-async] could not fetch versions for %s: %s", trial_id, e)
            self._rlmath_versions = None
        if session is not None:
            n = _align_turn_logprobs(session)
            if n:
                logger.warning("[rlmath-async] %s: aligned logprobs of %d turn(s)", trial_id, n)
        return session

    # transformers >= 5 returns a BatchEncoding (dict) from apply_chat_template(tokenize=True) instead of a list
    # of ids; the recipe's tokenizer-path helpers pass that through as prompt_ids (pydantic ValidationError) and
    # compare slices of it. Normalize to plain lists. Processor paths (multimodal) are left to the recipe.
    @staticmethod
    def _as_ids(x) -> list[int]:
        if hasattr(x, "keys"):
            x = x["input_ids"]
        if hasattr(x, "tolist"):
            x = x.tolist()
        if x and isinstance(x[0], list):  # batched [[...]]
            x = x[0]
        return list(x)

    def _chat_ids(self, messages, add_generation_prompt: bool) -> list[int]:
        return self._as_ids(
            self.tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=True)
        )

    def _tokenize_messages(self, messages):
        if self.processor is not None:
            return self._as_ids(super()._tokenize_messages(messages))
        return self._chat_ids(self._normalize_messages(messages), add_generation_prompt=True)

    def _get_system_prompt_ids(self) -> list[int]:
        if hasattr(self, "_cached_sys_ids"):
            return self._cached_sys_ids
        if self.processor is not None:
            self._cached_sys_ids = self._as_ids(super()._get_system_prompt_ids())
            return self._cached_sys_ids
        try:
            self._cached_sys_ids = self._chat_ids([], add_generation_prompt=False)
        except Exception:  # noqa: BLE001 - e.g. Qwen3.5/3.6 templates raise on an empty conversation
            self._cached_sys_ids = []
        return self._cached_sys_ids

    def _tokenize_tool_messages(self, messages):
        if self.processor is not None:
            return self._as_ids(super()._tokenize_tool_messages(messages))
        if not messages:
            return []
        messages = self._normalize_messages(messages)
        prefix_msgs = [] if any(m.get("role") == "user" for m in messages) else [{"role": "user", "content": "x"}]
        ids = self._chat_ids(prefix_msgs + messages, add_generation_prompt=True)
        if prefix_msgs:
            pfx_ids = self._chat_ids(prefix_msgs, add_generation_prompt=True)
            if ids[: len(pfx_ids)] == pfx_ids:
                ids = ids[len(pfx_ids):]
        sys_ids = self._get_system_prompt_ids()
        if sys_ids and ids[: len(sys_ids)] == sys_ids:
            ids = ids[len(sys_ids):]
        return ids

    async def _proxy_request(self, method, url, *, max_retries=5, base_delay=1.0, max_delay=30.0):
        """Stock retry loop with a longer per-request timeout (RLMATH_PROXY_TIMEOUT, default 120 s instead
        of 30 s): under load the proxy's event loop can take >30 s to answer, and the client then retries a
        POST that already succeeded (-> 409 on /sessions/{id}, which killed the rollouter on KT)."""
        import asyncio

        import aiohttp

        timeout = aiohttp.ClientTimeout(total=float(os.getenv("RLMATH_PROXY_TIMEOUT", "120")))
        last_error = None
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(method, url, timeout=timeout) as resp:
                        body = await resp.json() if resp.content_type == "application/json" else await resp.text()
                        if resp.status < 500:
                            return resp.status, body
                        last_error = RuntimeError(f"Proxy returned {resp.status}: {str(body)[:200]}")
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                last_error = e
            if attempt + 1 < max_retries:
                delay = min(base_delay * (2**attempt), max_delay)
                logger.warning(
                    "[rlmath-async] proxy %s %s failed (attempt %d/%d): %r; retrying in %.1fs",
                    method, url, attempt + 1, max_retries, last_error, delay,
                )
                await asyncio.sleep(delay)
        raise RuntimeError(f"Proxy request {method} {url} failed after {max_retries} attempts: {last_error!r}")

    # Idempotent session calls: a retried request whose first attempt succeeded server-side returns
    # 409 (already registered / already completed) or 404 (already deleted); that is success, not an error.
    async def _register_session(self, proxy_url: str, trial_id: str) -> None:
        status, _ = await self._proxy_request("POST", f"{proxy_url}/sessions/{trial_id}")
        if status == 409:
            logger.warning("[rlmath-async] session %s already registered (retried POST); continuing", trial_id)
            return
        if status >= 400:
            raise RuntimeError(f"Failed to register session {trial_id}: HTTP {status}")

    async def _complete_session(self, proxy_url: str, trial_id: str) -> None:
        status, _ = await self._proxy_request("POST", f"{proxy_url}/sessions/{trial_id}/complete")
        if status in (404, 409):
            logger.warning("[rlmath-async] complete %s -> HTTP %d (already done); continuing", trial_id, status)
            return
        if status >= 400:
            raise RuntimeError(f"Failed to complete session {trial_id}: HTTP {status}")

    async def _delete_session(self, proxy_url: str, trial_id: str) -> None:
        status, _ = await self._proxy_request("DELETE", f"{proxy_url}/sessions/{trial_id}")
        if status == 404:
            return
        if status >= 400:
            raise RuntimeError(f"Failed to delete session {trial_id}: HTTP {status}")

    async def _reset_session(self, proxy_url: str, trial_id: str) -> None:
        # book first: bumping the session epoch makes the proxy drop completions of the abandoned
        # attempt that are still generating, before the recorder is cleared for the retry
        try:
            await _get_proxy_actor().reset_session_book.remote(trial_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("[rlmath-async] could not reset book for %s: %s", trial_id, e)
        await super()._reset_session(proxy_url, trial_id)

    def _reconstruct_output(self, session, initial_messages, worker_cache=None) -> AgentLoopOutput:
        if tito.ENABLED:
            out = self._reconstruct_output_tito(session, initial_messages, worker_cache)
        else:
            out = super()._reconstruct_output(session, initial_messages, worker_cache)
        # Backstop: a prompt longer than rollout.prompt_length makes verl's _postprocess torch.cat fail
        # and kills the whole rollouter. Keep the trajectory (its reward still counts in the group) but
        # train on none of its tokens.
        if len(out.prompt_ids) > self.prompt_length:
            logger.error(
                "[rlmath-async] %s: prompt of %d tokens > prompt_length %d (stale/garbled recording?); "
                "masking the whole trajectory",
                getattr(session, "session_id", "?"),
                len(out.prompt_ids),
                self.prompt_length,
            )
            eos = getattr(self.tokenizer, "eos_token_id", None) or 0
            out.prompt_ids = list(out.prompt_ids[: self.prompt_length])
            out.response_ids = [eos]
            out.response_mask = [0]
            out.response_logprobs = [0.0] if out.response_logprobs is not None else None
        # tool-usage summary of this trajectory (rlmath/tools/* in W&B): commands sent, libraries imported, executables run
        try:
            from recipe.rlmath.tool_metrics import summarize_trajectory

            texts = []
            for turn in session.turns:
                t = turn.completion_text
                if not t and getattr(turn, "completion_token_ids", None):
                    t = self.tokenizer.decode(list(turn.completion_token_ids), skip_special_tokens=True)
                texts.append(t)
            self._rlmath_tools = summarize_trajectory(texts)
        except Exception as e:  # noqa: BLE001 - diagnostics only
            logger.warning("[rlmath-async] tool summary failed: %s", e)
        return out

    def _dead_reply_ids(self) -> list[int]:
        if not hasattr(self, "_cached_dead_ids"):
            from recipe.rlmath.async_proxy import _DEAD_SESSION_REPLY

            ids = self.tokenizer.encode(_DEAD_SESSION_REPLY, add_special_tokens=False)
            eos = getattr(self.tokenizer, "eos_token_id", None)
            self._cached_dead_ids = ids + [eos] if eos is not None else ids
        return self._cached_dead_ids

    def _reconstruct_output_tito(self, session, initial_messages, worker_cache=None) -> AgentLoopOutput:
        from recipe.agentic.proxyserver.vllm_provider import _normalize_messages as proxy_normalize

        prompt_ids = self._tokenize_messages(initial_messages)
        cache = worker_cache or []
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        num_turns = 0
        dead_ids = self._dead_reply_ids()
        for i, turn in enumerate(session.turns):
            if i < len(cache):
                turn_ids, turn_logprobs = list(cache[i]["token_ids"]), list(cache[i]["logprobs"])
            elif turn.completion_token_ids:
                turn_ids, turn_logprobs = list(turn.completion_token_ids), list(turn.completion_logprobs)
            else:
                turn_ids = self.tokenizer.encode(turn.completion_text, add_special_tokens=False)
                turn_logprobs = [0.0] * len(turn_ids)
            response_ids.extend(turn_ids)
            # The proxy's "budget exhausted" reply (async_proxy._DEAD_SESSION_REPLY) was never sampled from the
            # policy (logprob 0 everywhere). It used to be harmless because truncated trajectories were dropped from
            # the loss; with overlong_mode=penalize they stay in, so these tokens must not be trained on.
            not_sampled = turn_ids == dead_ids or (len(turn_ids) >= 8 and all(lp == 0.0 for lp in turn_logprobs))
            response_mask.extend([0 if not_sampled else 1] * len(turn_ids))
            response_logprobs.extend(turn_logprobs)
            num_turns += 1
            if i + 1 < len(session.turns):
                # identical to the proxy: normalize the delta, drop the assistant echo, gap_ids()
                delta = proxy_normalize(session.turns[i + 1].request_messages or [])
                gap_msgs = [m for m in delta if m.get("role") in tito.GAP_ROLES]
                if gap_msgs:
                    gap = tito.gap_ids(self.tokenizer, turn_ids, gap_msgs)
                    response_ids.extend(gap)
                    response_mask.extend([0] * len(gap))
                    response_logprobs.extend([0.0] * len(gap))
                    num_turns += len(gap_msgs)
        n = self.response_length
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:n],
            response_mask=response_mask[:n],
            response_logprobs=response_logprobs[:n] or None,
            num_turns=num_turns,
            metrics={},
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        from recipe.rlmath.overlong import is_infra_failure, is_llm_server_failure

        global _last_completed_at
        await _breaker_wait()
        sem = _get_trial_sem()
        if sem is not None:
            async with sem:
                t0 = time.time()
                self._rlmath_versions = None
                self._rlmath_tools = None
                output = await super().run(sampling_params, **kwargs)
        else:
            t0 = time.time()
            self._rlmath_versions = None
            self._rlmath_tools = None
            output = await super().run(sampling_params, **kwargs)
        wall = time.time() - t0

        status = error = None
        harbor_rewards = {}
        if isinstance(output.metrics, dict):
            status = output.metrics.get("harbor_status")
            error = output.metrics.get("harbor_error")
            harbor_rewards = output.metrics.get("harbor_rewards") or {}
        infra = is_infra_failure(status, error)
        if str(status or "").lower() == "completed":
            _last_completed_at = time.time()
        if infra:
            logger.warning("[rlmath-async] infra-failed trial (excluded from its GRPO group): %s", str(error)[:300])
            if is_llm_server_failure(status, error):
                _note_llm_failure(error)  # may raise -> aborts the run
            else:
                _note_infra_failure()
        output.metrics = AgentLoopMetrics(generate_sequences=wall, tool_calls=0.0)
        output.extra_fields["infra_failed"] = int(infra)

        v = self._rlmath_versions or {}
        vmin, vmax = v.get("min"), v.get("max")
        if vmin is None or vmax is None:
            # no LLM tokens recorded for this trial (agent never called the proxy):
            # stamp it with the newest version the proxy has seen so it is not "stale".
            latest = None
            try:
                latest = await _get_proxy_actor().get_latest_version.remote()
            except Exception:  # noqa: BLE001
                pass
            vmin = vmax = latest if latest is not None else 0

        output.extra_fields["min_global_steps"] = int(vmin)
        output.extra_fields["max_global_steps"] = int(vmax)
        output.extra_fields["num_resumes"] = int(v.get("aborts", 0) or 0)
        output.extra_fields["truncated_turns"] = int(v.get("truncated", 0) or 0)
        output.extra_fields["no_eos_turns"] = int(v.get("no_eos", 0) or 0)
        output.extra_fields["llm_turns"] = int(v.get("turns", 0) or 0)
        output.extra_fields["harbor_status"] = str(status or "unknown")
        from recipe.rlmath.tool_metrics import task_kind

        n_cmds, libs, exes = getattr(self, "_rlmath_tools", None) or (0, "", "")
        output.extra_fields["tool_cmds"] = int(n_cmds)
        output.extra_fields["tool_libs"] = str(libs)
        output.extra_fields["tool_exes"] = str(exes)
        output.extra_fields["task_kind"] = task_kind(kwargs.get("instance_id"))

        # optimize beat-best flags -> reward_extra_info (val-aux/<source>/opt_*/mean@N; rlmath/optimize/* in training).
        # verl stacks reward_extra_info with the FIRST trajectory's keys, so every trajectory must carry them, and a
        # None reward_score would send this one to the reward loop, which replaces reward_extra_info. Harbor always
        # returns a reward (or an error -> 0.0), so None only happens if a verifier wrote nothing; score it 0.0.
        from recipe.rlmath.optimize_beat import optimize_flags

        if output.reward_score is None:
            logger.warning("[rlmath-async] trial returned no reward; scoring it 0.0")
            output.reward_score = 0.0
        output.extra_fields["reward_extra_info"] = optimize_flags(harbor_rewards)
        return output
