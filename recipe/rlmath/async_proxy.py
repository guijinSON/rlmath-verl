"""Fully-async variant of the recipe's LLM proxy (recipe.agentic.proxyserver).

Why this exists
---------------
The recipe proxy's ``VLLMRayProvider._generate`` calls ``server.generate`` on a vLLM
server handle *directly* (recipe/agentic/proxyserver/vllm_provider.py:195-202). It
therefore bypasses verl's ``FullyAsyncLLMServerClient`` (the client that re-submits
an aborted request, verl fully_async_rollouter.py:52-151).  In fully-async training
every parameter sync starts with ``CheckpointEngineManager.abort_replicas()``
(verl checkpoint_engine/base.py:482-483), so every in-flight agent turn would come
back with ``stop_reason="aborted"`` and a *truncated* completion, which the recipe
provider would hand to the agent as if it were a finished answer (Qwen3-Thinking
cut mid-``<think>`` -> Terminus parse error, corrupted trajectory).

This module:
  1. monkeypatches ``VLLMRayProvider._generate`` (inside the proxy actor process
     only) with a resumable version: on ``aborted`` it re-submits
     ``prompt_ids + tokens_so_far`` with the remaining ``max_tokens`` (true partial
     rollout, token-exact, mirrors FullyAsyncLLMServerClient.generate) or, with
     ``partial_rollout=False``, regenerates the turn from scratch.  Either way the
     agent's HTTP call only sees one complete response.
  2. records, per session (= trial_id), per-call truncation (hit the per-call max_tokens,
     see ``classify_turn``) and the min/max weight version
     (``extra_fields["global_steps"]`` set by vllm_async_server.py:563 /
     vllm_rollout.py:197) of every token generated, so the agent loop can fill the
     ``min_global_steps``/``max_global_steps`` fields the fully-async trainer needs
     (detach_utils.py:151-161, fully_async_trainer.py:756).
  3. awaits the Ray ObjectRef directly instead of ``asyncio.to_thread(ray.get)``
     (vllm_provider.py:202), which silently capped concurrent LLM calls at the
     default thread-pool size (min(32, ncpu+4)).
  4. starts a Ray named actor under the recipe's ``PROXY_ACTOR_NAME`` so the stock
     ``RemoteAgentLoop._get_proxy_url`` discovers it unchanged.

No recipe file is modified.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("RLMATH_ASYNC_LOG_LEVEL", "INFO"))

_PATCH_FLAG = "_rlmath_async_patched"
# Total response budget per trajectory (= data.max_response_length, exported by the launcher). The trainer cuts
# every trajectory at this length and the overlong filter then drops it from the loss, so tokens generated past
# it carry no signal. With RLMATH_STOP_AFTER_TRUNC=1 (default when the overlong filter is on) a session that
# already had a truncated call is also dead: it will be filtered whatever happens next.
_RESPONSE_BUDGET = int(os.getenv("RLMATH_RESPONSE_BUDGET", "0") or 0)
_STOP_AFTER_TRUNC = os.getenv("RLMATH_STOP_AFTER_TRUNC", "1") == "1"
# Validation trials are never stopped after a truncated call: stopping saves training compute (the trajectory is
# filtered / scored 0 anyway), but in validation it turns one over-long call into a guaranteed 0 (10-07: 117-119
# of 128 val tasks died that way). Session ids are "<task dir name>-<22-char suffix>"; RLMATH_NO_STOP_TASKS_DIR
# (the val dir) lists the task names exempt from the stop. The total response budget still applies.
_NO_STOP_TASKS: frozenset[str] = frozenset(
    os.listdir(os.environ["RLMATH_NO_STOP_TASKS_DIR"]) if os.path.isdir(os.getenv("RLMATH_NO_STOP_TASKS_DIR", "")) else ()
)


def _stop_after_trunc_for(session_id: str) -> bool:
    return _STOP_AFTER_TRUNC and session_id.rsplit("-", 1)[0] not in _NO_STOP_TASKS


# Terminus-2 JSON turn that ends the episode (see the dead-session short-circuit in _generate_resumable)
_DEAD_SESSION_REPLY = (
    '{"analysis": "Response budget for this episode is exhausted.", "plan": "Stop.", '
    '"commands": [], "task_complete": true}'
)
_EXPECTED_GENERATE_PARAMS = [
    "self",
    "messages",
    "session_id",
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "tools",
    "repetition_penalty",
]


def _is_prompt_too_long(err_msg: str) -> bool:
    # recipe matches "Prompt length ... exceeds"; verl@9f18abe5 vllm_async_server.py:479-483
    # raises "Prompt length (N) leaves no room to generate ..." -> match both.
    return "Prompt length" in err_msg and ("exceeds" in err_msg or "no room" in err_msg)


class _VersionBook:
    """Per-session weight-version bookkeeping (lives on the provider instance).

    All access happens on the proxy actor's single asyncio loop (uvicorn is served
    from ``asyncio.create_task`` in LLMProxyServer.start, server.py:570-580), so no lock.
    """

    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}
        self.latest_version: int | None = None
        self.total_aborts = 0
        self.total_resumed_turns = 0
        self.total_restarted_turns = 0
        self.total_turns = 0
        self.total_truncated_turns = 0
        self.total_no_eos_turns = 0
        self.tito_turns = 0
        self.tito_fallbacks = 0
        self.tito: dict[str, tuple] = {}  # session -> (n_msgs, digest, prompt_ids, completion_ids), see tito.py
        # bumped on every trial retry: a request that started before the reset belongs to the
        # abandoned attempt and must not be recorded into the fresh session (it would become the
        # retry's "turn 0" carrying the old attempt's whole history -> 84k-token prompt crash)
        self.epochs: dict[str, int] = {}
        self.total_stale_dropped = 0
        self.init_len: dict[str, int] = {}  # session -> token length of its first prompt (= the trained prompt)
        self.total_short_circuit = 0

    def note_turn(
        self,
        session_id: str,
        vmin: int | None,
        vmax: int | None,
        n_aborts: int,
        truncated: bool = False,
        no_eos: bool = False,
    ) -> None:
        rec = self.sessions.setdefault(
            session_id, {"min": None, "max": None, "turns": 0, "aborts": 0, "truncated": 0, "no_eos": 0}
        )
        if vmin is not None:
            rec["min"] = vmin if rec["min"] is None else min(rec["min"], vmin)
        if vmax is not None:
            rec["max"] = vmax if rec["max"] is None else max(rec["max"], vmax)
            self.latest_version = vmax if self.latest_version is None else max(self.latest_version, vmax)
        rec["turns"] += 1
        rec["aborts"] += n_aborts
        rec["truncated"] += int(truncated)
        rec["no_eos"] += int(no_eos)
        self.total_turns += 1
        self.total_truncated_turns += int(truncated)
        self.total_no_eos_turns += int(no_eos)

    def reset_session(self, session_id: str) -> None:
        """Called when RemoteAgentLoop retries a trial: the retry's trajectory replaces the
        failed attempt's recording, so its versions/truncation counts must not leak into it."""
        self.sessions.pop(session_id, None)
        self.tito.pop(session_id, None)
        self.init_len.pop(session_id, None)
        self.epochs[session_id] = self.epochs.get(session_id, 0) + 1


def _eos_ids(tokenizer) -> set[int]:
    """Token ids that end a naturally finished Qwen3 generation (tokenizer eos + chat end tokens)."""
    ids: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    unk = getattr(tokenizer, "unk_token_id", None)
    for tok in ("<|im_end|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(tid, int) and tid >= 0 and tid != unk:
            ids.add(tid)
    return ids


def classify_turn(token_ids: list[int], max_tokens: int, stop_reason: str | None, eos_ids: set[int]) -> tuple[str | None, bool]:
    """Return (truncation_reason | None, no_eos) for one finished LLM call.

    * "length"  - the call used its whole per-call ``max_tokens`` budget without ending on an EOS
                  token, i.e. vLLM stopped it for length. vLLM reports both stop and length as
                  stop_reason="completed" (vllm_async_server.py:582-583), so the budget is the
                  signal. Tokens from aborted (partial) chunks count toward the budget, but the
                  budget can only be exhausted by the *final* generation of a resumed turn, so a
                  resumed turn is truncated only if that final generation hit it.
    * "gave_up" - more than ``max_resumes`` aborts; the turn was returned unfinished.
    * no_eos    - diagnostic only (not a truncation): finished below budget without an EOS token
                  (stop string, or vLLM clamped max_tokens to the remaining context).
    """
    if stop_reason == "aborted":
        return "gave_up", False
    ends_eos = bool(token_ids) and token_ids[-1] in eos_ids
    if len(token_ids) >= max_tokens and not ends_eos:
        return "length", False
    return None, not ends_eos


def install_provider_patch(partial_rollout: bool = True, max_resumes: int = 32, transient_retries: int = 8) -> None:
    """Patch ``VLLMRayProvider`` in *this* process. Idempotent."""
    from recipe.agentic.proxyserver import vllm_provider as vp

    cls = vp.VLLMRayProvider
    if getattr(cls, _PATCH_FLAG, False):
        return

    from recipe.rlmath.tito import ENABLED as tito, gap_ids, msgs_digest, split_delta

    params = list(inspect.signature(cls._generate).parameters)
    if params != _EXPECTED_GENERATE_PARAMS:
        raise RuntimeError(
            "recipe VLLMRayProvider._generate signature changed "
            f"({params} != {_EXPECTED_GENERATE_PARAMS}); re-check async_proxy.py before running"
        )

    orig_release = cls.release_session

    async def _generate_resumable(
        self,
        messages: list[dict[str, Any]],
        session_id: str | None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
        tools: list[dict] | None = None,
        repetition_penalty: float | None = None,
    ):
        """Drop-in replacement for VLLMRayProvider._generate (same return tuple)."""
        book: _VersionBook = self.__dict__.setdefault("_rlmath_book", _VersionBook())

        normalized = vp._normalize_messages(messages)
        tools = self._sanitize_tools(tools)

        # Token-exact continuation (see tito.py): turn k>=1 is prompted with exactly the
        # tokens the trainer will see (previous prompt + previous completion incl. its
        # <think> reasoning + gap), not a re-rendered chat template that drops reasoning.
        tito_state: dict = book.tito
        sticky_key_early = session_id or "default"
        epoch0 = book.epochs.get(sticky_key_early, 0)
        prompt_ids: list[int] | None = None
        st = tito_state.get(session_id) if (session_id and tito) else None
        if st is not None and not tools:
            prev_n, prev_digest, prev_prompt, prev_completion = st
            gap_msgs = split_delta(prev_n, normalized)
            if gap_msgs is not None and msgs_digest(normalized[:prev_n]) == prev_digest:
                prompt_ids = prev_prompt + prev_completion + gap_ids(self.tokenizer, prev_completion, gap_msgs)
                book.tito_turns += 1
            else:
                book.tito_fallbacks += 1
                if book.tito_fallbacks <= 20 or book.tito_fallbacks % 100 == 0:
                    print(
                        f"[rlmath-async][tito] session={session_id} not an append-only continuation "
                        f"(prev_n={prev_n}, now={len(normalized)}); falling back to chat template "
                        f"(fallbacks={book.tito_fallbacks}/{book.tito_turns + book.tito_fallbacks})",
                        flush=True,
                    )
        if prompt_ids is None:
            encoded = self.tokenizer.apply_chat_template(normalized, tools=tools, add_generation_prompt=True, tokenize=True)
            prompt_ids = list(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)

        # Enforce the trajectory's total response budget here instead of letting the agent run past it.
        # With TITO the prompt of turn k is exactly prompt_0 + everything the trainer will see as response so far
        # (completions + tool/user gaps), so len(prompt) - len(prompt_0) is the response length used.
        if session_id and _RESPONSE_BUDGET > 0:
            init = book.init_len.setdefault(session_id, len(prompt_ids))
            left = _RESPONSE_BUDGET - max(0, len(prompt_ids) - init)
            already_trunc = (book.sessions.get(session_id) or {}).get("truncated", 0) > 0
            if left <= 0 or (already_trunc and _stop_after_trunc_for(session_id)):
                book.total_short_circuit += 1
                k = book.total_short_circuit
                if k <= 20 or k % 200 == 0:
                    print(
                        f"[rlmath-async][budget] session={session_id} dead ("
                        f"{'budget exhausted' if left <= 0 else 'already truncated'}, used={_RESPONSE_BUDGET - left}"
                        f"/{_RESPONSE_BUDGET}); returning empty length-stop (total_short_circuit={k})",
                        flush=True,
                    )
                book.note_turn(session_id, None, None, 0, truncated=True)
                # Do NOT answer with an empty finish_reason=length: Terminus retries output-length errors
                # recursively inside the same turn -> instant length stops -> RecursionError -> trial error +
                # full retry. Answer with a well-formed "task complete" turn instead so the agent ends the
                # episode (and confirms on the follow-up question). The trajectory is already marked truncated
                # (overlong filter drops it from the loss); its tokens get logprob 0 and never train.
                text = _DEAD_SESSION_REPLY
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                eos = getattr(self.tokenizer, "eos_token_id", None)
                if eos is not None:
                    ids = ids + [eos]
                return ids, [0.0] * len(ids), "stop", text, None
            max_tokens = min(max_tokens, left)

        base_params: dict[str, Any] = {"temperature": temperature, "top_p": top_p, "logprobs": True}
        if stop:
            base_params["stop"] = stop
        if repetition_penalty is not None:
            base_params["repetition_penalty"] = repetition_penalty

        sticky_key = session_id or "default"
        token_ids: list[int] = []
        log_probs: list[float] = []
        stop_reason: str | None = None
        last_server_id: str | None = None
        turn_vmin: int | None = None
        turn_vmax: int | None = None
        n_aborts = 0
        n_transient = 0
        t0 = time.time()

        while True:
            remaining = max_tokens - len(token_ids)
            if remaining <= 0:
                stop_reason = "completed"  # same label vLLM gives a length stop (vllm_async_server.py:582-583)
                break
            params = dict(base_params)
            params["max_tokens"] = remaining

            server_id = None
            try:
                server_id, server = await self._lb.acquire_server.remote(request_id=sticky_key)
                last_server_id = server_id
                output = await server.generate.remote(
                    prompt_ids=prompt_ids + token_ids,
                    sampling_params=params,
                    request_id=uuid4().hex,
                )
            except Exception as e:  # noqa: BLE001
                err_msg = str(e)
                if _is_prompt_too_long(err_msg):
                    if token_ids:
                        # context filled up during a resumed turn: return what we have
                        stop_reason = "completed"
                        break
                    eos = getattr(self.tokenizer, "eos_token_id", None) or 0
                    logger.warning("[rlmath-async] prompt too long (%d tok) session=%s", len(prompt_ids), sticky_key)
                    # the agent gets finish_reason="length": the context is exhausted -> truncated
                    book.note_turn(sticky_key, None, None, n_aborts, truncated=True)
                    return [eos], [0.0], "length", "", None
                n_transient += 1
                if n_transient <= transient_retries:
                    logger.warning(
                        "[rlmath-async] generate error session=%s (%d/%d), retrying: %s",
                        sticky_key,
                        n_transient,
                        transient_retries,
                        err_msg[:300],
                    )
                    # ~60 s total: covers a request rejected while a weight sync is in progress
                    await asyncio.sleep(min(2.0 * n_transient, 10.0))
                    continue
                raise
            finally:
                if server_id is not None:
                    self._lb.release_server.remote(server_id=server_id)

            new_ids = list(output.token_ids or [])
            new_lps = list(output.log_probs) if output.log_probs else []
            if len(new_lps) != len(new_ids):
                new_lps = (new_lps + [0.0] * len(new_ids))[: len(new_ids)]
            version = (getattr(output, "extra_fields", None) or {}).get("global_steps")
            sr = getattr(output, "stop_reason", None)

            # vLLM always sets finish_reason on a finished request; None means the generator
            # ended without one (interrupted) -> treat like an abort instead of returning a
            # silently truncated turn (vllm_async_server.py:579-585 maps unknown -> as-is).
            if sr in ("aborted", "abort") or sr is None:
                n_aborts += 1
                book.total_aborts += 1
                if partial_rollout:
                    token_ids.extend(new_ids)
                    log_probs.extend(new_lps)
                    if version is not None and new_ids:
                        turn_vmin = version if turn_vmin is None else min(turn_vmin, version)
                        turn_vmax = version if turn_vmax is None else max(turn_vmax, version)
                else:
                    # restart the turn on the new weights: keep each turn single-version
                    token_ids, log_probs = [], []
                    turn_vmin = turn_vmax = None
                if n_aborts > max_resumes:
                    logger.error(
                        "[rlmath-async] session=%s exceeded %d resumes; returning %d tokens as-is",
                        sticky_key,
                        max_resumes,
                        len(token_ids),
                    )
                    stop_reason = "aborted"
                    break
                # vLLM>=0.12 keeps the engine paused until resume_generation(); the next
                # generate() waits for it. Small sleep avoids a hot loop on older vLLM.
                await asyncio.sleep(0.5)
                continue

            token_ids.extend(new_ids)
            log_probs.extend(new_lps)
            if version is not None:
                turn_vmin = version if turn_vmin is None else min(turn_vmin, version)
                turn_vmax = version if turn_vmax is None else max(turn_vmax, version)
            stop_reason = sr
            break

        if n_aborts:
            if partial_rollout:
                book.total_resumed_turns += 1
            else:
                book.total_restarted_turns += 1
            # evidence line for the smoke test (b)/(c): a turn survived a weight sync
            print(
                f"[rlmath-async][partial] session={sticky_key} turn resumed after {n_aborts} abort(s) "
                f"mode={'continue' if partial_rollout else 'restart'} tokens={len(token_ids)} "
                f"versions={turn_vmin}->{turn_vmax} wall={time.time() - t0:.1f}s "
                f"(total_aborts={book.total_aborts})",
                flush=True,
            )
        eos_ids = self.__dict__.get("_rlmath_eos_ids")
        if eos_ids is None:
            eos_ids = self.__dict__["_rlmath_eos_ids"] = _eos_ids(self.tokenizer)
        trunc_reason, no_eos = classify_turn(token_ids, max_tokens, stop_reason, eos_ids)
        book.note_turn(sticky_key, turn_vmin, turn_vmax, n_aborts, truncated=trunc_reason is not None, no_eos=no_eos)
        if trunc_reason is not None:
            k = book.total_truncated_turns
            if k <= 20 or k % 100 == 0:  # evidence line, rate-limited
                print(
                    f"[rlmath-async][trunc] session={sticky_key} turn truncated reason={trunc_reason} "
                    f"tokens={len(token_ids)} max_tokens={max_tokens} aborts={n_aborts} "
                    f"(total_truncated_turns={k}/{book.total_turns})",
                    flush=True,
                )

        if book.epochs.get(sticky_key_early, 0) != epoch0:
            book.total_stale_dropped += 1
            print(
                f"[rlmath-async][stale] session={sticky_key_early} was reset (trial retry) while this call was "
                f"generating; dropping the abandoned attempt's completion (total={book.total_stale_dropped})",
                flush=True,
            )
            raise RuntimeError(f"rlmath: stale completion for reset session {sticky_key_early}")
        if session_id and tito:
            tito_state[session_id] = (len(normalized), msgs_digest(normalized), prompt_ids, list(token_ids))

        completion_text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        return token_ids, log_probs, stop_reason, completion_text, last_server_id

    def _release_session(self, session_id: str) -> None:
        book = self.__dict__.get("_rlmath_book")
        if book is not None:
            book.sessions.pop(session_id, None)
            book.tito.pop(session_id, None)
            book.init_len.pop(session_id, None)
        return orig_release(self, session_id)

    cls._generate = _generate_resumable
    cls.release_session = _release_session
    setattr(cls, _PATCH_FLAG, True)
    logger.info(
        "[rlmath-async] VLLMRayProvider patched (partial_rollout=%s, max_resumes=%d, tito=%s)", partial_rollout, max_resumes, tito
    )


# ---------------------------------------------------------------------------
# Proxy actor (same public surface as recipe ProxyServerActor + version queries)
# ---------------------------------------------------------------------------

_actor_cls = None
_actor_handle = None  # keep a reference so the actor is not GC'd


def _get_actor_cls():
    global _actor_cls
    if _actor_cls is not None:
        return _actor_cls
    import ray

    @ray.remote(num_cpus=1)
    class AsyncProxyServerActor:
        def __init__(
            self,
            load_balancer: Any,
            model_path: str,
            host: str,
            port: int,
            tool_format: str | None,
            debug: bool,
            session_dump_dir: str | None,
            partial_rollout: bool,
            max_resumes: int,
        ):
            install_provider_patch(partial_rollout=partial_rollout, max_resumes=max_resumes)
            from recipe.agentic.proxyserver.ray_actor import _create_local_proxy

            self._lb = load_balancer
            self.proxy = _create_local_proxy(
                load_balancer=load_balancer,
                model_path=model_path,
                host=host,
                port=port,
                tool_format=tool_format,
                debug=debug,
                session_dump_dir=session_dump_dir,
            )
            self._url: str | None = None

        def _book(self) -> _VersionBook:
            return self.proxy._provider.__dict__.setdefault("_rlmath_book", _VersionBook())

        async def start(self) -> str:
            await self.proxy.start()
            ip = ray.util.get_node_ip_address().strip("[]")
            self._url = f"http://{ip}:{self.proxy._actual_port}"
            return self._url

        def get_url(self) -> str | None:
            return self._url

        def get_load_balancer(self) -> Any:
            return self._lb

        def pop_session_versions(self, session_id: str) -> dict | None:
            return self._book().sessions.pop(session_id, None)

        def reset_session_book(self, session_id: str) -> None:
            self._book().reset_session(session_id)

        def get_latest_version(self) -> int | None:
            return self._book().latest_version

        def get_stats(self) -> dict:
            b = self._book()
            return {
                "latest_version": b.latest_version,
                "total_turns": b.total_turns,
                "total_aborts": b.total_aborts,
                "resumed_turns": b.total_resumed_turns,
                "restarted_turns": b.total_restarted_turns,
                "truncated_turns": b.total_truncated_turns,
                "no_eos_turns": b.total_no_eos_turns,
                "open_sessions": len(b.sessions),
            }

        async def stop(self) -> None:
            if self.proxy is not None:
                await self.proxy.stop()

    _actor_cls = AsyncProxyServerActor
    return _actor_cls


def start_async_proxy_server(
    load_balancer: Any,
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 0,
    tool_format: str | None = "hermes",
    debug: bool = False,
    session_dump_dir: str | None = None,
    partial_rollout: bool = True,
    max_resumes: int = 32,
) -> str:
    """Start the patched proxy as the recipe's named actor on this (head) node."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from recipe.agentic.proxyserver.ray_actor import PROXY_ACTOR_NAME

    global _actor_handle
    try:
        ray.get_actor(PROXY_ACTOR_NAME)
        raise RuntimeError(
            f"a Ray actor named {PROXY_ACTOR_NAME!r} already exists (stale sync proxy?). "
            "It would point at the wrong load balancer; restart the rlmath Ray cluster."
        )
    except ValueError:
        pass

    actor = (
        _get_actor_cls()
        .options(
            name=PROXY_ACTOR_NAME,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(), soft=False),
        )
        .remote(load_balancer, model_path, host, port, tool_format, debug, session_dump_dir, partial_rollout, max_resumes)
    )
    _actor_handle = actor
    url = ray.get(actor.start.remote())
    print(
        f"[rlmath-async] proxy actor {PROXY_ACTOR_NAME!r} at {url} -> rollouter load balancer "
        f"(partial_rollout={partial_rollout}, max_resumes={max_resumes})",
        flush=True,
    )
    return url
