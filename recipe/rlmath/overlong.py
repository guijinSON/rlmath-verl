"""DAPO-style overlong filtering decision (pure python, no torch; unit-tested in async_check).

A trajectory is *truncated* if
  (a) its response hit ``max_response_length``: the same test verl uses for
      ``response_length/clip_ratio`` (metric_utils.py:75-81, 236), i.e. attention-mask response
      length == responses.shape[-1]; or
  (b) any of its LLM calls hit the per-call ``max_tokens`` (``truncated_turns > 0``, recorded by
      async_proxy.classify_turn; resumed turns count only if their final generation hit it).

With the filter enabled, truncated trajectories are dropped from the policy loss by zeroing their
response (= loss) mask after advantages are computed. They keep their reward and their place in
the GRPO group statistics.

``overlong_mode`` (rlmath_async.overlong_mode):
  * ``filter``   - the behaviour above (drop truncated trajectories from the loss).
  * ``penalize`` - truncated trajectories get reward 0 BEFORE the GRPO advantage is computed and stay in
                   the loss, so rambling past a length limit gets a negative advantage whenever a sibling
                   finishes. Filtering gave no signal against runaway thinking (Qwen3.5-4B on KT: 13-42%
                   of every batch was cut and dropped, and first-turn 32k truncations grew 29% -> 50% of val).
  * ``off``      - neither.

Infra-failed trials (the sandbox could not be reached / started: Docker errors, environment start
timeouts) are not the policy's fault. ``grpo_advantages_excluding`` recomputes GRPO advantages with
them left out of their group's mean/std, and gives them advantage 0.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

OVERLONG_MODES = ("filter", "penalize", "off")

# Harbor/agent errors that mean "the sandbox host failed", not "the policy failed".
# AgentTimeoutError is deliberately NOT here: an agent that runs out the clock is a policy failure.
INFRA_ERROR_RE = re.compile(
    r"docker|compose|EnvironmentStart|AddTestsDir|Failed to detect Docker platform|Cannot connect to the Docker"
    r"|Connection (refused|reset)|ConnectionError|No space left on device|OCI runtime|container .* (not found|is not running)"
    r"|sbx-docker\.sock|BuildError|ImageNotFound"
    # Apptainer sandboxes (JUPITER): Harbor's Singularity backend failing to start/feed the in-container server
    r"|Singularity|apptainer|FastAPI server|Failed to upload directory|Failed to start tmux session|ConnectError"
    r"|RemoteProtocolError|Server disconnected"
    # filesystem quota on the host (10-12, JUPITER c9: EDQUOT on trial dirs -> setup/verifier failures scored 0 and trained on)
    r"|Disk quota exceeded|Errno 122|No reward file found",
    re.IGNORECASE,
)


# The policy's own inference server failed (vLLM engine died, proxy 500). Also infra, but unrecoverable inside the
# run: on MLXP 10-06 every vLLM engine died on its first request and the run spent 3 h producing 500s before it
# crashed. async_agent_loop aborts the run when these persist (see _note_llm_failure).
LLM_ERROR_RE = re.compile(
    r"EngineDeadError|Verl-vllmException|InternalServerError|OpenAIException - Error code: 5\d\d|APIConnectionError"
    r"|ServiceUnavailable",
)


def is_llm_server_failure(status: Any, error: Any) -> bool:
    if str(status or "").lower() == "completed":
        return False
    return bool(error) and bool(LLM_ERROR_RE.search(str(error)))


def resolve_overlong_mode(mode: Any, overlong_filter: Any) -> str:
    """``mode`` wins when set; otherwise the legacy boolean ``overlong_filter`` picks filter/off."""
    if mode not in (None, "", "null", "None"):
        m = str(mode).strip().lower()
        if m not in OVERLONG_MODES:
            raise ValueError(f"rlmath_async.overlong_mode must be one of {OVERLONG_MODES}, got {mode!r}")
        return m
    return "filter" if bool(overlong_filter) else "off"


def is_infra_failure(status: Any, error: Any) -> bool:
    """True when a trial did not complete because of the sandbox host (see INFRA_ERROR_RE)."""
    if str(status or "").lower() == "completed":
        return False
    return bool(error) and bool(INFRA_ERROR_RE.search(str(error)) or LLM_ERROR_RE.search(str(error)))


def grpo_advantages_excluding(scores, index, valid, epsilon: float = 1e-6, norm_by_std: bool = True):
    """verl's compute_grpo_outcome_advantage (core_algos.py) on the ``valid`` trajectories only.

    scores: float tensor [B] (summed token-level rewards); index: group id per row; valid: list[bool].
    Same conventions as verl: a group with one member gets mean 0 / std 1, std is torch.std (unbiased).
    Invalid rows (and groups with no valid row) get advantage 0. Returns a float tensor [B].
    """
    import torch

    out = torch.zeros_like(scores, dtype=torch.float32)
    groups: dict[Any, list[int]] = {}
    for i, (g, ok) in enumerate(zip(index, valid, strict=True)):
        if ok:
            groups.setdefault(g, []).append(i)
    for rows in groups.values():
        s = scores[rows].float()
        if len(rows) == 1:
            mean, std = torch.tensor(0.0), torch.tensor(1.0)
        else:
            mean, std = s.mean(), s.std()
        out[rows] = (s - mean) / (std + epsilon) if norm_by_std else (s - mean)
    return out


def _as_int(x: Any) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def overlong_decision(
    response_lengths: Sequence[float],
    max_response_length: int,
    truncated_turns: Sequence[Any] | None,
    llm_turns: Sequence[Any] | None,
    loss_tokens: Sequence[float],
    enabled: bool,
) -> tuple[list[bool], dict[str, float]]:
    """Return (keep mask per trajectory, metrics).

    ``loss_tokens``: per-trajectory count of response_mask==1 tokens before filtering.
    ``rlmath/loss_tokens == 0`` after filtering means the caller must skip the actor update:
    verl's token-mean normalizes by the loss-mask sum (fsdp/transformer_impl.py:621-626 ->
    core_algos.agg_loss), so an all-zero mask would divide 0 by 0 and give NaN gradients.
    """
    n = len(response_lengths)
    tt = [_as_int(x) for x in (truncated_turns if truncated_turns is not None else [0] * n)]
    lt = [_as_int(x) for x in (llm_turns if llm_turns is not None else [0] * n)]
    hit_len = [int(round(float(r))) >= int(max_response_length) for r in response_lengths]
    hit_turn = [t > 0 for t in tt]
    truncated = [a or b for a, b in zip(hit_len, hit_turn, strict=True)]
    keep = [not t for t in truncated] if enabled else [True] * n

    kept_tokens = float(sum(tok for tok, k in zip(loss_tokens, keep, strict=True) if k))
    kept_trajs = sum(1 for tok, k in zip(loss_tokens, keep, strict=True) if k and tok > 0)
    denom = max(n, 1)
    metrics = {
        "rlmath/overlong_filter_enabled": float(bool(enabled)),
        # trajectories dropped from the loss this step (0 when the filter is off)
        "rlmath/overlong_filtered_frac": sum(1 for k in keep if not k) / denom,
        # would-be-filtered fraction, logged even with the filter off
        "rlmath/truncated_traj_frac": sum(truncated) / denom,
        "rlmath/truncated_by_response_len_frac": sum(hit_len) / denom,
        "rlmath/truncated_by_turn_frac": sum(hit_turn) / denom,
        # LLM calls that hit the per-call max_tokens / all LLM calls in the batch
        "rlmath/truncated_turn_frac": sum(tt) / max(sum(lt), 1),
        # trajectories / tokens that still contribute to the policy loss
        "rlmath/loss_trajectories": float(kept_trajs),
        "rlmath/loss_tokens": kept_tokens,
        "rlmath/actor_update_skipped": float(kept_tokens <= 0),
    }
    return keep, metrics
