"""Token-in/token-out consistency between rollout prompts and the trained sequence.

The bug this fixes
------------------
The trainer (``RemoteAgentLoop._reconstruct_output``) trains on ONE concatenated
sequence per trajectory::

    prompt(turn0 messages) + completion_0 + gap(new msgs) + completion_1 + gap + ...

where ``completion_k`` are the exact vLLM token ids (full ``<think>`` reasoning).
But the proxy rendered every turn's prompt from the agent's chat messages with the
chat template, and Qwen3(-Thinking) templates DROP the reasoning of earlier assistant
turns (Terminus sends terminal output as ``user`` messages, so every earlier turn is
"before the last user query").  So turn k>=1 was *sampled* without the earlier
reasoning in context but *trained* with it: a systematic train/rollout mismatch
(measured rollout_corr/kl ~0.045 on Qwen3-4B-Thinking, vs ~1e-3 expected).

Fix: both sides build the prompt of turn k>=1 as the exact continuation
``prompt_{k-1} + completion_{k-1} + gap_ids(prev_completion, new_msgs)`` with the
single ``gap_ids`` below, so rollout context == training context token for token.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

# both the proxy (rollout prompts) and the agent loop (trained sequence) read this, so they always agree
ENABLED = os.getenv("RLMATH_TITO", "1") == "1"

GAP_ROLES = ("tool", "user", "system")


def _special_id(tokenizer, tok: str) -> int | None:
    tid = tokenizer.convert_tokens_to_ids(tok)
    return tid if isinstance(tid, int) and tid >= 0 and tid != getattr(tokenizer, "unk_token_id", None) else None


def _template_ids(tokenizer, msgs: list[dict[str, Any]], add_generation_prompt: bool) -> list[int]:
    enc = tokenizer.apply_chat_template(msgs, add_generation_prompt=add_generation_prompt, tokenize=True)
    return list(enc["input_ids"] if hasattr(enc, "keys") else enc)


def gap_ids(tokenizer, prev_completion_ids: list[int], msgs: list[dict[str, Any]]) -> list[int]:
    """Tokens between the previous completion and the next completion.

    = [<|im_end|> if the previous completion did not end with it] + "\\n"
      + chat-template rendering of ``msgs`` + generation prompt (e.g. "<|im_start|>assistant\\n<think>\\n").
    ``msgs`` must already be normalized and contain only GAP_ROLES.
    """
    if not msgs:
        return []
    # Render msgs on their own (tool-only deltas need a dummy user turn in front so the
    # template emits the tool wrapper; strip the dummy's tokens again).
    if any(m.get("role") == "user" for m in msgs):
        pfx: list[dict[str, Any]] = []
    else:
        pfx = [{"role": "user", "content": "x"}]
    ids = _template_ids(tokenizer, pfx + msgs, add_generation_prompt=True)
    if pfx:
        pfx_ids = _template_ids(tokenizer, pfx, add_generation_prompt=False)
        if ids[: len(pfx_ids)] == pfx_ids:
            ids = ids[len(pfx_ids):]
    # the template puts "<|im_end|>\n" after every message; the completion ends at <|im_end|>
    lead: list[int] = []
    im_end = _special_id(tokenizer, "<|im_end|>")
    if im_end is not None:
        if not prev_completion_ids or prev_completion_ids[-1] != im_end:
            lead.append(im_end)
        nl = tokenizer.encode("\n", add_special_tokens=False)
        if ids[: len(nl)] != nl:
            lead.extend(nl)
    return lead + ids


def split_delta(prev_n: int, msgs: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """New gap messages of an append-only continuation, or None if not a clean continuation.

    A clean continuation of a conversation that had ``prev_n`` messages at the previous
    call is: exactly one assistant message (the previous completion) followed by >=1
    GAP_ROLES messages.
    """
    delta = msgs[prev_n:]
    if len(delta) < 2 or delta[0].get("role") != "assistant":
        return None
    rest = delta[1:]
    if any(m.get("role") not in GAP_ROLES for m in rest):
        return None
    return rest


def msgs_digest(msgs: list[dict[str, Any]]) -> str:
    return hashlib.sha1(json.dumps(msgs, sort_keys=True, default=str).encode()).hexdigest()
