"""Terminus-2 adapted to verl's RemoteAgentLoop.

The loop passes model_base_url / api_key / top_p, which Terminus2's strict options model
rejects. Map them: top_p -> llm_call_kwargs, api_key -> OPENAI_API_KEY for LiteLLM,
model_base_url is redundant with api_base. Summarization is off (its extra LLM calls would
not belong to the policy trajectory) and terminal recording is off (asciinema cannot be
installed in no-network sandboxes).
"""
import inspect
import logging
import os

from harbor.agents.base import BaseAgent
from harbor.agents.terminus_2.terminus_2 import Terminus2, Terminus2Options

_log = logging.getLogger(__name__)
_ACCEPTED = (set(Terminus2Options.model_fields) | set(inspect.signature(Terminus2.__init__).parameters)
             | set(inspect.signature(BaseAgent.__init__).parameters))


_THINK_END = "</think>"


class _StripThinking:
    """Parser proxy: reasoning models (Qwen3-*-Thinking) return their reasoning trace in the
    message content, which breaks Terminus's JSON parser. Parse only the text after the last
    </think>."""

    def __init__(self, inner):
        self._inner = inner

    @staticmethod
    def _strip(text):
        if text and _THINK_END in text:
            return text.rsplit(_THINK_END, 1)[1].lstrip()
        return text

    def parse_response(self, response):
        return self._inner.parse_response(self._strip(response))

    def __getattr__(self, name):
        # Terminus only calls salvage_truncated_response if hasattr(parser, ...) — and only the XML parser has it.
        # Expose it (with <think> stripping) only when the wrapped parser really implements it; defining it
        # unconditionally made every length-truncated reply crash the trial with AttributeError (+ a full retry).
        attr = getattr(self._inner, name)  # raises AttributeError -> hasattr() False, as for the bare parser
        if name == "salvage_truncated_response":
            return lambda response, *a, **k: attr(self._strip(response), *a, **k)
        return attr


class RLTerminus2(Terminus2):
    def __init__(self, *args, model_base_url=None, api_key=None, top_p=None, **kwargs):
        call_kwargs = dict(kwargs.pop("llm_call_kwargs", None) or {})
        if top_p is not None:
            call_kwargs.setdefault("top_p", top_p)
        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
        if model_base_url and not kwargs.get("api_base"):
            kwargs["api_base"] = model_base_url
        # drop options meant for other agents (e.g. the recipe's swe-agent cost limits)
        for k in [k for k in kwargs if k not in _ACCEPTED]:
            _log.warning("RLTerminus2: dropping unsupported kwarg %r", k)
            kwargs.pop(k)
        kwargs.setdefault("enable_summarize", False)
        kwargs.setdefault("record_terminal_session", False)
        super().__init__(*args, llm_call_kwargs=call_kwargs or None, **kwargs)
        self._parser = _StripThinking(self._parser)
