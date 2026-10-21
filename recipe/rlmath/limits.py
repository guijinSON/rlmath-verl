"""Live-tunable rollout concurrency (no restart needed to change it).

``RLMATH_LIMITS_FILE`` names a file with ``<key> <value>`` lines (``#`` comments allowed), re-read every 30 s:

    max_concurrent_prompts  64   # rollouter: GRPO groups (prompts) in flight; clamped to [1, verl's own cap]
    trials_per_worker       56   # each AgentLoopWorker process: Harbor trials running at once (x num_workers in total)

Capping trials rather than groups keeps sandbox load fixed while a group in its tail (waiting on its slowest trial)
no longer blocks capacity: the freed trial slots go to the next group's trials. A key that is missing (or no file)
falls back to the launch-time value (rlmath_async.max_concurrent_prompts / RLMATH_TRIALS_PER_WORKER).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

_log = logging.getLogger(__name__)

LIMITS_FILE = os.getenv("RLMATH_LIMITS_FILE") or None
TTL = 30.0
_cache: list = [0.0, {}]


def read_limits() -> dict[str, int]:
    """{key: int} from RLMATH_LIMITS_FILE, cached for TTL seconds; {} when unset/unreadable."""
    if not LIMITS_FILE:
        return {}
    now = time.time()
    if now - _cache[0] < TTL:
        return _cache[1]
    out: dict[str, int] = {}
    try:
        with open(LIMITS_FILE) as f:
            for line in f:
                parts = line.split("#", 1)[0].split()
                if len(parts) >= 2:
                    out[parts[0]] = int(float(parts[1]))
    except (OSError, ValueError) as e:
        _log.warning("[rlmath-limits] cannot read %s (%s); keeping previous limits", LIMITS_FILE, e)
        out = _cache[1]
    _cache[0], _cache[1] = now, out
    return out


class ResizableLimiter:
    """asyncio concurrency limiter whose limit is re-evaluated while waiting (``limit_fn`` is called each time).
    Lowering the limit lets running holders finish; raising it admits waiters within ~TTL seconds."""

    def __init__(self, limit_fn):
        self._limit_fn = limit_fn
        self.active = 0
        self._cond = asyncio.Condition()

    def limit(self) -> int:
        return max(1, int(self._limit_fn()))

    async def __aenter__(self):
        async with self._cond:
            while self.active >= self.limit():
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=TTL)
                except asyncio.TimeoutError:
                    pass  # re-check: the limit may have been raised in the file
            self.active += 1
        return self

    async def __aexit__(self, *exc):
        async with self._cond:
            self.active -= 1
            self._cond.notify_all()
        return False
