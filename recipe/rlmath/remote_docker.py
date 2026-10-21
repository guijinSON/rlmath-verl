"""Harbor Docker environment for a *remote* Docker daemon (DOCKER_HOST=ssh://...).

Bind mounts resolve on the daemon's host, not here, so logs written to /logs inside the
container would never reach this machine. Declare the environment as not mounted and
skip the mounts override: Harbor then uploads tests and downloads /logs/{agent,verifier}
through the Docker API (docker cp), exactly as it does for remote sandboxes.

``RLMATH_ENV_LIFECYCLE_PER_WORKER`` (unset/0 = no limit) caps concurrent ``start``/``stop``
calls per agent-loop worker process. ``docker compose up/down`` is the thread-hungry part of a
trial (dozens of threads per call); on hosts with a small cgroup pids.max (KT: 8192) a burst of
100+ simultaneous starts (validation, step boundaries) exhausts it, while steady-state trials
are cheap. Throttling only the lifecycle lets many more trials run concurrently.

Multi-VM routing: ``RLMATH_DOCKER_HOSTS_FILE`` names a file with one ``<docker host> [weight]`` per line, e.g.

    unix:///tmp/sbx-docker.sock   1.0
    unix:///tmp/sbx4-docker.sock  0.4

Each new trial goes to the host with the fewest assigned-and-not-yet-stopped trials (in this worker process) per unit of
weight, ties broken at random; trials are counted when assigned, because Harbor creates a batch of environments before
any of them starts. The file
is re-read every 20 s, so hosts can be added, re-weighted (weight 0 = drain: no new trials) or removed while a run is
going; a ``unix://`` host whose socket does not exist (tunnel down) is skipped. A trial stays on its host for its whole
life: plain ``docker`` calls get ``-H <host>`` (Harbor builds them through the class method ``_engine_cmd``, so each
host gets its own subclass) and ``docker compose`` calls get ``DOCKER_HOST=<host>`` in their environment. Without the
file every trial uses DOCKER_HOST from the process environment, as before.
"""
import asyncio
import collections
import logging
import os
import random
import time

from harbor.environments.docker.docker import DockerEnvironment

_log = logging.getLogger(__name__)

_LIMIT = int(os.getenv("RLMATH_ENV_LIFECYCLE_PER_WORKER", "0") or 0)
_sem = None

_HOSTS_FILE = os.getenv("RLMATH_DOCKER_HOSTS_FILE") or None
_HOSTS_TTL = 20.0
_hosts_cache: list = [0.0, []]  # [read time, [(host, weight), ...]]
_active: collections.Counter = collections.Counter()  # host -> trials assigned and not yet stopped (this process)
_routed: dict = {}  # host -> subclass bound to it


def _lifecycle_sem():
    global _sem
    if _sem is None and _LIMIT > 0:
        _sem = asyncio.Semaphore(_LIMIT)
    return _sem


def _read_hosts() -> list:
    if not _HOSTS_FILE:
        return []
    now = time.time()
    if now - _hosts_cache[0] < _HOSTS_TTL:
        return _hosts_cache[1]
    hosts = []
    try:
        with open(_HOSTS_FILE) as f:
            for line in f:
                parts = line.split("#", 1)[0].split()
                if not parts:
                    continue
                host, weight = parts[0], float(parts[1]) if len(parts) > 1 else 1.0
                if weight <= 0:
                    continue
                if host.startswith("unix://") and not os.path.exists(host[len("unix://"):]):
                    continue
                hosts.append((host, weight))
    except (OSError, ValueError) as e:
        _log.warning("[rlmath-docker] cannot read %s (%s); keeping the previous host list", _HOSTS_FILE, e)
        hosts = _hosts_cache[1]
    _hosts_cache[0], _hosts_cache[1] = now, hosts
    return hosts


def pick_host() -> str | None:
    """Least loaded host relative to its weight; None = use DOCKER_HOST from the environment."""
    hosts = _read_hosts()
    if not hosts:
        return None
    return min(hosts, key=lambda hw: ((_active[hw[0]] + 1) / hw[1], random.random()))[0]


def routed_class(host: str):
    cls = _routed.get(host)
    if cls is None:
        cls = type(f"RemoteDockerEnvironment_h{len(_routed)}", (RemoteDockerEnvironment,), {"_rlmath_host": host})
        _routed[host] = cls
    return cls


class RemoteDockerEnvironment(DockerEnvironment):
    _rlmath_host: str | None = None  # set on the per-host subclasses

    def __new__(cls, *args, **kwargs):
        host = None
        if cls is RemoteDockerEnvironment:
            host = pick_host()
            if host:
                cls = routed_class(host)
        obj = super().__new__(cls)
        if host:
            obj._rlmath_counted = True  # counted from assignment until stop()
            _active[host] += 1
        return obj

    @classmethod
    def _engine_cmd(cls, *args: str) -> list[str]:
        cmd = super()._engine_cmd(*args)
        if cls._rlmath_host:
            cmd = [cmd[0], "-H", cls._rlmath_host, *cmd[1:]]
        return cmd

    def _compose_env_vars(self, include_os_env: bool = True) -> dict[str, str]:
        env = super()._compose_env_vars(include_os_env=include_os_env)
        # only the environment of the docker compose process itself; include_os_env=False builds container env
        if self._rlmath_host and include_os_env:
            env["DOCKER_HOST"] = self._rlmath_host
        return env

    @property
    def capabilities(self):
        return super().capabilities.model_copy(update={"mounted": False})

    def _write_mounts_compose_file(self):
        self._cleanup_mounts_compose_file()
        return None

    async def start(self, force_build: bool):
        sem = _lifecycle_sem()
        if sem is None:
            return await super().start(force_build)
        async with sem:
            return await super().start(force_build)

    async def stop(self, delete: bool):
        try:
            sem = _lifecycle_sem()
            if sem is None:
                return await super().stop(delete)
            async with sem:
                return await super().stop(delete)
        finally:
            if getattr(self, "_rlmath_counted", False):
                self._rlmath_counted = False
                _active[self._rlmath_host] -= 1
