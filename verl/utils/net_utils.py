# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ipaddress
import os
import random
import socket


def is_ipv4(ip_str: str) -> bool:
    """
    Check if the given string is an IPv4 address

    Args:
        ip_str: The IP address string to check

    Returns:
        bool: Returns True if it's an IPv4 address, False otherwise
    """
    try:
        ipaddress.IPv4Address(ip_str)
        return True
    except ipaddress.AddressValueError:
        return False


def is_ipv6(ip_str: str) -> bool:
    """
    Check if the given string is an IPv6 address

    Args:
        ip_str: The IP address string to check

    Returns:
        bool: Returns True if it's an IPv6 address, False otherwise
    """
    try:
        ipaddress.IPv6Address(ip_str)
        return True
    except ipaddress.AddressValueError:
        return False


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def _listen_port_range() -> tuple[int, int] | None:
    spec = os.environ.get("RLMATH_LISTEN_PORT_RANGE", "20000-29999").strip().lower()
    if spec in ("", "0", "off", "none"):
        return None
    lo, hi = (int(x) for x in spec.split("-"))
    return lo, hi


def bind_listen_port(sock: socket.socket, address: str = "") -> int:
    """Bind `sock` to a free port in RLMATH_LISTEN_PORT_RANGE (default 20000-29999) and return it.

    bind(0) returns a port from the kernel's ephemeral range (32768-60999 on JUPITER/MLXP/KT). Callers here close the
    probe socket and bind the real server later, and in that gap an outgoing connection can take the same port: torch's
    TCPStore then fails with EADDRINUSE (arxivmath c6 2457157 and construct c1, CheckpointEngineWorker init). Ports
    below the ephemeral range are never handed to outgoing connections. Random start from SystemRandom because workers
    seed the global `random` identically. Set RLMATH_LISTEN_PORT_RANGE=off to restore bind(0).
    """
    rng = _listen_port_range()
    if rng is not None:
        lo, hi = rng
        n = hi - lo + 1
        start = random.SystemRandom().randrange(n)
        for i in range(n):
            port = lo + (start + i) % n
            try:
                sock.bind((address, port))
                return port
            except OSError:
                continue
    sock.bind((address, 0))
    return sock.getsockname()[1]


def get_free_port(address: str, with_alive_sock: bool = False) -> tuple[int, socket.socket | None]:
    """Find a free port on the given address.

    By default the socket is closed internally, suitable for immediate use.
    Set with_alive_sock=True to keep the socket open as a port reservation,
    preventing other calls from getting the same port. The caller is
    responsible for closing the socket before the port is actually bound
    by the target service (e.g. NCCL, uvicorn).
    """
    family = socket.AF_INET6 if is_valid_ipv6_address(address) else socket.AF_INET

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = bind_listen_port(sock, address)
    if with_alive_sock:
        return port, sock
    sock.close()
    return port, None
