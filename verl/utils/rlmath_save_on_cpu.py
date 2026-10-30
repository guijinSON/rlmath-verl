"""Selective save-on-CPU for the training forward (RLMATH_SAVE_ON_CPU=1, 10-19).

torch.autograd.graph.save_on_cpu copies EVERY saved tensor and crashes on nested/jagged tensors (verl's no-padding loss:
"SymIntArrayRef expected to contain only concrete integers"). This version moves only plain dense CUDA tensors of at
least MIN_BYTES with concrete shapes (in practice the per-layer checkpoint inputs, the bulk of the memory) to pinned
CPU memory (pageable by default) and leaves everything else on the GPU. Values are unchanged.
"""
import torch

import os

MIN_BYTES = 16 << 20
PIN = os.environ.get("RLMATH_SAVE_ON_CPU_PIN") == "1"
# only offload micro-batches of at least this many tokens (10-30): peak memory is set by the longest trajectory, so short
# ones run at full speed on the GPU; 0 = offload every micro-batch
MIN_TOKENS = int(os.environ.get("RLMATH_SAVE_ON_CPU_MIN_TOKENS", "0"))


def _pack(t):
    try:
        if (t.is_nested or not t.is_cuda or t.layout != torch.strided or torch.compiler.is_compiling()
                or any(not isinstance(s, int) for s in t.shape) or t.numel() * t.element_size() < MIN_BYTES):
            return t
        # pageable, not pinned (10-19): torch caches pinned host blocks forever -> ~80 GB RSS per trainer process and
        # Ray's OOM killer on 128x16/256x16 nodes; pageable buffers are returned to the OS as soon as backward frees them
        cpu = torch.empty(tuple(t.shape), dtype=t.dtype, pin_memory=PIN)
        cpu.copy_(t)
        return (t.device, cpu)
    except Exception:
        return t


def _unpack(p):
    if isinstance(p, tuple):
        return p[1].to(p[0], non_blocking=True)
    return p


def save_on_cpu_ctx():
    return torch.autograd.graph.saved_tensors_hooks(_pack, _unpack)


def micro_batch_tokens(micro_batch):
    ids = micro_batch["input_ids"]
    return ids.values().numel() if ids.is_nested else ids.numel()   # padded layout counts padding: offloads more, never less


def wants_save_on_cpu(micro_batch):
    if MIN_TOKENS <= 0:
        return True
    try:
        return micro_batch_tokens(micro_batch) >= MIN_TOKENS
    except Exception:
        return True
