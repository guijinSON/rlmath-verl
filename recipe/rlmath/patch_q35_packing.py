"""Make transformers' Qwen3.5 Gated-DeltaNet respect packed-sequence boundaries (sequence packing fix).

verl's remove-padding + dynamic-batch path packs several trajectories into ONE row of shape (1, total_len)
and marks the boundaries only through position_ids (each sequence restarts at 0). Full-attention layers use
that (flash-attn varlen); transformers 5.3's Qwen3_5GatedDeltaNet ignores it: causal_conv1d gets
seq_idx=None and chunk_gated_delta_rule gets no cu_seqlens, so the conv window and the recurrent state leak
from one trajectory into the next. That is why the Qwen3.5 runs had to use DYNAMIC_BSZ=false MICRO_BS=1
(one sequence per forward), which made the trainer the bottleneck.

This patch:
  1. passes the decoder layer's (text) position_ids into the linear-attention layer;
  2. in Qwen3_5GatedDeltaNet.forward, when there is no cache and batch==1, derives cu_seqlens from the
     positions where position_ids == 0;
  3. runs the depthwise causal conv per segment (torch path; causal_conv1d's seq_idx when it is installed);
  4. passes cu_seqlens to fla's chunk_gated_delta_rule (fla >= 0.5 supports varlen with batch 1).
Unpacked inputs (one sequence, or any cached/generation call) take exactly the original code path.

Usage: python patch_q35_packing.py <site-packages dir>     (idempotent; keeps <file>.prepack backups)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

MARK = "# rlmath packing patch"

DECODER_OLD = """            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )"""
DECODER_NEW = """            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                position_ids=position_ids,  # rlmath packing patch
            )"""

SIG_OLD = """        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape
"""
SIG_NEW = """        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape

        # rlmath packing patch: a packed (remove-padding) row holds several sequences; position_ids restart at 0
        # at every sequence start. Derive cu_seqlens so neither the conv nor the recurrence crosses a boundary.
        packed_cu_seqlens = None
        if cache_params is None and batch_size == 1 and position_ids is not None:
            _pos = position_ids
            while _pos.dim() > 1:
                _pos = _pos[0]
            if _pos.shape[-1] == seq_len:
                _starts = torch.nonzero(_pos == 0, as_tuple=False).flatten()
                if _starts.numel() > 1:
                    if int(_starts[0]) != 0:
                        _starts = torch.cat([_starts.new_zeros(1), _starts])
                    packed_cu_seqlens = torch.cat([_starts, _starts.new_tensor([seq_len])]).to(torch.long)
"""

CONV_OLD = """            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])"""
CONV_NEW = """            if packed_cu_seqlens is not None:  # rlmath packing patch
                if self.causal_conv1d_fn is not None:
                    _seq_idx = torch.zeros(seq_len, dtype=torch.int32, device=mixed_qkv.device)
                    _seq_idx[packed_cu_seqlens[1:-1]] = 1
                    _seq_idx = torch.cumsum(_seq_idx, 0).to(torch.int32).unsqueeze(0)
                    mixed_qkv = self.causal_conv1d_fn(
                        x=mixed_qkv,
                        weight=self.conv1d.weight.squeeze(1),
                        bias=self.conv1d.bias,
                        activation=self.activation,
                        seq_idx=_seq_idx,
                    )
                else:
                    _b = packed_cu_seqlens.tolist()
                    mixed_qkv = torch.cat(
                        [F.silu(self.conv1d(mixed_qkv[:, :, s:e])[:, :, : e - s]) for s, e in zip(_b[:-1], _b[1:])],
                        dim=-1,
                    )
            elif self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])"""

CHUNK_OLD = """        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )"""
CHUNK_NEW = """        if not use_precomputed_states:
            _varlen = {}
            if packed_cu_seqlens is not None:  # rlmath packing patch
                if self.chunk_gated_delta_rule is torch_chunk_gated_delta_rule:
                    raise RuntimeError("packed Qwen3.5 sequences need fla's chunk_gated_delta_rule (cu_seqlens)")
                _varlen = {"cu_seqlens": packed_cu_seqlens}
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                **_varlen,
            )"""


def patch_file(path: Path, gdn_class: str) -> str:
    src = path.read_text()
    if MARK in src:
        return f"already patched: {path}"
    for name, old in (("decoder", DECODER_OLD), ("signature", SIG_OLD), ("conv", CONV_OLD), ("chunk", CHUNK_OLD)):
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"{path}: expected exactly one '{name}' block, found {n} - transformers version changed?")
    if f"class {gdn_class}(" not in src:
        raise SystemExit(f"{path}: {gdn_class} not found")
    backup = path.with_suffix(path.suffix + ".prepack")
    if not backup.exists():
        shutil.copy2(path, backup)
    src = (
        src.replace(DECODER_OLD, DECODER_NEW)
        .replace(SIG_OLD, SIG_NEW)
        .replace(CONV_OLD, CONV_NEW)
        .replace(CHUNK_OLD, CHUNK_NEW)
    )
    path.write_text(src)
    return f"patched: {path} (backup {backup.name})"


def main() -> None:
    site = Path(sys.argv[1])
    targets = [
        (site / "transformers/models/qwen3_5/modeling_qwen3_5.py", "Qwen3_5GatedDeltaNet"),
        (site / "transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py", "Qwen3_5MoeGatedDeltaNet"),
    ]
    for path, cls in targets:
        if path.exists():
            try:
                print(patch_file(path, cls))
            except SystemExit as e:
                print(f"SKIP {path.name}: {e}")
        else:
            print(f"missing: {path}")


if __name__ == "__main__":
    main()
