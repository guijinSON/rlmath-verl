"""Numerical check of the Qwen3.5 sequence-packing patch (patch_q35_packing.py).

Runs 3 sequences of different lengths through Qwen3.5 (bf16, flash_attention_2) two ways:
  (a) one at a time, shape (1, L_i), position_ids = arange(L_i)
  (b) packed into ONE row (1, sum L_i), position_ids = concat(arange(L_i))  <- verl remove-padding layout
and compares per-token log-probs of the next token and the gradient of a Gated-DeltaNet weight.
Also runs (c) = packed but with position_ids = arange(sum L_i), i.e. what the model sees when the boundaries are
ignored, to show how large the leak is that the patch removes.

Usage: python check_q35_packing.py <model dir or id> [lens, default 700,2300,4100]
"""

import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

torch.manual_seed(0)
name = sys.argv[1]
lens = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "700,2300,4100").split(",")]
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
model.train()  # dropout is 0; train() so the backward path is the training one
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
vocab = model.config.get_text_config().vocab_size
seqs = [torch.randint(1000, vocab - 1000, (n,), device="cuda") for n in lens]
gdn = next(m for m in model.modules() if type(m).__name__.endswith("GatedDeltaNet"))
probe = gdn.in_proj_qkv.weight


def logps(ids, pos):
    out = model(input_ids=ids[None], position_ids=pos[None], use_cache=False)
    lp = F.log_softmax(out.logits.float(), -1)[0, :-1]
    return lp.gather(-1, ids[1:, None])[:, 0]


# warm-up: the first forward/backward in a process autotunes the fla/Triton kernels; run every shape once
for s in seqs:
    logps(s, torch.arange(len(s), device="cuda")).sum().backward()
_ids = torch.cat(seqs)
model(input_ids=_ids[None], position_ids=torch.cat([torch.arange(len(s), device="cuda") for s in seqs])[None], use_cache=False).logits.float().mean().backward()

# (a) separate
model.zero_grad()
sep = []
for s in seqs:
    lp = logps(s, torch.arange(len(s), device="cuda"))
    lp.sum().backward()
    sep.append(lp.detach())
g_sep = probe.grad.detach().clone()

# (b) packed with boundaries
model.zero_grad()
ids = torch.cat(seqs)
pos = torch.cat([torch.arange(len(s), device="cuda") for s in seqs])
out = model(input_ids=ids[None], position_ids=pos[None], use_cache=False)
lp_all = F.log_softmax(out.logits.float(), -1)[0]
packed, off, total = [], 0, 0
for s in seqs:
    lp = lp_all[off : off + len(s) - 1].gather(-1, s[1:, None])[:, 0]
    packed.append(lp)
    total = total + lp.sum()
    off += len(s)
total.backward()
g_pack = probe.grad.detach().clone()
packed = [p.detach() for p in packed]

# (c) packed, boundaries ignored (one long sequence) - only forward, for the leak magnitude
with torch.no_grad():
    out = model(input_ids=ids[None], position_ids=torch.arange(len(ids), device="cuda")[None], use_cache=False)
    lp_all = F.log_softmax(out.logits.float(), -1)[0]
    leak, off = [], 0
    for s in seqs:
        leak.append(lp_all[off : off + len(s) - 1].gather(-1, s[1:, None])[:, 0])
        off += len(s)

print(f"lens={lens} gdn={type(gdn).__name__}")
for i, n in enumerate(lens):
    d = (packed[i] - sep[i]).abs()
    dl = (leak[i] - sep[i]).abs()
    print(
        f"seq{i} len={n}: |packed-separate| mean={d.mean():.2e} max={d.max():.2e}   "
        f"|no-boundary-separate| mean={dl.mean():.2e} max={dl.max():.2e}"
    )
rel = (g_pack - g_sep).norm() / g_sep.norm()
print(f"grad {type(gdn).__name__}.in_proj_qkv: rel err packed vs separate = {rel:.2e} (|g|={g_sep.norm():.3e})")
