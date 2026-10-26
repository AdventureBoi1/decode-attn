# decode-attn

Single-token grouped-query attention. One new query reads a KV cache that is split across the GPU, and every query head that shares a KV head reuses that tile instead of reading it again. A second kernel merges the partial softmax state.

This is the attention half of a decode step. The MLP half is [fused-decode](https://github.com/AdventureBoi1/fused-decode). PyTorch 2.11 on Windows is not built with FlashAttention, and its memory-efficient kernel rejects this head layout (`Hq=32`, `Hkv=8`). `scaled_dot_product_attention` therefore runs the math kernel. That is the baseline below.

## Install

Install a CUDA PyTorch wheel first, then:

```
pip install git+https://github.com/AdventureBoi1/decode-attn.git
```

```python
import torch.nn.functional as F
import decode_attn

decode_attn.enable()
# q: [B, Hq, 1, D], k/v: [B, Hkv, S, D]
y = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
decode_attn.disable()
```

`enable` only replaces single-token CUDA calls in float16 or bfloat16. A longer prompt, a mask, or dropout stays on PyTorch. Call `enable` before another file does `from torch.nn.functional import scaled_dot_product_attention`.

## Result

RTX 4070 Laptop, PyTorch 2.11.0+cu128, Triton 3.6.0. A 64MB fp16 copy in this process moves **224 GB/s**. Llama 3 8B widths: batch 1, 32 query heads, 8 KV heads, head dim 128.

| Cache | dtype | This kernel | PyTorch SDPA | Bandwidth |
| --- | --- | ---: | ---: | ---: |
| 1024 | fp16 | 62 µs | 523 µs | 68 GB/s |
| 4096 | fp16 | 109 µs | 2.22 ms | 153 GB/s |
| 16384 | fp16 | **320 µs** | 8.48 ms | **210 GB/s** |
| 16384 | bf16 | **320 µs** | 8.49 ms | **210 GB/s** |

At a 16k cache the kernel is **27×** the PyTorch call and at **94%** of the copy. The KV traffic is 64MB, which does not fit in the 32MB L2, so that percentage is DRAM. Max absolute error against an fp32 reference is 3e-5 for fp16 and 2e-4 for bf16.

The 4k row is a 16MB working set, so it can sit in L2. It also moves with the laptop's clocks: another pass of the same shape on this card was 68 µs. The 16k row was 320 µs both times. A 128-token cache is launch-bound (56 µs fp16 against PyTorch at 220 µs).

FlashAttention is not in this wheel (`Torch was not compiled with flash attention`). The memory-efficient kernel refuses unequal query and KV head counts. Forcing cuDNN returned a result in the probe, and the default dispatcher still took the same time as the math backend. The speedup is against the kernel PyTorch actually runs here, not against a Linux FlashAttention build.

## Algorithm

Online softmax, the same recurrence as FlashAttention. Masked keys get a large negative score rather than `-inf`, so an empty tile does not turn into NaN while the running max is still empty.

The cache is cut into splits (256 tokens at 16k, 512 at 4k, 128 at 1k). Each program owns one KV head and one split, and updates every query head in that group before dropping the tile. The merge is the usual log-sum-exp combination of per-split max, sum, and output accumulator. Scores and the accumulator are fp32. The probability-value product is rounded back to the activation dtype so it can use the tensor cores.

`q` is `[B, Hq, D]` or `[B, Hq, 1, D]`. `k` and `v` are `[B, Hkv, S, D]`, the cache including the new token. `Hq` must be divisible by `Hkv`. There is no backward.

## Reproduce

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest
.venv\Scripts\python -m decode_attn.bench --out results\rtx4070-laptop.json
```

The machine this was measured on already has that interpreter at `..\fused-decode\.venv`. `DECODE_ATTN_SPLIT` and `DECODE_ATTN_TILE` override the tile choice.

## Limits

- One query token. This is not a prefill kernel.
- Head dim up to 256, and at most 16 query heads per KV head.
- Inference only.
- A datacenter GPU with FlashAttention compiled in is a different baseline. The 27× figure belongs to this Windows wheel.
