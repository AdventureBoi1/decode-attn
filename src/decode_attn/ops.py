"""Single-token decode attention.

``q`` is the new token. ``k`` and ``v`` are the cache, including that token.
Query heads share KV heads in contiguous groups, the same layout as Llama.
"""

from __future__ import annotations

import torch

from decode_attn.kernels import decode_attention_out

_ACTIVATION = (torch.float16, torch.bfloat16)


def _check(t: torch.Tensor, name: str) -> None:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if not t.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if t.dtype not in _ACTIVATION:
        raise TypeError(f"{name} must be float16 or bfloat16, got {t.dtype}")


def decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attend one query token to a grouped-query KV cache.

    ``q`` is ``[B, Hq, D]`` or ``[B, Hq, 1, D]``. ``k`` and ``v`` are
    ``[B, Hkv, S, D]``. ``Hq`` must be divisible by ``Hkv``.
    """
    _check(q, "q")
    _check(k, "k")
    _check(v, "v")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q, k, and v must have the same dtype")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    squeezed = False
    if q.ndim == 4:
        if q.shape[-2] != 1:
            raise ValueError("q must contain one token: [B, Hq, 1, D]")
        q = q.squeeze(-2)
        squeezed = True
    if q.ndim != 3 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("expected q [B, Hq, D] and k, v [B, Hkv, S, D]")
    if k.shape != v.shape:
        raise ValueError(f"k shape {tuple(k.shape)} != v shape {tuple(v.shape)}")
    b, hq, d = q.shape
    bk, hkv, _s, dk = k.shape
    if b != bk or d != dk:
        raise ValueError("batch and head dim of q must match k")
    if hkv == 0 or hq % hkv != 0:
        raise ValueError(f"Hq ({hq}) must be divisible by Hkv ({hkv})")
    if scale is None:
        scale = d**-0.5
    scale = float(scale)
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    expected = (b, hq, d)
    if out is None:
        out = torch.empty(expected, device=q.device, dtype=q.dtype)
    else:
        if squeezed:
            raise ValueError("pass out only with a rank-3 q")
        if tuple(out.shape) != expected or out.dtype != q.dtype or out.device != q.device:
            raise ValueError(f"out must be {expected} {q.dtype} on {q.device}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
    decode_attention_out(q, k, v, scale, out)
    if squeezed:
        return out[:, :, None, :]
    return out
