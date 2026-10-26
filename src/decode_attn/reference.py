"""fp32 attention reference. KV heads are repeated to match query heads."""

from __future__ import annotations

import torch


def decode_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    if q.ndim == 4:
        if q.shape[-2] != 1:
            raise ValueError("reference expects one query token")
        q = q.squeeze(-2)
    b, hq, d = q.shape
    hkv = k.shape[1]
    if hq % hkv != 0:
        raise ValueError("Hq must be divisible by Hkv")
    group = hq // hkv
    if scale is None:
        scale = d**-0.5
    k_full = k.float().repeat_interleave(group, dim=1)
    v_full = v.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("bhd,bhsd->bhs", q.float(), k_full) * scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhs,bhsd->bhd", probs, v_full)
    return out.to(dtype=q.dtype)
