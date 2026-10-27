"""Route single-token CUDA SDPA calls through the decode kernel.

Call ``enable`` before importing ``scaled_dot_product_attention`` under
another name. Longer prompts, masks, and dropout stay on PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from decode_attn.ops import decode_attention

_original = None
_fast_calls = 0
_DTYPES = (torch.float16, torch.bfloat16)
# Against nightly 2.16.0.dev20261026+cu132 on an RTX 4070 Laptop, this kernel
# was slower at 512, 1024, and 2048, faster at 4096 and 8192, and tied at 16384.
_FLASH_WIN_MIN = 4096
_FLASH_WIN_MAX = 8192


def flash_attention_available() -> bool:
    probe = getattr(torch.backends.cuda, "is_flash_attention_available", None)
    if probe is None:
        return False
    try:
        return bool(probe())
    except Exception:
        return False


def should_route(seq_len: int, flash_available: bool) -> bool:
    """Use this kernel when it beat PyTorch on the measured 4070."""
    if seq_len < 1:
        return False
    if not flash_available:
        return True
    return _FLASH_WIN_MIN <= seq_len <= _FLASH_WIN_MAX


def fast_calls() -> int:
    return _fast_calls


def _route(*args, **kwargs):
    global _fast_calls
    if len(args) < 3:
        return None
    query, key, value = args[0], args[1], args[2]
    attn_mask = kwargs.get("attn_mask", args[3] if len(args) > 3 else None)
    dropout_p = kwargs.get("dropout_p", args[4] if len(args) > 4 else 0.0)
    is_causal = kwargs.get("is_causal", args[5] if len(args) > 5 else False)
    scale = kwargs.get("scale", None)
    enable_gqa = kwargs.get("enable_gqa", False)
    if attn_mask is not None or dropout_p not in (0, 0.0):
        return None
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return None
    if query.shape[-2] != 1:
        return None
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        return None
    if query.dtype not in _DTYPES or key.dtype != query.dtype or value.dtype != query.dtype:
        return None
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1] or key.shape != value.shape:
        return None
    hq, hkv = query.shape[1], key.shape[1]
    if hq != hkv and not (enable_gqa and hkv > 0 and hq % hkv == 0):
        return None
    if hq % hkv != 0:
        return None
    if not should_route(key.shape[-2], flash_attention_available()):
        return None
    # q_len == 1 with the bottom-right causal mask attends to every key.
    _ = is_causal
    out = decode_attention(query, key, value, scale=scale)
    _fast_calls += 1
    return out


def enable() -> None:
    """Replace ``F.scaled_dot_product_attention`` for single-token CUDA calls."""
    global _original
    current = F.scaled_dot_product_attention
    if getattr(current, "_decode_attn", False):
        return
    _original = current

    def wrapped(*args, **kwargs):
        try:
            routed = _route(*args, **kwargs)
        except Exception:
            routed = None
        if routed is None:
            return _original(*args, **kwargs)
        return routed

    wrapped._decode_attn = True
    F.scaled_dot_product_attention = wrapped


def disable() -> None:
    global _original
    if _original is None:
        return
    F.scaled_dot_product_attention = _original
    _original = None
