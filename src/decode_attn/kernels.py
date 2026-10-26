"""Split-KV decode attention.

One query token attends to a KV cache. Query heads that share a KV head are
updated in the same program, so that cache tile is read once. The cache is
split across programs; a second kernel merges the partial softmax statistics.

Math is the online softmax from Milakov and Gimelshein, the same recurrence
FlashAttention uses. Masked keys use a large negative score rather than -inf
so an empty tile does not produce NaN when the running max is still empty.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

def split_size() -> int:
    value = int(os.environ.get("DECODE_ATTN_SPLIT", "256"))
    if value < 16 or value & (value - 1):
        raise ValueError("DECODE_ATTN_SPLIT must be a power of two and at least 16")
    return value


def tile_size() -> int:
    value = int(os.environ.get("DECODE_ATTN_TILE", "64"))
    if value < 16 or value & (value - 1):
        raise ValueError("DECODE_ATTN_TILE must be a power of two and at least 16")
    return value


def launch_config(seq_len: int) -> tuple[int, int]:
    """Split and tile chosen on an RTX 4070 Laptop. Env vars override both."""
    if os.environ.get("DECODE_ATTN_SPLIT") or os.environ.get("DECODE_ATTN_TILE"):
        return split_size(), tile_size()
    if seq_len >= 8192:
        return 256, 64
    if seq_len >= 2048:
        return 512, 64
    if seq_len >= 512:
        return 128, 128
    return 64, 64


def _pow2(n: int) -> int:
    value = 1
    while value < n:
        value *= 2
    return value


def head_dim_padded(d: int) -> int:
    padded = _pow2(d)
    if padded > 256:
        raise ValueError(f"head dim {d} is above the supported maximum of 256")
    if padded < 16:
        padded = 16
    return padded


def group_padded(group: int) -> int:
    padded = _pow2(group)
    if padded > 16:
        raise ValueError(f"query group {group} is above the supported maximum of 16")
    return padded


@triton.jit
def _decode_partial(
    q_ptr,
    k_ptr,
    v_ptr,
    m_ptr,
    l_ptr,
    acc_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_mb,
    stride_mh,
    stride_ms,
    stride_lb,
    stride_lh,
    stride_ls,
    stride_ab,
    stride_ah,
    stride_as,
    stride_ad,
    Hkv,
    S,
    D,
    sm_scale,
    G: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SPLIT: tl.constexpr,
    TILE: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    batch = tl.program_id(0)
    kv_h = tl.program_id(1)
    split = tl.program_id(2)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_g = tl.arange(0, GROUP)
    d_mask = offs_d < D
    g_mask = offs_g < G
    start = split * SPLIT
    # Heads in a group share this KV head. Padding rows stay zero and are not stored.
    q = tl.load(
        q_ptr + batch * stride_qb + (kv_h * G + tl.minimum(offs_g, G - 1))[:, None] * stride_qh + offs_d[None, :] * stride_qd,
        mask=g_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    acc = tl.zeros((GROUP, HEAD_DIM), dtype=tl.float32)
    m_i = tl.full((GROUP,), -1.0e30, dtype=tl.float32)
    l_i = tl.zeros((GROUP,), dtype=tl.float32)
    k_base = batch * stride_kb + kv_h * stride_kh
    v_base = batch * stride_vb + kv_h * stride_vh

    for tile in tl.range(0, SPLIT, TILE):
        offs_s = start + tile + tl.arange(0, TILE)
        s_mask = offs_s < S
        k = tl.load(
            k_ptr + k_base + offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd,
            mask=s_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        v = tl.load(
            v_ptr + v_base + offs_s[:, None] * stride_vs + offs_d[None, :] * stride_vd,
            mask=s_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        scores = tl.where(s_mask[None, :], scores, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        prob = tl.where(s_mask[None, :], tl.exp(scores - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(prob, axis=1)
        prob_mv = prob.to(tl.bfloat16) if OUT_BF16 else prob.to(tl.float16)
        acc = acc * alpha[:, None] + tl.dot(prob_mv, v, out_dtype=tl.float32)
        m_i = m_new

    for gi in tl.static_range(GROUP):
        if gi < G:
            hq = kv_h * G + gi
            sel = offs_g == gi
            m_g = tl.sum(tl.where(sel, m_i, 0))
            l_g = tl.sum(tl.where(sel, l_i, 0))
            acc_g = tl.sum(tl.where(sel[:, None], acc, 0), axis=0)
            tl.store(m_ptr + batch * stride_mb + hq * stride_mh + split * stride_ms, m_g)
            tl.store(l_ptr + batch * stride_lb + hq * stride_lh + split * stride_ls, l_g)
            tl.store(
                acc_ptr + batch * stride_ab + hq * stride_ah + split * stride_as + offs_d * stride_ad,
                acc_g,
                mask=d_mask,
            )


@triton.jit
def _decode_reduce(
    m_ptr,
    l_ptr,
    acc_ptr,
    out_ptr,
    stride_mb,
    stride_mh,
    stride_ms,
    stride_lb,
    stride_lh,
    stride_ls,
    stride_ab,
    stride_ah,
    stride_as,
    stride_ad,
    stride_ob,
    stride_oh,
    stride_od,
    n_splits,
    D,
    HEAD_DIM: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    batch = tl.program_id(0)
    hq = tl.program_id(1)
    offs_d = tl.arange(0, HEAD_DIM)
    d_mask = offs_d < D
    m = -1.0e30
    for split in tl.range(0, n_splits):
        ms = tl.load(m_ptr + batch * stride_mb + hq * stride_mh + split * stride_ms)
        m = tl.maximum(m, ms)
    l = 0.0
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for split in tl.range(0, n_splits):
        base_m = batch * stride_mb + hq * stride_mh + split * stride_ms
        ms = tl.load(m_ptr + base_m)
        ls = tl.load(l_ptr + batch * stride_lb + hq * stride_lh + split * stride_ls)
        partial = tl.load(
            acc_ptr + batch * stride_ab + hq * stride_ah + split * stride_as + offs_d * stride_ad,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        alpha = tl.exp(ms - m)
        l += ls * alpha
        acc += partial * alpha
    out = tl.where(l > 0, acc / l, 0.0)
    stored = out.to(tl.bfloat16) if OUT_BF16 else out.to(tl.float16)
    tl.store(out_ptr + batch * stride_ob + hq * stride_oh + offs_d * stride_od, stored, mask=d_mask)


_WORK = {}


def _workspace(b: int, hq: int, n_splits: int, d: int, device: torch.device):
    key = (device.type, device.index, b, hq, n_splits, d)
    found = _WORK.get(key)
    if found is None:
        found = (
            torch.empty(b, hq, n_splits, device=device, dtype=torch.float32),
            torch.empty(b, hq, n_splits, device=device, dtype=torch.float32),
            torch.empty(b, hq, n_splits, d, device=device, dtype=torch.float32),
        )
        _WORK[key] = found
    return found


def decode_attention_out(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    out: torch.Tensor,
) -> None:
    """Write attention into ``out``. Tensors are [B, Hq, D], [B, Hkv, S, D], [B, Hkv, S, D]."""
    b, hq, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    if s == 0 or b == 0 or hq == 0:
        out.zero_()
        return
    group = hq // hkv
    pad = group_padded(group)
    hd = head_dim_padded(d)
    split, tile = launch_config(s)
    if tile > split:
        raise ValueError("DECODE_ATTN_TILE cannot exceed DECODE_ATTN_SPLIT")
    n_splits = triton.cdiv(s, split)
    m_part, l_part, acc_part = _workspace(b, hq, n_splits, d, q.device)
    _decode_partial[(b, hkv, n_splits)](
        q,
        k,
        v,
        m_part,
        l_part,
        acc_part,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        m_part.stride(0),
        m_part.stride(1),
        m_part.stride(2),
        l_part.stride(0),
        l_part.stride(1),
        l_part.stride(2),
        acc_part.stride(0),
        acc_part.stride(1),
        acc_part.stride(2),
        acc_part.stride(3),
        hkv,
        s,
        d,
        scale,
        G=group,
        GROUP=pad,
        HEAD_DIM=hd,
        SPLIT=split,
        TILE=tile,
        OUT_BF16=q.dtype == torch.bfloat16,
        num_warps=4,
        num_stages=3,
    )
    _decode_reduce[(b, hq)](
        m_part,
        l_part,
        acc_part,
        out,
        m_part.stride(0),
        m_part.stride(1),
        m_part.stride(2),
        l_part.stride(0),
        l_part.stride(1),
        l_part.stride(2),
        acc_part.stride(0),
        acc_part.stride(1),
        acc_part.stride(2),
        acc_part.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        n_splits,
        d,
        HEAD_DIM=hd,
        OUT_BF16=q.dtype == torch.bfloat16,
        num_warps=4,
    )
