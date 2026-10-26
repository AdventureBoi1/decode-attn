"""Benchmark decode attention against PyTorch scaled_dot_product_attention.

Times are CUDA-event medians. Warmup is excluded. GB/s counts one read of K,
one read of V, and the write of the output, which is the traffic of a kernel
that reuses a KV head across the query group.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from decode_attn import decode_attention
from decode_attn.reference import decode_attention_reference


@triton.jit
def _copy_kernel(dst, src, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst + offs, tl.load(src + offs, mask=mask, other=0), mask=mask)


def _copy_gbs() -> float:
    n = 32 * 1024 * 1024
    src = torch.empty(n, device="cuda", dtype=torch.float16)
    dst = torch.empty_like(src)
    block = 4096
    grid = (triton.cdiv(n, block),)

    def run():
        _copy_kernel[grid](dst, src, n, BLOCK=block, num_warps=8)

    us = time_us(run, iters=20, warmup=5, rounds=3)
    return (2 * n * 2) / (us * 1e-6) / 1e9


def _sdpa_backends() -> dict:
    found = {}
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return {"import": "torch.nn.attention.sdpa_kernel is unavailable"}
    q = torch.randn(1, 32, 1, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 8, 1024, 128, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    names = [name for name in dir(SDPBackend) if name.isupper()]
    for name in names:
        backend = getattr(SDPBackend, name)
        try:
            with sdpa_kernel(backend):
                F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
            torch.cuda.synchronize()
            found[name] = "ok"
        except Exception as exc:
            found[name] = f"{type(exc).__name__}: {exc}"
    return found


def time_us(fn, iters: int = 30, warmup: int = 10, rounds: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    samples = []
    for _ in range(rounds):
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters * 1000.0)
    return statistics.median(samples)


def _sdpa(q, k, v, scale, backend=None):
    q4 = q[:, :, None, :]
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        sdpa_kernel = None
        SDPBackend = None

    def run():
        return F.scaled_dot_product_attention(q4, k, v, scale=scale, enable_gqa=True)

    if backend is None or sdpa_kernel is None:
        return run()
    chosen = getattr(SDPBackend, backend)
    with sdpa_kernel(chosen):
        return run()


def _bytes(k, v, out) -> int:
    return k.nbytes + v.nbytes + out.nbytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark decode attention")
    parser.add_argument("--out", type=Path, default=Path("results/latest.json"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(0)
    lengths = [1024, 4096] if args.quick else [128, 1024, 4096, 16384]
    dtypes = [torch.float16] if args.quick else [torch.float16, torch.bfloat16]
    rows = []
    gpu = torch.cuda.get_device_name(0)
    print(gpu, torch.__version__)
    copy_gbs = _copy_gbs()
    backends = _sdpa_backends()
    print(f"copy roof {copy_gbs:.1f} GB/s")
    print("sdpa backends", backends)
    for dtype in dtypes:
        for s in lengths:
            b, hq, hkv, d = 1, 32, 8, 128
            q = torch.randn(b, hq, d, device="cuda", dtype=dtype)
            k = torch.randn(b, hkv, s, d, device="cuda", dtype=dtype)
            v = torch.randn(b, hkv, s, d, device="cuda", dtype=dtype)
            scale = d**-0.5
            ours = time_us(lambda q=q, k=k, v=v, scale=scale: decode_attention(q, k, v, scale))
            try:
                eager = time_us(lambda q=q, k=k, v=v, scale=scale: _sdpa(q, k, v, scale))
            except Exception as exc:
                eager = None
                print("sdpa failed", exc)
            try:
                math_us = time_us(lambda q=q, k=k, v=v, scale=scale: _sdpa(q, k, v, scale, "MATH"))
            except Exception as exc:
                math_us = None
                print("math sdpa failed", exc)
            out = decode_attention(q, k, v, scale)
            ref = decode_attention_reference(q, k, v, scale)
            err = (out.float() - ref.float()).abs().max().item()
            nbytes = _bytes(k, v, out)
            row = {
                "name": f"S={s} {dtype}".replace("torch.", ""),
                "us": ours,
                "sdpa_us": eager,
                "math_us": math_us,
                "speedup_vs_sdpa": (eager / ours) if eager else None,
                "speedup_vs_math": (math_us / ours) if math_us else None,
                "gbs": nbytes / (ours * 1e-6) / 1e9,
                "pct_copy_roof": (nbytes / (ours * 1e-6) / 1e9) / copy_gbs * 100.0,
                "max_abs_vs_fp32": err,
                "bytes": nbytes,
            }
            rows.append(row)
            sdpa_s = f"{eager:.1f}" if eager else "fail"
            math_s = f"{math_us:.1f}" if math_us else "fail"
            print(
                f"{row['name']}: ours {ours:.1f} us  sdpa {sdpa_s}  math {math_s}  "
                f"abs {err:.3e}  {row['gbs']:.1f} GB/s"
            )
    payload = {
        "gpu": gpu,
        "torch": torch.__version__,
        "copy_gbs": copy_gbs,
        "sdpa_backends": backends,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
