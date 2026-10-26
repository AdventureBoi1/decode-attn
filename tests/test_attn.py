import pytest
import torch

from decode_attn import decode_attention
from decode_attn.reference import decode_attention_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _tol(dtype, s):
    if dtype == torch.float16:
        return {"rtol": 2e-2, "atol": 2e-2} if s <= 256 else {"rtol": 4e-2, "atol": 4e-2}
    return {"rtol": 4e-2, "atol": 4e-2}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "b,hq,hkv,s,d",
    [
        (1, 1, 1, 1, 16),
        (1, 4, 2, 17, 64),
        (2, 8, 2, 129, 64),
        (1, 8, 2, 128, 128),
        (1, 4, 4, 200, 32),
    ],
)
def test_matches_fp32_reference(dtype, b, hq, hkv, s, d):
    torch.manual_seed(0)
    q = torch.randn(b, hq, d, device="cuda", dtype=dtype)
    k = torch.randn(b, hkv, s, d, device="cuda", dtype=dtype)
    v = torch.randn(b, hkv, s, d, device="cuda", dtype=dtype)
    out = decode_attention(q, k, v)
    ref = decode_attention_reference(q, k, v)
    torch.testing.assert_close(out, ref, **_tol(dtype, s))


def test_rank4_query_and_strided_cache():
    torch.manual_seed(1)
    q = torch.randn(1, 4, 1, 32, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 40, 2, 32, device="cuda", dtype=torch.float16).transpose(1, 2)
    assert not k.is_contiguous()
    v = torch.randn(1, 2, 40, 32, device="cuda", dtype=torch.float16)
    out = decode_attention(q, k, v)
    ref = decode_attention_reference(q, k, v)
    assert out.shape == (1, 4, 1, 32)
    torch.testing.assert_close(out, ref[:, :, None, :], rtol=2e-2, atol=2e-2)


def test_rejected_inputs():
    q = torch.randn(1, 4, 16)
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    with pytest.raises(ValueError, match="CUDA"):
        decode_attention(q, k, v)
    q = torch.randn(1, 3, 16, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    with pytest.raises(ValueError, match="divisible"):
        decode_attention(q, k, v)
