"""Single-token grouped-query attention, split across the KV cache."""

from decode_attn.ops import decode_attention
from decode_attn.sdpa import disable, enable, fast_calls

__all__ = ["decode_attention", "disable", "enable", "fast_calls"]
__version__ = "0.1.0"
