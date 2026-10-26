"""Single-token grouped-query attention, split across the KV cache."""

from decode_attn.ops import decode_attention

__all__ = ["decode_attention"]
__version__ = "0.1.0"
