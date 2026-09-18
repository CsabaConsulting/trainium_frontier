"""Opt-in scratch boundary for the native NKI SDPA path.

FP32 queries select FP32 matrix operands in the installed attention backward,
even under compiler BF16 matmult autocast. Cast Q/K/V together immediately before
SDPA, retaining its FP32 statistics and returning the original query dtype.
This is an intentional precision change, not an exact-math kernel replacement.
"""
import torch
import torch.nn.functional as F


_SDPA = F.scaled_dot_product_attention


def bf16_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
              is_causal=False, *, scale=None, enable_gqa=False):
    original_dtype = query.dtype
    out = _SDPA(query.to(torch.bfloat16), key.to(torch.bfloat16),
                value.to(torch.bfloat16), attn_mask=attn_mask,
                dropout_p=dropout_p, is_causal=is_causal,
                scale=scale, enable_gqa=enable_gqa)
    return out.to(original_dtype)
