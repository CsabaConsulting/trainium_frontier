"""Experimental normalization policy; not enabled in the training entrypoint.

Retain FP32 residual accumulation and FP32 RMS statistics/epsilon, but store
normalized branch inputs in BF16. This intentionally changes rounding, including
the final head input. It also lets Linear consume its existing BF16 weight cache.
No cross-token statistic or dynamic scale is introduced.
"""
import torch
import torch.nn.functional as F


FP32_EPS = 1.1920928955078125e-7


def bf16_branch_norm(x):
    return F.rms_norm(x.float(), (x.shape[-1],), eps=FP32_EPS).to(torch.bfloat16)
