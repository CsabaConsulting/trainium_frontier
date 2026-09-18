"""Static slice/elementwise spelling of the existing depthwise causal convolution.

This is a diagnostic lowering variant, not a new architecture. Input and taps
are rounded to the same input dtype as CausalConv; products/sums use FP32 before
the output cast. The existing parameter and its optimizer group are unchanged.
"""
import torch
import torch.nn.functional as F


def shift_causal_conv(self, x):
    length = x.shape[1]
    xf = x.float()
    taps = self.weight.to(dtype=x.dtype).float()
    out = torch.zeros_like(xf)
    for k in range(self.kernel_size):
        shift = self.kernel_size - 1 - k
        if shift >= length:
            continue
        source = F.pad(xf[:, :length - shift, :], (0, 0, shift, 0)) if shift else xf
        out = out + source * taps[:, 0, k]
    return out.to(x.dtype)
