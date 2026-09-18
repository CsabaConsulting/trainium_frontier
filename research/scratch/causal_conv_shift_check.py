#!/usr/bin/env python3
"""Check static causal-convolution lowering against independent torch conv1d.

Random nonzero weights and arbitrary upstream gradients exercise the learned
branch. Cases include noncontiguous inputs, short sequences, multiple batches,
and future/batch perturbations. This does not validate Neuron code generation.
"""
import json
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from causal_conv_shift import shift_causal_conv


def reference(x, w):
    y = F.conv1d(F.pad(x.transpose(1, 2), (w.shape[-1] - 1, 0)),
                 w.to(x.dtype), groups=x.shape[-1])
    return y.transpose(1, 2)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(291)
    results = []
    for dtype in [torch.float32, torch.bfloat16]:
        for batch, length, width, taps in [(1, 17, 7, 1), (2, 17, 7, 2),
                                          (2, 128, 128, 2), (2, 17, 7, 4), (2, 2, 7, 4)]:
            x = torch.randn(batch, width, length).transpose(1, 2).to(dtype)
            w = torch.randn(width, 1, taps) * 0.1
            grad = torch.randn_like(x)
            outputs = []
            for fn in [reference, lambda a, b: shift_causal_conv(SimpleNamespace(weight=b, kernel_size=taps), a)]:
                xx = x.detach().clone().requires_grad_()
                ww = w.detach().clone().requires_grad_()
                y = fn(xx, ww)
                dx, dw = torch.autograd.grad(y, (xx, ww), grad)
                outputs.append((y.detach(), dx, dw))
            # CPU BF16 conv1d backward is not an exact arithmetic oracle: it can
            # round intermediate gradients earlier than the static FP32 spelling.
            # Compare both against an independent FP64 convolution/gradient with
            # the SAME rounded operands. This changes the reference explicitly,
            # not the tolerance of a failed comparison to the old implementation.
            xd = x.double().requires_grad_()
            wd = w.to(dtype).double().requires_grad_()
            yd = reference(xd, wd)
            dxd, dwd = torch.autograd.grad(yd, (xd, wd), grad.double())
            expected = (yd.to(dtype), dxd.to(dtype), dwd.to(dtype).float())
            errors = []
            for actual, gold in zip(outputs[1], expected):
                torch.testing.assert_close(actual, gold,
                                           rtol=0 if dtype == torch.bfloat16 else 3e-5,
                                           atol=0 if dtype == torch.bfloat16 else 1e-5)
                errors.append(float((actual.detach().float() - gold.detach().float()).abs().max()))
            legacy_errors = [float((a.float() - b.detach().float()).abs().max())
                             for a, b in zip(outputs[0], expected)]
            module = SimpleNamespace(weight=w, kernel_size=taps)
            cut = length // 2
            changed = x.clone()
            changed[:, cut:] += 100
            torch.testing.assert_close(shift_causal_conv(module, changed)[:, :cut],
                                       outputs[1][0][:, :cut], rtol=0, atol=0)
            if batch > 1:
                changed = x.clone()
                changed[0] += 100
                torch.testing.assert_close(shift_causal_conv(module, changed)[1:],
                                           outputs[1][0][1:], rtol=0, atol=0)
            results.append(dict(dtype=str(dtype), shape=list(x.shape), taps=taps,
                                reference="FP64 convolution with input-dtype operands, final dtype rounding",
                                shift_max_abs_forward_dx_dw=errors,
                                legacy_max_abs_forward_dx_dw=legacy_errors))
    print(json.dumps(dict(cases=results, passed=True)))


if __name__ == "__main__":
    main()
