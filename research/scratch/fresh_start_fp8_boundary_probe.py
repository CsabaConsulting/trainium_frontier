#!/usr/bin/env python3
"""Distinguish FP16 precision/range hypotheses with exact FP8 operands.

No benchmark and no training. These patterns include an overflow discriminator:
FP32 accumulation does not imply the preceding two-product adder has FP32 range.
Record actual behavior; do not silently accept a nonfinite output for training.
"""
import json

import torch
import torch_neuronx  # noqa: F401

from fresh_start_fp8_probe import build_kernels


def main():
    torch.set_num_threads(2)
    fp8, _, _ = build_kernels()
    scales = torch.ones((128, 3), dtype=torch.float32, device="neuron")
    # Every operand below is exactly representable in legacy E4M3 and BF16.
    patterns = (
        ("unit", (1.0, 1.0), (1.0, 1.0)),
        ("min_subnormal", (2.0 ** -9, 2.0 ** -9), (2.0 ** -9, 2.0 ** -9)),
        ("fp16_halfway_even", (1.0, 2.0 ** -6), (1.0, 2.0 ** -5)),
        ("large_plus_small", (128.0, 1.0), (128.0, 1.0)),
        ("max_cancel", (240.0, 240.0), (240.0, -240.0)),
        ("max_same_sign", (240.0, 240.0), (240.0, 240.0)),
    )
    for name, av, bv in patterns:
        a = torch.tensor(av).repeat(128).reshape(1, 256).repeat(128, 1).bfloat16()
        b = torch.tensor(bv).repeat(128).reshape(256, 1).repeat(1, 128).bfloat16()
        y = fp8[1](a.T.contiguous().to("neuron"), b.to("neuron"), scales, 1, True).cpu()
        ideal_pair = av[0] * bv[0] + av[1] * bv[1]
        fp16_pair = float(torch.tensor(ideal_pair).half())
        print(json.dumps(dict(
            check="fp8_pair_boundary", case=name, operands_a=av, operands_b=bv,
            ideal_dot=ideal_pair * 128,
            fp16_pair_model=str(fp16_pair * 128),
            actual_unique=[str(float(v)) for v in y.unique()],
            finite_elements=int(torch.isfinite(y).sum()), elements=y.numel(),
        )), flush=True)
    print("COMPLETED: boundary observations only; no speed or training claim", flush=True)


if __name__ == "__main__":
    main()
