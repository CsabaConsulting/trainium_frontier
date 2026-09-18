#!/usr/bin/env python3
"""Test explicit accumulation hypotheses against saved native FP8 outputs.

This is CPU-only. It distinguishes ordinary FP32 accumulation error from a
possible rounded two-product partial inside double-row matmul. Hypothesis fits
are empirical evidence for these operands, not a hardware specification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def errors(actual, reference):
    delta = actual.double() - reference.double()
    return dict(max_abs=float(delta.abs().max()),
                relative_rms=float(delta.square().mean().sqrt()
                                   / reference.double().square().mean().sqrt().clamp_min(1e-300)),
                exact_elements=int((actual.double() == reference.double()).sum()),
                elements=actual.numel())


def analyze(path):
    case = torch.load(path, map_location="cpu", weights_only=True)
    a, b = case["aq"].double(), case["bq"].double()
    measured = case["y8_float"]
    inverse = 1.0 / (case["scale_a"] * case["scale_b"])
    k, n = b.shape
    exact = a @ b * inverse
    rows = [dict(hypothesis="full_dot_fp32_output", **errors(measured, exact.float()))]
    # Process modest row chunks: the largest saved shape must not create a
    # multi-billion-element product tensor. The products themselves are exact
    # FP64 representations of the dyadic FP8 inputs.
    for dtype in (torch.float16, torch.bfloat16):
        for group in (2, 4, 8, 16):
            if k % group:
                continue
            predicted = torch.empty_like(exact)
            for start in range(0, a.shape[0], 8):
                aa = a[start:start + 8]
                products = aa[:, :, None] * b[None, :, :]
                partials = products.reshape(aa.shape[0], k // group, group, n).sum(2)
                predicted[start:start + len(aa)] = partials.to(dtype).double().sum(1) * inverse
            rows.append(dict(hypothesis=f"adjacent_{group}_product_sum_round_{dtype}",
                             measured_vs_prediction=errors(measured, predicted.float()),
                             prediction_vs_exact=errors(predicted, exact)))
    return dict(file=str(path), shape=[a.shape[0], k, n],
                measured_vs_exact=errors(measured, exact), hypotheses=rows,
                hardware_specification_claim=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = [analyze(path) for path in args.cases]
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
