#!/usr/bin/env python3
"""Measure bounded exp approximations, including normalized loss sensitivity.

The probability/gradient diagnostic isolates exponential error using FP64
normalization on the CPU; it is not a device softmax or end-to-end gradient
gate. Any promising candidate still needs fused-kernel and training checks.
"""
import argparse
import json
import math

import torch

from exp_polynomial_nki import build_kernel
from mlp_fused_bf16_probe import timing

NAMES = {0: "scalar_exp", 1: "vector_polynomial5", 2: "vector_polynomial7"}
LIMITS = {0: None, 1: 2.5e-5, 2: 2e-6}


def emit(kind, record):
    print(kind + " " + json.dumps(record), flush=True)


def check(kernel, x, name, eligible):
    reference = x.double().exp()
    device = x.to("neuron")
    for method, title in NAMES.items():
        out = kernel[2](device, method, 2).cpu()
        repeated = kernel[2](device, method, 2).cpu()
        relative = (out.double() / reference - 1).abs()
        index = int(relative.flatten().argmax())
        finite = bool(torch.isfinite(out).all())
        positive = bool((out > 0).all())
        exact_repeat = torch.equal(out, repeated)
        zero = x == 0
        zero_exact = not bool(zero.any()) or bool((out[zero] == 1).all())
        accuracy = LIMITS[method] is None or float(relative.max()) <= LIMITS[method]
        passed = finite and positive and exact_repeat and zero_exact and accuracy
        eligible[method] = eligible[method] and passed
        emit("EXP_POLYNOMIAL_GATE", dict(
            case=name, method=title, shape=list(x.shape), passed=passed,
            finite=finite, positive=positive, repeated_exact=exact_repeat,
            zero_exact=zero_exact, relative_error_limit=LIMITS[method],
            max_relative_error=float(relative.max()),
            relative_rms=float((out.double() - reference).norm() / reference.norm()),
            worst=dict(x=float(x.flatten()[index]), got=float(out.flatten()[index]),
                       expected=float(reference.flatten()[index])),
            native_accuracy_is_diagnostic=method == 0))
        if name == "loss_patterns":
            # Four actual loss patterns, repeated to fill a legal kernel tile.
            n, vocab = 256, 8192
            e = out.double().reshape(n, vocab)
            truth = reference.reshape(n, vocab)
            probability = e / e.sum(dim=1, keepdim=True)
            expected_probability = truth / truth.sum(dim=1, keepdim=True)
            labels = torch.arange(n) % vocab
            loss = -probability[torch.arange(n), labels].log()
            expected_loss = -expected_probability[torch.arange(n), labels].log()
            gradient = probability.clone()
            expected_gradient = expected_probability.clone()
            gradient[torch.arange(n), labels] -= 1
            expected_gradient[torch.arange(n), labels] -= 1
            emit("EXP_LOSS_SENSITIVITY", dict(
                method=title, normalization_dtype="float64 CPU (isolates exp error)",
                max_probability_abs_error=float((probability - expected_probability).abs().max()),
                max_loss_abs_error=float((loss - expected_loss).abs().max()),
                gradient_relative_rms=float((gradient - expected_gradient).norm()
                                            / expected_gradient.norm()),
                fused_device_or_backward_gate=False))
    assert torch.equal(device.cpu(), x), "input mutated"


def boundary_inputs(rows, cols):
    x = torch.linspace(-30, 30, rows * cols, dtype=torch.float32)
    centers = torch.tensor([k * math.log(2) for k in range(-43, 44)], dtype=torch.float32)
    edges = torch.cat((centers, torch.nextafter(centers, torch.full_like(centers, -math.inf)),
                       torch.nextafter(centers, torch.full_like(centers, math.inf)),
                       torch.tensor([0., -0., 1e-20, -1e-20, 1e-7, -1e-7, -30., 30.])))
    x[:edges.numel()] = edges
    return x.reshape(rows, cols)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-shape", action="store_true")
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()
    import torch_neuronx
    torch.set_num_threads(4)
    kernel = torch_neuronx.wrap_nki(build_kernel())
    eligible = {method: True for method in NAMES}
    emit("EXP_POLYNOMIAL_CONTRACT", dict(
        domain=[-30, 30], coefficient_source="Chebyshev interpolation; c0 constrained to1",
        relative_error_limits=LIMITS, exact_exp_zero_required=True,
        scratch_only=True, neuron_logical_cores=2))
    with torch.no_grad():
        check(kernel, boundary_inputs(256, 2048), "range_and_exponent_boundaries", eligible)
        x = torch.linspace(-30, 30, 8192).repeat(256, 1)
        labels = torch.arange(256) % 8192
        for row in range(256):
            kind = row % 4
            if kind == 1:
                x[row].fill_(-15.)
                x[row, labels[row]] = 0.
            elif kind == 2:
                x[row].zero_()
                x[row, labels[row]] = -30.
            elif kind == 3:
                x[row].zero_()
        check(kernel, x.reshape(1024, 2048), "loss_patterns", eligible)
        if args.real_shape:
            x = boundary_inputs(4096, 2048)
            check(kernel, x, "real_8m_shape", eligible)
            device = x.to("neuron")
            arms = {NAMES[m]: (lambda method=m: kernel[2](device, method, 2))
                    for m, passed in eligible.items() if passed}
            measured = timing(arms, args.steps, torch.neuron.synchronize)
            emit("EXP_POLYNOMIAL_BENCH", dict(
                shape=list(x.shape), **measured, hbm_load_and_store_included=True,
                whole_loss_or_training_step=False, eligible=eligible))
    emit("EXP_POLYNOMIAL_RESULT", dict(eligible=eligible, integrated_into_training=False))
    assert eligible[0], "native structural control failed"


if __name__ == "__main__":
    main()
