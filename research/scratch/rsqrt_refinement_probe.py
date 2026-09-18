#!/usr/bin/env python3
"""Expose Scalar rsqrt seed and Newton stages without changing training.

This is a precision diagnostic. Extra HBM stores deliberately expose stages;
its runtime must not be compared with a production AdamW kernel. Inputs cover
the bounded variance contract of AdamW math mode4, including its failures.
"""
import json
import math

import torch


STAGES = ("scaled_scalar_seed", "seed_squared", "variance_times_seed_squared",
          "one_newton", "two_newton", "gpsimd_rsqrt", "scalar_sqrt")


def build_kernel():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def diagnose(variance, lanes: int = 2):
        """Return seven FP32 intermediate arrays for positive FP32 variances.

        Args:
            variance: Contiguous [N, F], within [2**-116, 2**64].
            lanes: Matching LNC2 grid size (2).
        Returns:
            Scalar seed, its square/product, two corrections and controls.
        Notes:
            The seed scaling and first correction match AdamW math mode4.
            Scalar activation rsqrt is supported by the installed NKI0.5.
        """
        rows, cols = variance.shape
        assert variance.dtype == nl.float32 and cols <= 2048
        assert lanes == 2 and rows % (128 * lanes) == 0
        seed = nl.ndarray(variance.shape, dtype=nl.float32, buffer=nl.shared_hbm)
        square = nl.ndarray(variance.shape, dtype=nl.float32, buffer=nl.shared_hbm)
        product = nl.ndarray(variance.shape, dtype=nl.float32, buffer=nl.shared_hbm)
        first = nl.ndarray(variance.shape, dtype=nl.float32, buffer=nl.shared_hbm)
        second = nl.ndarray(variance.shape, dtype=nl.float32, buffer=nl.shared_hbm)
        gpsimd = nl.ndarray(variance.shape, dtype=nl.float32, buffer=nl.shared_hbm)
        scalar_sqrt = nl.ndarray(variance.shape, dtype=nl.float32, buffer=nl.shared_hbm)
        for tile in nl.affine_range(rows // (128 * lanes)):
            start = (tile * lanes + nl.program_id(0)) * 128
            source = nl.ndarray((128, cols), dtype=nl.float32, buffer=nl.sbuf)
            estimate = nl.ndarray((128, cols), dtype=nl.float32, buffer=nl.sbuf)
            work = nl.ndarray((128, cols), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=source, src=variance[start:start + 128, :])
            nisa.activation(dst=estimate, op=nl.rsqrt, data=source,
                            scale=4294967296.0)
            nisa.tensor_scalar(dst=estimate, data=estimate, op0=nl.multiply,
                               operand0=65536.0)
            nisa.dma_copy(dst=seed[start:start + 128, :], src=estimate)
            nisa.tensor_tensor(dst=work, data1=estimate, data2=estimate, op=nl.multiply)
            nisa.dma_copy(dst=square[start:start + 128, :], src=work)
            nisa.tensor_tensor(dst=work, data1=source, data2=work, op=nl.multiply)
            nisa.dma_copy(dst=product[start:start + 128, :], src=work)
            nisa.tensor_scalar(dst=work, data=work, op0=nl.multiply,
                               operand0=-0.5, op1=nl.add, operand1=1.5)
            nisa.tensor_tensor(dst=estimate, data1=estimate, data2=work, op=nl.multiply)
            nisa.dma_copy(dst=first[start:start + 128, :], src=estimate)
            nisa.tensor_tensor(dst=work, data1=estimate, data2=estimate, op=nl.multiply)
            nisa.tensor_tensor(dst=work, data1=source, data2=work, op=nl.multiply)
            nisa.tensor_scalar(dst=work, data=work, op0=nl.multiply,
                               operand0=-0.5, op1=nl.add, operand1=1.5)
            nisa.tensor_tensor(dst=estimate, data1=estimate, data2=work, op=nl.multiply)
            nisa.dma_copy(dst=second[start:start + 128, :], src=estimate)
            nisa.tensor_scalar(dst=work, data=source, op0=nl.rsqrt, operand0=0.0)
            nisa.dma_copy(dst=gpsimd[start:start + 128, :], src=work)
            nisa.activation(dst=work, op=nl.sqrt, data=source)
            nisa.dma_copy(dst=scalar_sqrt[start:start + 128, :], src=work)
        return seed, square, product, first, second, gpsimd, scalar_sqrt

    return diagnose


def emit(kind, record):
    print(kind + " " + json.dumps(record), flush=True)


def main():
    import torch_neuronx
    torch.set_num_threads(4)
    # All 180 exponents in the contract, with 4096 mantissas per exponent.
    fractions = 1 + torch.arange(4096, dtype=torch.float32) / 4096
    exponents = torch.arange(-116, 64, dtype=torch.int32)
    grid = torch.ldexp(fractions[None, :], exponents[:, None]).flatten()
    # Variance estimates use CPU FP32 arithmetic to place the recorded failing
    # updates in the diagnostic. They are labelled as reconstructions, not as
    # a claim of bit-identical device moment accumulation.
    gradient = torch.tensor(-1194.599609375, dtype=torch.float32)
    variance = torch.tensor(1034481.4375, dtype=torch.float32)
    one_minus_beta2 = torch.tensor(.04, dtype=torch.float32)
    bias2 = torch.tensor(1 - .96 ** 17, dtype=torch.float32)
    large = torch.lerp(variance, gradient.square(), one_minus_beta2) / bias2
    tiny = torch.tensor(-7.873063978536265e-9, dtype=torch.float32).square()
    points = {
        "small_first_step_variance_cpu_fp32": tiny,
        "large_failure_variance_cpu_fp32": large,
        "contract_lower_endpoint": torch.tensor(2.0 ** -116),
        "contract_upper_endpoint": torch.tensor(2.0 ** 64),
        "unit_variance": torch.tensor(1.0),
    }
    cols = 1024
    count = math.ceil((grid.numel() + len(points)) / (256 * cols)) * 256 * cols
    source = torch.full((count,), 1., dtype=torch.float32)
    source[:grid.numel()] = grid
    indices = {}
    for offset, (name, value) in enumerate(points.items()):
        index = grid.numel() + offset
        source[index] = value
        indices[name] = index
    source = source.reshape(-1, cols)
    assert bool(((source >= 2.0 ** -116) & (source <= 2.0 ** 64)).all())
    device = source.to("neuron")
    kernel = torch_neuronx.wrap_nki(build_kernel())
    with torch.no_grad():
        outputs = tuple(x.cpu() for x in kernel[2](device, 2))
        repeated = tuple(x.cpu() for x in kernel[2](device, 2))
    assert torch.equal(device.cpu(), source), "diagnostic mutated its input"
    assert all(torch.equal(a, b) for a, b in zip(outputs, repeated)), "not repeatable"
    truth = source.double().rsqrt()
    seed = outputs[0].double()
    exact_first_from_seed = seed * (1.5 - .5 * source.double() * seed.square())
    for name, value in zip(STAGES, outputs):
        if name in ("seed_squared", "variance_times_seed_squared"):
            continue
        expected = source.double().sqrt() if name == "scalar_sqrt" else truth
        error = (value.double() / expected - 1).abs()
        index = int(error.flatten().argmax())
        record = dict(stage=name, count=source.numel(),
                      finite_count=int(torch.isfinite(value).sum()),
                      positive_count=int((value > 0).sum()),
                      max_relative_error=float(error.max()),
                      relative_rms=float((value.double() - expected).norm() / expected.norm()),
                      worst=dict(variance=float(source.flatten()[index]),
                                 scaled_argument=float(source.flatten()[index]) * 2.0 ** 32,
                                 actual=float(value.flatten()[index]),
                                 expected=float(expected.flatten()[index])),
                      numerical_range_diagnostic=True, production_timing=False)
        emit("RSQRT_STAGE_ACCURACY", record)
    drift = ((outputs[3].double() - exact_first_from_seed) / truth).abs()
    emit("RSQRT_NEWTON_ARITHMETIC", dict(
        max_relative_difference_from_fp64_recurrence=float(drift.max()),
        fp64_recurrence_max_relative_error=float((exact_first_from_seed / truth - 1).abs().max()),
        purpose="separate initial estimate error from correction arithmetic"))
    # Include each worst seed/correction input as well as the explicit failures.
    for stage in (0, 3, 4):
        index = int((outputs[stage].double() / truth - 1).abs().flatten().argmax())
        indices["worst_" + STAGES[stage]] = index
    for name, index in indices.items():
        x = float(source.flatten()[index])
        values = {stage: float(value.flatten()[index])
                  for stage, value in zip(STAGES, outputs)}
        emit("RSQRT_POINT", dict(name=name, variance=x, scaled_argument=x * 2.0 ** 32,
                                  fp64_rsqrt=x ** -.5, fp64_sqrt=x ** .5,
                                  **values))
    emit("RSQRT_DIAGNOSTIC_DONE", dict(repeated_exact=True, input_unchanged=True,
                                      training_changed=False, timed=False))


if __name__ == "__main__":
    main()
