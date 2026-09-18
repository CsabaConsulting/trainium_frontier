#!/usr/bin/env python3
"""Native Trainium2 FP8 double-row feasibility, with explicit scale/layout costs.

This is NOT an FP8 training implementation. It probes a forward-only island using
legacy E4M3, not OCP E4M3FN or MXFP8. Scalars use a fixed supplied calibration for
the repeated benchmark; finding amax online is additional unmeasured work. Both
NKI variants include the input transpose in the headline timings. CPU FP32 and an
independent legacy-E4M3 quantizer supply the numerical references.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch


def e4m3_legacy(x):
    """IEEE-style E4M3 round-to-nearest-even, finite input saturated to +/-240."""
    magnitude = x.abs().clamp(max=240.0)
    exponent = torch.floor(torch.log2(magnitude.clamp_min(2.0 ** -9)))
    spacing = torch.pow(2.0, exponent - 3).clamp_min(2.0 ** -9)
    rounded = (torch.round(magnitude / spacing) * spacing).clamp(max=240.0)
    return x.sign() * rounded


def build_kernels():
    global nisa, nl
    import nki
    import nki.isa as nisa
    import nki.language as nl

    @nki.jit
    def roundtrip_e4m3(x, scale):
        """Expose the actual device conversion, independently of matrix multiply."""
        rows, cols = x.shape
        assert rows % 128 == 0 and cols <= 8192
        out = nl.ndarray(x.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
        s = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=s, src=scale)
        for tile in nl.affine_range(rows // 128):
            v = nl.ndarray((128, cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            q = nl.ndarray((128, cols), dtype=nl.float8_e4m3, buffer=nl.sbuf)
            r = nl.ndarray((128, cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=v, src=x[tile * 128:(tile + 1) * 128, :])
            nisa.tensor_scalar(dst=q, data=v, op0=nl.multiply, operand0=s)
            nisa.tensor_copy(dst=r, src=q)
            nisa.dma_copy(dst=out[tile * 128:(tile + 1) * 128, :], src=r)
        return out

    @nki.jit
    def native_fp8_gemm(a_t, b, scales, lanes: int = 1, fp32_output: bool = False):
        """[K,M] BF16 and [K,N] BF16 -> [M,N] BF16 via scaled legacy FP8.

        M multiple128; K multiple256; N multiple128 and <=512 in this first probe.
        Scales [128,3] contain scale_A, scale_B, inverse_product, all runtime FP32.
        K is packed as [128,2] for each TensorEngine double-row instruction.
        """
        k, m = a_t.shape
        _, n = b.shape
        assert lanes in (1, 2) and m % (128 * lanes) == 0
        assert k % 256 == 0 and n % 128 == 0 and n <= 512
        a_view = a_t.reshape((k // 2, 2, m))
        b_view = b.reshape((k // 2, 2, n))
        output_dtype = nl.float32 if fp32_output else nl.bfloat16
        out = nl.ndarray((m, n), dtype=output_dtype, buffer=nl.shared_hbm)
        sa = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        sb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        inverse = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sa, src=scales[:, 0:1])
        nisa.dma_copy(dst=sb, src=scales[:, 1:2])
        nisa.dma_copy(dst=inverse, src=scales[:, 2:3])
        lane = nl.program_id(0)
        for tile in nl.affine_range(m // (128 * lanes)):
            mi = tile * lanes + lane
            acc = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.psum)
            for ki in nl.affine_range(k // 256):
                ab = nl.ndarray((128, 2, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
                bb = nl.ndarray((128, 2, n), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=ab, src=a_view[ki * 128:(ki + 1) * 128, :, mi * 128:(mi + 1) * 128])
                nisa.dma_copy(dst=bb, src=b_view[ki * 128:(ki + 1) * 128, :, :])
                aq = nl.ndarray((128, 2, 128), dtype=nl.float8_e4m3, buffer=nl.sbuf)
                bq = nl.ndarray((128, 2, n), dtype=nl.float8_e4m3, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=aq, data=ab, op0=nl.multiply, operand0=sa)
                nisa.tensor_scalar(dst=bq, data=bb, op0=nl.multiply, operand0=sb)
                nisa.nc_matmul(dst=acc, stationary=aq, moving=bq, accumulate=(ki > 0),
                               perf_mode=nisa.matmul_perf_mode.double_row)
            acc_sb = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=acc_sb, src=acc)
            result = nl.ndarray((128, n), dtype=output_dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=result, data=acc_sb, op0=nl.multiply, operand0=inverse)
            nisa.dma_copy(dst=out[mi * 128:(mi + 1) * 128, :], src=result)
        return out

    @nki.jit
    def nki_bf16_gemm(a_t, b, lanes: int = 1, fp32_output: bool = False):
        """Comparable simple BF16 tiling; same transposed input and output shape."""
        k, m = a_t.shape
        _, n = b.shape
        assert lanes in (1, 2) and m % (128 * lanes) == 0
        assert k % 128 == 0 and n % 128 == 0 and n <= 512
        output_dtype = nl.float32 if fp32_output else nl.bfloat16
        out = nl.ndarray((m, n), dtype=output_dtype, buffer=nl.shared_hbm)
        lane = nl.program_id(0)
        for tile in nl.affine_range(m // (128 * lanes)):
            mi = tile * lanes + lane
            acc = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.psum)
            for ki in nl.affine_range(k // 128):
                ab = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
                bb = nl.ndarray((128, n), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=ab, src=a_t[ki * 128:(ki + 1) * 128, mi * 128:(mi + 1) * 128])
                nisa.dma_copy(dst=bb, src=b[ki * 128:(ki + 1) * 128, :])
                nisa.nc_matmul(dst=acc, stationary=ab, moving=bb, accumulate=(ki > 0))
            result = nl.ndarray((128, n), dtype=output_dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=result, src=acc)
            nisa.dma_copy(dst=out[mi * 128:(mi + 1) * 128, :], src=result)
        return out

    return native_fp8_gemm, nki_bf16_gemm, roundtrip_e4m3


def sync():
    torch.neuron.synchronize()


def relative_rms(actual, reference):
    return float((actual.float() - reference.float()).square().mean().sqrt()
                 / reference.float().square().mean().sqrt().clamp_min(1e-20))


def accumulation_error(actual, a, b, inverse=1.0):
    """Compare FP32 output against FP64 dot products and a declared FP32 bound.

    FP8/BF16 products are exactly representable in FP32. The conservative
    sequential-accumulation bound is gamma_K * sum(abs(a*b)), u=2**-24.
    Power-of-two rescaling introduces no normal-range rounding. This diagnostic
    deliberately removes BF16 output rounding; a failed bound is not waived.
    """
    ad, bd = a.double(), b.double()
    reference = (ad @ bd) * inverse
    absolute_products = (ad.abs() @ bd.abs()) * abs(inverse)
    k = a.shape[1]
    gamma_k = (k * 2.0 ** -24) / (1.0 - k * 2.0 ** -24)
    # The reference itself sums in FP64. Include its conservative error bound.
    gamma64 = (k * 2.0 ** -53) / (1.0 - k * 2.0 ** -53)
    bound = (gamma_k + gamma64) * absolute_products
    error = (actual.double() - reference).abs()
    return dict(elements=actual.numel(), finite=bool(torch.isfinite(actual).all()),
                failures=int((error > bound).sum()), max_absolute_error=float(error.max()),
                max_error_over_bound=float((error / bound.clamp_min(1e-300)).max()),
                relative_rms=float(error.square().mean().sqrt()
                                   / reference.square().mean().sqrt().clamp_min(1e-300)),
                gamma_k=gamma_k,
                reference_dtype="float64", output_dtype=str(actual.dtype))


def run_case(fp8, bf16, roundtrip, shape, steps, diagnostic_only=False,
             check_accumulation=False, save_diagnostic_dir=None):
    m, k, n = shape
    torch.manual_seed(17 + k)
    a = (torch.randn(m, k) * 0.5).bfloat16()
    b = (torch.randn(k, n) * k ** -0.5).bfloat16()
    # Calibration is intentionally separate: amax overhead is NOT included below.
    # Powers of two make the dequantization reference unambiguous.
    sa = float(2.0 ** torch.floor(torch.log2(128.0 / a.float().abs().max())))
    sb = float(2.0 ** torch.floor(torch.log2(128.0 / b.float().abs().max())))
    scalars = torch.tensor([sa, sb, 1.0 / (sa * sb)]).repeat(128, 1).to("neuron")
    quant_reference = (e4m3_legacy(a.float() * sa) @ e4m3_legacy(b.float() * sb)) / (sa * sb)
    full_reference = a.float() @ b.float()
    ad, bd = a.to("neuron"), b.to("neuron")
    at = ad.T.contiguous()
    lanes = 2 if m % 256 == 0 else 1
    y8 = fp8[lanes](at, bd, scalars, lanes).cpu()
    y16 = bf16[lanes](at, bd, lanes).cpu()
    # Isolate conversion from layout/accumulation before interpreting tolerance.
    aq = roundtrip(at, scalars[:, 0:1].contiguous()).cpu().float().T
    bq = roundtrip(bd, scalars[:, 1:2].contiguous()).cpu().float()
    device_quant_reference = (aq @ bq) / (sa * sb)
    # A predeclared second model characterizes flush-to-zero behavior, if any.
    # This is a hypothesis until measured, not an assertion from documentation.
    # Do not relax the numerical gate merely because a discrepancy is small.
    aq_ftz = torch.where(aq.abs() < 2.0 ** -6, 0.0, aq)
    bq_ftz = torch.where(bq.abs() < 2.0 ** -6, 0.0, bq)
    ftz_reference = (aq_ftz @ bq_ftz) / (sa * sb)
    conversions = []
    for name, actual, expected, source in (
        ("a", aq, e4m3_legacy(a.float() * sa), a.float() * sa),
        ("b", bq, e4m3_legacy(b.float() * sb), b.float() * sb),
    ):
        mismatch = actual != expected
        where = mismatch.nonzero()[:8]
        examples = [dict(input=float(source[i, j]), actual=float(actual[i, j]),
                         expected=float(expected[i, j])) for i, j in where]
        expected_ftz = torch.where(expected.abs() < 2.0 ** -6, 0.0, expected)
        conversions.append(dict(operand=name, mismatches=int(mismatch.sum()),
                                mismatches_if_flush_subnormal=int((actual != expected_ftz).sum()),
                                elements=actual.numel(), examples=examples))
    print(json.dumps(dict(check="fp8_rounding_diagnostic", shape=shape, lanes=lanes,
                          conversions=conversions,
                          fp8_vs_quant_reference_relative_rms=relative_rms(y8, quant_reference),
                          fp8_vs_device_quant_reference_relative_rms=relative_rms(y8, device_quant_reference),
                          fp8_vs_flush_subnormal_matmul_relative_rms=relative_rms(y8, ftz_reference),
                          fp8_max_abs_vs_flush_subnormal_matmul=float((y8.float() - ftz_reference.bfloat16().float()).abs().max()),
                          fp8_max_abs_vs_device_quant_reference=float((y8.float() - device_quant_reference.bfloat16().float()).abs().max()),
                          bf16_vs_fp32_relative_rms=relative_rms(y16, full_reference))), flush=True)
    if check_accumulation:
        y8_float = fp8[lanes](at, bd, scalars, lanes, True).cpu()
        y16_float = bf16[lanes](at, bd, lanes, True).cpu()
        fp8_error = accumulation_error(y8_float, aq, bq, 1.0 / (sa * sb))
        bf16_error = accumulation_error(y16_float, a.float(), b.float())
        print(json.dumps(dict(check="fp32_accumulation_bound", shape=shape, lanes=lanes,
                              fp8=fp8_error, bf16=bf16_error,
                              fp8_bf16_output_equals_rounded_fp32=int((y8 == y8_float.bfloat16()).sum()),
                              bf16_output_equals_rounded_fp32=int((y16 == y16_float.bfloat16()).sum()),
                              elements=y8.numel())), flush=True)
        if save_diagnostic_dir:
            destination = save_diagnostic_dir / ("x".join(map(str, shape)) + ".pt")
            torch.save(dict(a=a, b=b, aq=aq, bq=bq, y8=y8, y16=y16,
                            y8_float=y8_float, y16_float=y16_float,
                            scale_a=sa, scale_b=sb, lanes=lanes), destination)
            print(json.dumps(dict(check="fp8_diagnostic_saved", path=str(destination),
                                  bytes=destination.stat().st_size)), flush=True)
        if not diagnostic_only:
            assert fp8_error["finite"] and fp8_error["failures"] == 0
            assert bf16_error["finite"] and bf16_error["failures"] == 0
    if diagnostic_only:
        return
    assert all(x["mismatches"] == 0 for x in conversions), "FP8 conversion differs from independent reference"
    torch.testing.assert_close(y8.float(), quant_reference.bfloat16().float(), rtol=0.009, atol=1e-5)
    torch.testing.assert_close(y16.float(), full_reference.bfloat16().float(), rtol=0.009, atol=1e-5)
    assert torch.isfinite(y8).all() and torch.isfinite(y16).all()
    print(json.dumps(dict(check="fp8_correctness", shape=shape, scale_a=sa, scale_b=sb,
                          fp8_vs_quant_reference_relative_rms=relative_rms(y8, quant_reference),
                          fp8_vs_fp32_relative_rms=relative_rms(y8, full_reference),
                          bf16_vs_fp32_relative_rms=relative_rms(y16, full_reference))), flush=True)
    arms = {"native_torch_bf16": lambda: ad @ bd,
            "nki_bf16_layout_included": lambda: bf16[lanes](ad.T.contiguous(), bd, lanes),
            "nki_fp8_layout_cast_included": lambda: fp8[lanes](ad.T.contiguous(), bd, scalars, lanes),
            "nki_fp8_pretransposed_lower_bound": lambda: fp8[lanes](at, bd, scalars, lanes)}
    for fn in arms.values():
        for _ in range(3):
            fn()
        sync()
    samples = {name: [] for name in arms}
    for rep in range(3):
        order = list(arms) if rep % 2 == 0 else list(reversed(arms))
        for name in order:
            sync()
            start = time.perf_counter()
            for _ in range(steps):
                arms[name]()
            sync()
            samples[name].append((time.perf_counter() - start) * 1000 / steps)
    print(json.dumps(dict(check="fp8_benchmark", shape=shape, lanes=lanes, calibration_amax_included=False,
                          samples_ms=samples, median_ms={k: statistics.median(v) for k, v in samples.items()},
                          training_speed_or_quality_claim=False)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--large", action="store_true", help="also test an 8192-token projection tile")
    parser.add_argument("--check-accumulation", action="store_true",
                        help="Also return FP32 outputs and compare with FP64 dot products "
                             "using a declared FP32 accumulation error bound")
    parser.add_argument("--save-diagnostic-dir", type=Path,
                        help="Save CPU operands and measured outputs for independent offline "
                             "analysis; requires --check-accumulation and a new directory")
    parser.add_argument("--diagnostic-only", action="store_true",
                        help="Characterize rounding/accumulation on every shape without "
                             "claiming correctness or running performance benchmarks")
    args = parser.parse_args()
    if args.save_diagnostic_dir:
        if not args.check_accumulation:
            parser.error("--save-diagnostic-dir requires --check-accumulation")
        args.save_diagnostic_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    import torch_neuronx  # noqa: F401
    fp8, bf16, roundtrip = build_kernels()
    for shape in ((128, 256, 128), (256, 768, 512)):
        run_case(fp8, bf16, roundtrip, shape, args.steps, args.diagnostic_only,
                 args.check_accumulation, args.save_diagnostic_dir)
    if args.large:
        run_case(fp8, bf16, roundtrip, (8192, 768, 512), args.steps, args.diagnostic_only,
                 args.check_accumulation, args.save_diagnostic_dir)
    print("COMPLETED: FP8 characterization only; numerical gate still required" if args.diagnostic_only
          else "PASS: native FP8 primitive probes completed", flush=True)


if __name__ == "__main__":
    main()
