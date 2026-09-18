#!/usr/bin/env python3
"""Forward-only Trn2 GEMM probe with full-weight SBUF reuse and N tiling.

The previous prototype was limited to N<=512 and requantized weights for every
128-row activation tile. Here each physical core loads/quantizes its complete
weight operand once per call, and each activation tile once across output tiles.
This is NOT a training implementation or an update-amortized weight cache.
Separate arms include activation layout only, both activation/weight layouts,
and no layout conversion. Native matmul AND F.linear are controls. Online amax
calibration and backward are excluded and remain necessary future work.

API: installed NKI0.5, NC-v3 legacy E4M3, double_row, 512-wide PSUM tiles.
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_matmul.html
"""
import argparse
import json
import statistics
import time

import torch

from fresh_start_fp8_probe import e4m3_legacy, eleven_bit_pair_reference, relative_rms


def build_kernel():
    global nl, nisa
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def cached_gemm(a_t, b, scales, lanes: int = 2,
                    use_fp8: bool = True, fp32_output: bool = False):
        k, m = a_t.shape
        bk, n = b.shape
        assert bk == k and lanes in (1, 2)
        pack = 2 if use_fp8 else 1
        tk = 128 * pack
        tn = min(n, 512)  # NC-v3 moving/PSUM free-dimension limit, not gen4.
        assert m % (128 * lanes) == 0 and k % tk == 0 and n % tn == 0
        assert tn % 128 == 0
        nk, nn = k // tk, n // tn
        cache_dtype = nl.float8_e4m3 if use_fp8 else nl.bfloat16
        element_bytes = 1 if use_fp8 else 2
        # Leave room for DMA/cast/output buffers; this is a capacity gate, not
        # an assertion about compiler allocation or achieved occupancy.
        assert (k * n + k * 128) * element_bytes + 4 * 1024**2 <= nl.tile_size.sbuf_size_bytes
        output_dtype = nl.float32 if fp32_output else nl.bfloat16
        out = nl.ndarray((m, n), dtype=output_dtype, buffer=nl.shared_hbm)
        av = a_t.reshape((k // pack, pack, m))
        bv = b.reshape((k // pack, pack, n))
        weights = nl.ndarray((128, nn, nk, pack, tn), dtype=cache_dtype, buffer=nl.sbuf)
        if use_fp8:
            sa = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            sb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            inverse = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=sa, src=scales[:, 0:1])
            nisa.dma_copy(dst=sb, src=scales[:, 1:2])
            nisa.dma_copy(dst=inverse, src=scales[:, 2:3])
        for ni in nl.affine_range(nn):
            for ki in nl.affine_range(nk):
                wb = nl.ndarray((128, pack, tn), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=wb,
                              src=bv[ki * 128:(ki + 1) * 128, :, ni * tn:(ni + 1) * tn])
                if use_fp8:
                    nisa.tensor_scalar(dst=weights[:, ni, ki, :, :], data=wb,
                                       op0=nl.multiply, operand0=sb)
                else:
                    nisa.tensor_copy(dst=weights[:, ni, ki, :, :], src=wb)
        lane = nl.program_id(0)
        for mi0 in nl.affine_range(m // (128 * lanes)):
            mi = mi0 * lanes + lane
            activations = nl.ndarray((128, nk, pack, 128), dtype=cache_dtype, buffer=nl.sbuf)
            for ki in nl.affine_range(nk):
                ab = nl.ndarray((128, pack, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=ab,
                              src=av[ki * 128:(ki + 1) * 128, :, mi * 128:(mi + 1) * 128])
                if use_fp8:
                    nisa.tensor_scalar(dst=activations[:, ki, :, :], data=ab,
                                       op0=nl.multiply, operand0=sa)
                else:
                    nisa.tensor_copy(dst=activations[:, ki, :, :], src=ab)
            for ni in nl.affine_range(nn):
                acc = nl.ndarray((128, tn), dtype=nl.float32, buffer=nl.psum)
                for ki in nl.affine_range(nk):
                    if use_fp8:
                        nisa.nc_matmul(dst=acc, stationary=activations[:, ki, :, :],
                                       moving=weights[:, ni, ki, :, :], accumulate=(ki > 0),
                                       perf_mode=nisa.matmul_perf_mode.double_row)
                    else:
                        nisa.nc_matmul(dst=acc,
                                       stationary=activations[:, ki, :, :].reshape((128, 128)),
                                       moving=weights[:, ni, ki, :, :].reshape((128, tn)),
                                       accumulate=(ki > 0))
                summed = nl.ndarray((128, tn), dtype=nl.float32, buffer=nl.sbuf)
                result = nl.ndarray((128, tn), dtype=output_dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=summed, src=acc)
                if use_fp8:
                    nisa.tensor_scalar(dst=result, data=summed, op0=nl.multiply, operand0=inverse)
                else:
                    nisa.tensor_copy(dst=result, src=summed)
                nisa.dma_copy(dst=out[mi * 128:(mi + 1) * 128, ni * tn:(ni + 1) * tn], src=result)
        return out

    return cached_gemm


def sync():
    torch.neuron.synchronize()


def sampled_reference(aq, bq, inverse, y32):
    m, k = aq.shape
    n = bq.shape[1]
    rows = sorted({i for i in (0, 1, 127, 128, 255, m // 2, m - 129, m - 128, m - 2, m - 1)
                   if 0 <= i < m})
    cols = sorted({min(n - 1, start + offset) for start in range(0, n, 512)
                   for offset in (0, 1, 127, 511)})
    reference, bound = eleven_bit_pair_reference(aq[rows], bq[:, cols], inverse)
    error = (y32[rows][:, cols].double() - reference).abs()
    result = dict(rows=rows, columns=cols, checked_outputs=error.numel(),
                  failures=int((error > bound).sum()), max_abs=float(error.max()),
                  max_error_over_bound=float((error / bound.clamp_min(1e-300)).max()),
                  reference="previously characterized empirical 11-significand-bit pair sums")
    assert result["failures"] == 0, result
    return result


def run_case(kernel, shape, steps, seed):
    m, k, n = shape
    torch.manual_seed(seed + k + n)
    a = (torch.randn(m, k) * 0.5).bfloat16()
    b = (torch.randn(k, n) / k**0.5).bfloat16()
    sa = float(2.0 ** torch.floor(torch.log2(128.0 / a.float().abs().max())))
    sb = float(2.0 ** torch.floor(torch.log2(128.0 / b.float().abs().max())))
    inverse = 1.0 / (sa * sb)
    aq, bq = e4m3_legacy(a.float() * sa), e4m3_legacy(b.float() * sb)
    scales = torch.tensor([sa, sb, inverse]).repeat(128, 1).to("neuron")
    ad, bd = a.to("neuron"), b.to("neuron")
    native_weight = bd.T.contiguous()
    at = ad.T.contiguous()
    lanes = 2 if m % 256 == 0 else 1
    y8 = kernel[lanes](at, bd, scales, lanes, True, False).cpu()
    y16 = kernel[lanes](at, bd, scales, lanes, False, False).cpu()
    y32 = kernel[lanes](at, bd, scales, lanes, True, True).cpu()
    assert torch.isfinite(y8).all() and torch.isfinite(y16).all() and torch.isfinite(y32).all()
    sample = sampled_reference(aq, bq, inverse, y32)
    # All-output envelope under the already characterized pair-rounding model:
    # per-pair u=2^-11, then FP32 accumulation, then BF16 output rounding.
    ref8 = aq.double() @ bq.double() * inverse
    abs8 = aq.double().abs() @ bq.double().abs() * inverse
    gamma = (k // 2) * 2.0**-24 / (1 - (k // 2) * 2.0**-24)
    before_output = (2.0**-11 + (1 + 2.0**-11) * gamma) * abs8
    bound8 = before_output + 2.0**-8 * (ref8.abs() + before_output) + 1e-30
    error8 = (y8.double() - ref8).abs()
    ref16 = a.double() @ b.double()
    abs16 = a.double().abs() @ b.double().abs()
    gamma16 = k * 2.0**-24 / (1 - k * 2.0**-24)
    bound16 = gamma16 * abs16 + 2.0**-8 * (ref16.abs() + gamma16 * abs16) + 1e-30
    error16 = (y16.double() - ref16).abs()
    check = dict(shape=shape, seed=seed, lanes=lanes, scale_a=sa, scale_b=sb,
                 fp8_bound_failures=int((error8 > bound8).sum()),
                 bf16_bound_failures=int((error16 > bound16).sum()),
                 fp8_quantized_relative_rms=relative_rms(y8, ref8),
                 fp8_unquantized_relative_rms=relative_rms(y8, ref16),
                 bf16_relative_rms=relative_rms(y16, ref16), sampled_pair_reference=sample,
                 fp8_cache_bytes=k * n, bf16_cache_bytes=2 * k * n,
                 reference_is_hardware_specification=False)
    print("TILED_FP8_GATE " + json.dumps(check), flush=True)
    assert check["fp8_bound_failures"] == 0 and check["bf16_bound_failures"] == 0
    assert check["fp8_quantized_relative_rms"] < 0.0025 and check["bf16_relative_rms"] < 0.0025
    arms = {
        "native_bf16": lambda: ad @ bd,
        "native_bf16_linear_nk_weight": lambda: torch.nn.functional.linear(ad, native_weight),
        "weight_layout_refresh_only": lambda: native_weight.T.contiguous(),
        "cached_nki_bf16_layout_included": lambda: kernel[lanes](ad.T.contiguous(), bd, scales, lanes, False, False),
        "cached_nki_fp8_layout_cast_included": lambda: kernel[lanes](ad.T.contiguous(), bd, scales, lanes, True, False),
        "cached_nki_fp8_both_layouts_cast_included": lambda: kernel[lanes](ad.T.contiguous(), native_weight.T.contiguous(), scales, lanes, True, False),
        "cached_nki_fp8_pretransposed_lower_bound": lambda: kernel[lanes](at, bd, scales, lanes, True, False),
    }
    for fn in arms.values():
        for _ in range(3):
            fn()
        sync()
    samples = {name: [] for name in arms}
    for repeat in range(3):
        order = list(arms) if repeat % 2 == 0 else list(reversed(arms))
        for name in order:
            sync()
            start = time.perf_counter()
            for _ in range(steps):
                arms[name]()
            sync()
            samples[name].append((time.perf_counter() - start) * 1000 / steps)
    medians = {name: statistics.median(v) for name, v in samples.items()}
    print("TILED_FP8_BENCH " + json.dumps(dict(
        shape=shape, samples_ms=samples, median_ms=medians,
        online_amax_included=False, activation_transpose_in_headline=True,
        weight_quantization_in_headline=True, weight_reuse_scope="one GEMM call",
        estimated_fp8_ms_if_weight_layout_amortized_over_four_calls=(
            medians["cached_nki_fp8_layout_cast_included"] + medians["weight_layout_refresh_only"] / 4),
        amortized_layout_is_estimate_not_training_measurement=True,
        training_speed_or_quality_claim=False)), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mlp-shapes", action="store_true")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=71)
    args = ap.parse_args()
    torch.set_num_threads(4)
    import torch_neuronx  # noqa: F401
    kernel = build_kernel()
    shapes = [(128, 256, 128), (256, 768, 512)]
    if args.mlp_shapes:
        shapes += [(8192, 1024, 4096), (8192, 4096, 1024)]
    for shape in shapes:
        print("TILED_FP8_CASE " + json.dumps(dict(shape=shape, seed=args.seed)), flush=True)
        run_case(kernel, shape, args.steps, args.seed)
    print("PASS: tiled forward-only FP8 probe; no training claim", flush=True)


if __name__ == "__main__":
    main()
