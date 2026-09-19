#!/usr/bin/env python3
"""Gate forward-produced FP8 activations and charge their producer/consumer cost.

Retain all original independent forward gates. The new outputs are raw legacy
bytes already present on chip and per-row hidden metadata. Compare with a real
forward that saves the metadata and then repacks its BF16 activations. Neither
comparison assumes these backward inputs can be obtained for free.
"""
import argparse
import gc
import json

import torch

from fresh_start_fp8_probe import e4m3_legacy
from fp8_mlp_island_probe import make_inputs, run_case, row_reference
from fp8_mlp_token_rows_nki import build_kernel
from fp8_packed_weight_cache_nki import build_kernel as build_packer
from fp8_packed_weight_cache_probe import decode_legacy, pack_gate, selected_lanes
from fp8_rms_scale_contract import policy, normalize
from fp8_weight_cache_nki import build_kernels as build_calibration
from fp8_weight_cache_probe import require_exact
from fp8_weight_gradient_nki import build_kernels as build_weight_gradient
from mlp_fused_bf16_probe import timing


def check_bytes(name, actual, previous):
    a, b = actual.cpu(), previous.cpu()
    assert a.dtype == b.dtype == torch.uint8
    different = a != b
    zeros = ((a & 127) == 0) & ((b & 127) == 0)
    assert not bool((different & ~zeros).any())
    require_exact(name + ":decoded", decode_legacy(a), decode_legacy(b))
    print("FP8_FORWARD_BYTE_REPACK " + json.dumps(dict(
        name=name, signed_zero_changes=int((different & zeros).sum()),
        nonzero_byte_changes=0, passed=True)), flush=True)


def run(shape, case, kernel, calibrate, quantize, pack, byte_pack, steps, benchmark):
    # The original independent quantizer, pair-accumulation, rounding and row
    # independence checks are prerequisites, not replaced by self-comparison.
    run_case(shape, case, kernel, calibrate, quantize, True, True, pack)
    tokens, channels, hidden = shape
    x, up, down = make_inputs(shape, case, True)
    um, uq, pu = pack_gate(case + ":saved_up", up, calibrate, quantize, pack)
    dm, dq, pd = pack_gate(case + ":saved_down", down, calibrate, quantize, pack)
    scale, bound = policy(channels)
    xm = torch.tensor([scale, 1.0 / scale, bound]).repeat(128, 1).to("neuron")
    xd = x.to("neuron")

    def call(diagnostic=False, save=False, metadata=False, values=xd,
             current=(uq, dq, um, dm, pu, pd)):
        u, d, us, ds, packed_u, packed_d = current
        return kernel[2](values, u, d, us, ds, 2, 128, diagnostic,
                         scale, packed_u, packed_d, save, metadata)

    original = call(diagnostic=True)
    candidate = call(diagnostic=True, save=True)
    assert len(original) == 8 and len(candidate) == 10
    for index, (observed, expected) in enumerate(zip(candidate[:8], original)):
        require_exact(case + f":all_original_stages:{index}", observed, expected)
    y, relu, xq, hq, _, _, _, hm = candidate[:8]
    xbytes, hbytes = candidate[8:]
    require_exact(case + ":independent_input_byte_decoder", decode_legacy(xbytes),
                  e4m3_legacy(x.float() * scale))
    hidden_values = relu.cpu().square()
    expected_hm, _ = row_reference(hidden_values)
    require_exact(case + ":independent_hidden_metadata", hm, expected_hm)
    require_exact(case + ":independent_hidden_byte_decoder", decode_legacy(hbytes),
                  e4m3_legacy(hidden_values.float() * expected_hm[:, 0:1]))
    expected = (y, relu, xq, hq, xbytes, hbytes, hm)
    normal = call(save=True)
    repeated = call(save=True)
    for index, (observed, again, truth) in enumerate(zip(normal, repeated, expected)):
        require_exact(case + f":ordinary_vs_diagnostic:{index}", observed, truth)
        require_exact(case + f":repeat:{index}", again, truth)
    metadata_only = call(metadata=True)
    for index, (observed, truth) in enumerate(zip(metadata_only, (*expected[:4], hm))):
        require_exact(case + f":metadata_only:{index}", observed, truth)
    check_bytes(case + ":input_repack", xbytes, byte_pack[2](xq, xm, 2, False))
    check_bytes(case + ":hidden_repack", hbytes, byte_pack[2](hq, hm, 2, True))
    require_exact(case + ":input_unchanged", xd, x)
    if case in ("random", "real_random"):
        changed = x.clone()
        changed[127] = -changed[127] * 7
        changed[128] += 11
        changed[127:129] = normalize(changed[127:129])
        altered = call(save=True, values=changed.to("neuron"))
        keep = torch.ones(tokens, dtype=torch.bool); keep[127:129] = False
        for index, (observed, truth) in enumerate(zip(altered, expected)):
            require_exact(case + f":saved_row_independence:{index}",
                          observed.cpu()[keep], truth.cpu()[keep])
    print("FP8_FORWARD_SAVE_GATE " + json.dumps(dict(
        case=case, shape=list(shape), passed=True, all_original_stages_exact=True,
        independent_byte_decoders=True, hidden_scales_exact=True,
        existing_bf16_saves_retained=True, backward_or_quality_promotion=False)), flush=True)
    if benchmark:
        masters = up.float().to("neuron"), down.float().to("neuron")
        ul, dl = selected_lanes(up.shape[0]), selected_lanes(down.shape[0])

        def update(mode):
            ub, db = (master.bfloat16() for master in masters)
            us, ds = calibrate[1](ub), calibrate[1](db)
            u, packed_u = pack[ul](ub, us, ul)
            d, packed_d = pack[dl](db, ds, dl)
            current = (u, d, us, ds, packed_u, packed_d)
            for _ in range(2):
                if mode == "save":
                    value = call(save=True, current=current)
                elif mode == "repack":
                    first = call(metadata=True, current=current)
                    bx = byte_pack[2](first[2], xm, 2, False)
                    bh = byte_pack[2](first[3], first[4], 2, True)
                    value = (*first[:4], bx, bh, first[4])
                else:
                    value = call(current=current)
            return value

        baseline, saved = update("repack"), update("save")
        for index in (0, 1, 2, 3, 6):
            require_exact("timed_complete_producer:" + str(index), saved[index], baseline[index])
        for index in (4, 5):
            check_bytes("timed_complete_producer:" + str(index), saved[index], baseline[index])
        measured = timing({name: (lambda mode=mode: update(mode)) for name, mode in (
            ("ordinary_fp8_forward", "ordinary"), ("forward_then_repack", "repack"),
            ("forward_saves_fp8_bytes", "save"))}, steps, torch.neuron.synchronize)
        print("FP8_FORWARD_SAVE_BENCH " + json.dumps(dict(shape=list(shape), **measured,
            units="ms per online weight preparation plus two forward activation producers",
            all_master_casts_calibration_packing_and_saves_charged=True,
            metadata_save_charged_in_both_backward_input_producers=True,
            ordinary_forward_does_not_produce_required_backward_inputs=True,
            backward_gradients_optimizer_whole_model_excluded=True)), flush=True)
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-shape", action="store_true")
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    import torch_neuronx
    from torch_neuronx import wrap_nki
    torch.set_num_threads(4)
    kernel, pack = wrap_nki(build_kernel()), wrap_nki(build_packer())
    calibrate, quantize = (wrap_nki(k) for k in build_calibration())
    byte_pack = wrap_nki(build_weight_gradient()[1])
    cases = ("random", "zero_input", "zero_up", "zero_down", "outliers",
             "changed_weights", "token_scale_range", "norm_spikes", "norm_tiny")
    for case in cases:
        run((512, 256, 512), case, kernel, calibrate, quantize, pack, byte_pack, args.steps, False)
    if args.real_shape:
        run((16384, 1024, 4096), "real_random", kernel, calibrate, quantize, pack,
            byte_pack, args.steps, True)
    print("FP8_FORWARD_SAVE_RESULT " + json.dumps(dict(
        passed=True, cases=len(cases) + int(args.real_shape), real_shape=args.real_shape,
        whole_backward_or_quality_gain_established=False)), flush=True)


if __name__ == "__main__":
    main()
