#!/usr/bin/env python3
"""Gate both packed weight orientations from one calibrated NKI producer.

Reuse the existing independent cache references and complete-gradient gates.
The new producer occupies the full_compiled arm; full_previous_compiled retains
the source132/135 compiled producer as an explicit performance control.
"""
import argparse
import json

import torch

from fp8_compiled_cache_probe import make_producers, run_case
from fp8_complete_mlp_probe import build_functions
from fp8_dual_weight_cache_nki import build_kernel as build_dual
from fp8_gradient_prepare_nki import build_kernels as build_gradient_prepare
from fp8_mlp_data_backward_nki import build_kernel as build_backward
from fp8_mlp_token_rows_nki import build_kernel as build_forward
from fp8_packed_weight_cache_nki import build_kernel as build_pack
from fp8_packed_weight_cache_probe import selected_lanes
from fp8_weight_cache_nki import build_kernels as build_calibration
from fp8_weight_gradient_nki import build_kernels as build_weight_gradient


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-shape", action="store_true")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--resident", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    import torch_neuronx
    from torch_neuronx import wrap_nki
    torch.set_num_threads(4)
    forward, backward, pack, dual = (wrap_nki(fn()) for fn in
                                    (build_forward, build_backward, build_pack, build_dual))
    partial, finish = (wrap_nki(k) for k in build_gradient_prepare())
    product = wrap_nki(build_weight_gradient()[2])
    calibrate = wrap_nki(build_calibration()[0])
    producers = make_producers(calibrate, pack)
    producers["full_previous_compiled"] = producers["full_compiled"]

    def prepare(up, down, xm, one):
        u, d = up.detach().bfloat16(), down.detach().bfloat16()
        ul, dl = selected_lanes(u.shape[0]), selected_lanes(d.shape[0])
        uq, um, uf, ub = dual[ul](u, ul, args.resident)
        dq, dm, df, db = dual[dl](d, dl, args.resident)
        return uq, dq, um, dm, uf, df, ub, db, xm, one

    producers["full_compiled"] = torch.compile(
        prepare, backend="neuron", fullgraph=True, dynamic=False)
    print("FP8_DUAL_CACHE_CONTRACT " + json.dumps(dict(
        resident=args.resident, new_candidate_arm="full_compiled",
        previous_compiled_control="full_previous_compiled",
        quantization_policy_unchanged=True, independent_cpu_metadata_qdq_layouts=True,
        no_bf16_transpose_in_new_producer=True, both_orientations_from_same_bytes=True,
        duplicate_calibration_per_physical_core_charged=True,
        all_current_master_casts_and_preparation_charged=True)), flush=True)

    def functions(channels):
        _, complete, _, native = build_functions(channels, forward, backward, partial, finish, product)
        return native, complete

    small = functions(256)
    cases = ("random", "zero_up", "zero_down", "outliers", "changed_weights",
             "tiny_weights", "large_weights")
    for case in cases:
        run_case((512, 256, 512), case, producers, small, args.steps)
    if args.real_shape:
        # Source135 supplies the independent four-product and FP64 native
        # bounds at this shape. This gate requires identical cache bytes and
        # exact complete gradients from the changed producer before timing.
        run_case((32768, 1024, 4096), "real_random", producers, functions(1024),
                 args.steps, True, microbatches=1)
    print("FP8_DUAL_CACHE_RESULT " + json.dumps(dict(
        passed=True, cases=len(cases) + int(args.real_shape), resident=args.resident,
        real_shape=args.real_shape, full_model_or_quality_claim=False)), flush=True)


if __name__ == "__main__":
    main()
