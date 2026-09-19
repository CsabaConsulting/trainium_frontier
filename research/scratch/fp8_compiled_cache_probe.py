#!/usr/bin/env python3
"""Gate and measure compiling the unchanged, online FP8 cache producer.

The wave124 FP8 forward/backward math and quantization policy are unchanged.
Compare every cache with independent CPU scales/legacy decode and the existing
eager producer, then gate exact complete outputs/gradients and changed masters.
Both native BF16 and FP8 get eager and compiled preparation controls.
"""
import argparse
import gc
import json

import torch

from fp8_complete_mlp_probe import build_functions, prepare_caches, run as independent_complete_gate
from fp8_gradient_prepare_nki import build_kernels as build_gradient_prepare
from fp8_mlp_data_backward_nki import build_kernel as build_backward
from fp8_mlp_island_probe import make_inputs
from fp8_mlp_token_rows_nki import build_kernel as build_forward
from fp8_packed_weight_cache_nki import build_kernel as build_pack
from fp8_packed_weight_cache_probe import decode_legacy
from fp8_rms_scale_contract import normalize, policy
from fp8_weight_cache_nki import build_kernels as build_calibration
from fp8_weight_cache_probe import calibration_reference, require_exact
from fp8_weight_gradient_nki import build_kernels as build_weight_gradient
from fresh_start_fp8_probe import e4m3_legacy
from mlp_fused_bf16_probe import timing


def emit(kind, **record):
    print(kind + " " + json.dumps(record), flush=True)


def make_producers(calibrate, pack):
    result = {}
    for mode in ("native", "full"):
        def make(selected):
            def prepare(up, down, xm, one):
                return prepare_caches((up, down), calibrate, pack, selected, (xm, one))
            return prepare
        fn = make(mode)
        result[mode + "_eager"] = fn
        result[mode + "_compiled"] = torch.compile(
            fn, backend="neuron", fullgraph=True, dynamic=False)
    return result


def independent_packed(name, actual, weight, metadata):
    rows, columns = weight.shape
    raw = e4m3_legacy(weight.float() * metadata[0, 0])
    width = actual.shape[-1]
    expected = raw.reshape(rows // width, width, columns // 128, 128).permute(
        0, 3, 2, 1).contiguous()
    require_exact(name, decode_legacy(actual), expected)


def cache_gate(name, caches, masters, fixed):
    assert len(caches) == 10 and all(not t.requires_grad for t in caches)
    up, down = [x.detach().cpu().bfloat16() for x in masters]
    for side, weight, q, m, fw, bw in (
        ("up", up, caches[0], caches[2], caches[4], caches[6]),
        ("down", down, caches[1], caches[3], caches[5], caches[7]),
    ):
        metadata, quantized = calibration_reference(weight)
        require_exact(name + ":" + side + ":metadata", m, metadata)
        require_exact(name + ":" + side + ":bf16_qdq", q, quantized)
        independent_packed(name + ":" + side + ":forward_bytes", fw, weight, metadata)
        independent_packed(name + ":" + side + ":backward_bytes", bw, weight.T.contiguous(), metadata)
    for i in range(2):
        require_exact(name + ":fixed" + str(i), caches[8 + i], fixed[i])


def exact_tuple(name, actual, expected):
    assert len(actual) == len(expected)
    for i, (a, b) in enumerate(zip(actual, expected)):
        assert a.dtype == b.dtype and a.shape == b.shape
        require_exact(name + ":" + str(i), a, b)


def run_case(shape, case, producers, functions, steps, benchmark=False, microbatches=2):
    assert microbatches in (1, 2)
    tokens, channels, hidden = shape
    x, up, down = make_inputs(shape, case, True)
    if case == "tiny_weights":
        up = (up.float() * 2.0**-110).bfloat16()
        down = (down.float() * 2.0**-110).bfloat16()
    elif case == "large_weights":
        up[0, 0] = 2.0**60
        down[-1, -1] = -(2.0**60)
    masters = tuple(v.float().to("neuron").requires_grad_() for v in (up, down))
    originals = tuple(v.detach().cpu().clone() for v in masters)
    scale, bound = policy(channels)
    fixed = (torch.tensor([scale, 1.0 / scale, bound]).repeat(128, 1).to("neuron"),
             torch.ones(128, 3).to("neuron"))
    caches = {name: fn(*masters, *fixed) for name, fn in producers.items()}
    cache_gate(case + ":eager", caches["full_eager"], masters, fixed)
    cache_gate(case + ":compiled", caches["full_compiled"], masters, fixed)
    exact_tuple(case + ":full_same_bytes", caches["full_compiled"], caches["full_eager"])
    exact_tuple(case + ":native_same_casts", caches["native_compiled"], caches["native_eager"])
    exact_tuple(case + ":repeat", producers["full_compiled"](*masters, *fixed), caches["full_compiled"])
    for i, master in enumerate(masters):
        require_exact(case + ":master_unchanged" + str(i), master, originals[i])

    # Cache-only extreme cases need no new forward/backward arithmetic gates.
    # The existing component can overflow with deliberately 2**60 weights.
    if case in ("large_weights", "tiny_weights"):
        emit("FP8_COMPILED_CACHE_GATE", case=case, shape=list(shape), passed=True,
             independent_metadata_qdq_bytes=True, component_gradient_checked=False)
        return

    generator = torch.Generator().manual_seed(97151 + channels + tokens)
    xs = (x, normalize(torch.randn(x.shape, generator=generator).bfloat16()))[:microbatches]
    inputs = tuple(v.to("neuron").requires_grad_() for v in xs)
    gradients = tuple((torch.randn(x.shape, generator=generator) * .03125).bfloat16().to("neuron")
                      for _ in range(microbatches))
    native, complete = functions

    def update(mode, keep=False):
        for leaf in (*inputs, *masters):
            leaf.grad = None
        current = producers[mode](*masters, *fixed)
        fn = native if mode.startswith("native") else complete
        saved = []
        for values, gradient in zip(inputs, gradients):
            output = fn(values, *masters, *current)
            output.backward(gradient)
            if keep:
                saved.append(output.detach().cpu().clone())
        if keep:
            return tuple(saved) + tuple(leaf.grad.detach().cpu().clone() for leaf in (*inputs, *masters))
        return output

    for mode in ("native", "full"):
        eager = update(mode + "_eager", True)
        compiled = update(mode + "_compiled", True)
        exact_tuple(case + ":component_" + mode, compiled, eager)
        assert all(v.dtype == torch.float32 for v in compiled[-2:])
    # Leave parameters out of autograd for cache creation, but refresh from the
    # actual changed FP32 masters. AOT must not freeze the input weight values.
    with torch.no_grad():
        masters[0].mul_(.75).add_(.000173)
        masters[1].mul_(1.25).sub_(.000291)
    updated = producers["full_compiled"](*masters, *fixed)
    cache_gate(case + ":updated", updated, masters, fixed)
    exact_tuple(case + ":updated_compilation", updated, producers["full_eager"](*masters, *fixed))
    assert any(not torch.equal(updated[i].cpu(), caches["full_compiled"][i].cpu()) for i in (0, 1))
    exact_tuple(case + ":updated_component", update("full_compiled", True), update("full_eager", True))
    with torch.no_grad():
        for master, original in zip(masters, originals):
            master.copy_(original.to("neuron"))
    exact_tuple(case + ":restore", producers["full_compiled"](*masters, *fixed), caches["full_compiled"])
    emit("FP8_COMPILED_CACHE_GATE", case=case, shape=list(shape), passed=True,
         independent_metadata_qdq_bytes=True, component_gradient_checked=True,
         exact_complete_outputs_and_gradients=True, microbatches=microbatches,
         two_distinct_microbatches=microbatches == 2,
         fp32_master_gradients=True, fp32_master_accumulation=microbatches == 2,
         updated_and_restored_master_caches=True)
    if benchmark:
        prep = timing({name: (lambda key=name: producers[key](*masters, *fixed))
                       for name in producers}, steps, torch.neuron.synchronize)
        full = timing({name: (lambda key=name: update(key)) for name in producers},
                      steps, torch.neuron.synchronize)
        emit("FP8_COMPILED_CACHE_BENCH", shape=list(shape), cache_preparation=prep,
             complete_component=full, units="ms", full_microbatches=microbatches,
             all_online_casts_calibration_transposes_packing_charged=True,
             same_compiled_forward_backward_in_each_precision_pair=True,
             native_compiled_cache_control=True, new_quantization_policy=False,
             optimizer_and_collectives_excluded=True, model_quality_claim=False)
    gc.collect()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-shape", action="store_true")
    ap.add_argument("--real-tokens", type=int, choices=(16384, 32768), default=16384)
    ap.add_argument("--real-microbatches", type=int, choices=(1, 2), default=2)
    ap.add_argument("--independent-real-gates", action="store_true",
                    help="Repeat independent forward/quantizer/four-product gates at the actual shape")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    if args.real_shape and args.real_tokens != 16384 and not args.independent_real_gates:
        ap.error("A new token shape requires --independent-real-gates before timing")
    import torch_neuronx
    from torch_neuronx import wrap_nki
    torch.set_num_threads(4)
    forward, backward, pack = (wrap_nki(fn()) for fn in (build_forward, build_backward, build_pack))
    partial, finish = (wrap_nki(k) for k in build_gradient_prepare())
    _, byte_pack, product = (wrap_nki(k) for k in build_weight_gradient())
    calibrate, quantize = (wrap_nki(k) for k in build_calibration())
    producers = make_producers(calibrate, pack)

    def functions(channels):
        _, complete, _, native = build_functions(channels, forward, backward, partial, finish, product)
        return native, complete

    small = functions(256)
    cases = ("random", "zero_up", "zero_down", "outliers", "changed_weights",
             "tiny_weights", "large_weights")
    for case in cases:
        run_case((512, 256, 512), case, producers, small, args.steps)
    if args.real_shape:
        shape = (args.real_tokens, 1024, 4096)
        applications = build_functions(1024, forward, backward, partial, finish, product)
        if args.independent_real_gates:
            kernels = (forward, backward, partial, finish, product, calibrate, quantize,
                       pack, byte_pack)
            # This prerequisite retains the established two-microbatch
            # accumulation test and native FP64 rounding-bound control. Its
            # two-microbatch timings are separate from the requested shape's
            # measurement below (the winning pack32 model uses one microbatch).
            independent_complete_gate(shape, "real_random", kernels, applications,
                                      min(args.steps, 5), True)
        run_case(shape, "real_random", producers, (applications[3], applications[1]),
                 args.steps, True, args.real_microbatches)
    emit("FP8_COMPILED_CACHE_RESULT", passed=True, cases=len(cases) + int(args.real_shape),
         real_shape=args.real_shape, real_tokens=args.real_tokens,
         real_microbatches=args.real_microbatches,
         independent_real_gates=args.independent_real_gates,
         fullgraph=True, full_model_or_quality_claim=False)


if __name__ == "__main__":
    main()
