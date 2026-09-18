#!/usr/bin/env python3
"""Require exact loss/gradient agreement before timing CE scheduling variants."""
import argparse
import gc
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T
from mlp_fused_bf16_probe import timing
from softcap_ce_parallel_nki import build_kernels


def emit(kind, record):
    print(kind + " " + json.dumps(record), flush=True)


def calls():
    from torch_neuronx import wrap_nki
    forward, backward = (wrap_nki(k) for k in build_kernels())

    def original(z, target, gradient):
        loss, lse = T._softcap_ce_fwd(z, target)
        return loss, lse, T._softcap_ce_bwd(z, target, lse, gradient)

    result = {"original": original}
    for name, lanes, gather in (("mask_one_core", 1, False),
                                 ("mask_two_cores", 2, False),
                                 ("gather_two_cores", 2, True)):
        def call(z, target, gradient, lanes=lanes, gather=gather):
            loss, lse = forward[lanes](z, target, lanes, gather)
            return loss, lse, backward[lanes](z, target, lse, gradient, lanes)
        result[name] = call
    return result


def check(arms, z, labels, gradient, case, eligible):
    inputs = (z.to("neuron"), labels.float().reshape(-1, 1).to("neuron"),
              gradient.float().reshape(-1, 1).to("neuron"))
    baseline = tuple(x.cpu() for x in arms["original"](*inputs))
    for name, function in arms.items():
        output = tuple(x.cpu() for x in function(*inputs))
        differences = [float((a - b).abs().max()) for a, b in zip(output, baseline)]
        exact = [torch.equal(a, b) for a, b in zip(output, baseline)]
        finite = all(bool(torch.isfinite(a).all()) for a in output)
        repeat = tuple(x.cpu() for x in function(*inputs))
        repeated = all(torch.equal(a, b) for a, b in zip(output, repeat))
        passed = all(exact) and finite and repeated
        eligible[name] &= passed
        emit("CE_PARALLEL_GATE", dict(
            case=case, shape=list(z.shape), arm=name, finite=finite,
            exact_loss_lse_dz=exact, max_abs_loss_lse_dz=differences,
            repeated_exact=repeated, passed=passed))
        del output, repeat
    assert torch.equal(inputs[0].cpu(), z), "logits were mutated"
    assert torch.equal(inputs[1].cpu().flatten(), labels.float()), "labels were mutated"
    assert torch.equal(inputs[2].cpu().flatten(), gradient.float()), "upstream gradient mutated"
    # An independent FP64 calculation on a bounded row sample characterizes
    # the existing approximation; eligibility above still requires exact
    # agreement with the current training kernel, not a relaxed FP64 bound.
    sample = min(z.shape[0], 128)
    t = (z[:sample].double() / T.LOGIT_SOFTCAP).tanh()
    y = t * T.LOGIT_SOFTCAP
    target = y.gather(1, labels[:sample, None])
    truth_loss = (y - target).exp().sum(dim=1).log()
    probabilities = y.softmax(dim=1)
    probabilities[torch.arange(sample), labels[:sample]] -= 1
    truth_gradient = probabilities * (1 - t.square()) * gradient[:sample, None].double()
    emit("CE_PARALLEL_FP64_DIAGNOSTIC", dict(
        case=case, sampled_rows=sample, native_loss_max_abs=float(
            (baseline[0][:sample, 0].double() - truth_loss).abs().max()),
        native_gradient_max_abs=float((baseline[2][:sample].double() - truth_gradient).abs().max()),
        candidate_requires_exact_native_agreement=True))
    del inputs, baseline
    gc.collect()


def row_independence(arms, eligible):
    generator = torch.Generator().manual_seed(1755)
    z = torch.randn(256, 8192, generator=generator) * 6
    target = (torch.arange(256) * 31) % 8192
    gradient = torch.linspace(-1., 1., 256)
    original = (z.to("neuron"), target.float().reshape(-1, 1).to("neuron"),
                gradient.reshape(-1, 1).to("neuron"))
    z[127] = -z[127] + 13
    z[128] *= 2
    changed = (z.to("neuron"), original[1], original[2])
    keep = torch.ones(256, dtype=torch.bool)
    keep[127:129] = False
    for name, function in arms.items():
        before = tuple(a.cpu() for a in function(*original))
        after = tuple(a.cpu() for a in function(*changed))
        passed = all(torch.equal(a[keep], b[keep]) for a, b in zip(before, after))
        eligible[name] &= passed
        emit("CE_PARALLEL_ROW_INDEPENDENCE", dict(arm=name, passed=passed,
                                                 changed_rows=[127, 128]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-shape", action="store_true")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    import torch_neuronx
    torch.set_num_threads(4)
    assert T._HAS_NKI and T.LOGIT_SOFTCAP == 15.0
    arms = calls()
    eligible = {name: True for name in arms}
    emit("CE_PARALLEL_CONTRACT", dict(
        original_forward_lnc=getattr(T._softcap_ce_fwd, "lnc", None),
        original_backward_lnc=getattr(T._softcap_ce_bwd, "lnc", None),
        exact_native_agreement_required=True, backward_required=True,
        fixed_softcap=T.LOGIT_SOFTCAP, primitive_approximation_changed=False))
    generator = torch.Generator().manual_seed(5785)
    labels = torch.randint(8192, (256,), generator=generator)
    labels[:4] = torch.tensor([0, 8191, 4095, 4096])
    gradient = torch.randn(256, generator=generator) * .5
    with torch.no_grad():
        for name in ("tiny", "random", "saturation", "uniform", "confident_correct",
                     "confident_wrong", "zero_gradient"):
            z = torch.randn(256, 8192, generator=generator)
            if name == "tiny":
                z *= .001
            elif name in ("random", "zero_gradient"):
                z *= 6
            elif name == "saturation":
                z *= 1000
            elif name == "uniform":
                z.fill_(.7)
            else:
                z.zero_()
                z[torch.arange(256), labels] = 500 if name == "confident_correct" else -500
                if name == "confident_wrong":
                    z[torch.arange(256), (labels + 1) % 8192] = 500
            g = torch.zeros_like(gradient) if name == "zero_gradient" else gradient
            check(arms, z, labels, g, name, eligible)
        row_independence(arms, eligible)
        if args.real_shape:
            # B8*T2048 and B16*T1024 have this same loss shape. Round logits
            # through BF16 as the current final Linear does, then upcast.
            z = (torch.randn(16384, 8192, generator=generator) * 6).bfloat16().float()
            labels = torch.randint(8192, (16384,), generator=generator)
            gradient = torch.full((16384,), 1 / 16384)
            check(arms, z, labels, gradient, "real_training_shape_bf16_values", eligible)
            device = (z.to("neuron"), labels.float().reshape(-1, 1).to("neuron"),
                      gradient.reshape(-1, 1).to("neuron"))
            selected = {name: (lambda fn=function: fn(*device)) for name, function in arms.items()
                        if eligible[name]}
            measured = timing(selected, args.steps, torch.neuron.synchronize)
            emit("CE_PARALLEL_BENCH", dict(shape=list(z.shape), **measured,
                                           full_forward_backward=True,
                                           input_output_hbm_included=True,
                                           whole_model=False, eligible=eligible))
    emit("CE_PARALLEL_RESULT", dict(eligible=eligible, integrated_into_training=False))
    assert eligible["original"], "current native control failed structural checks"


if __name__ == "__main__":
    main()
