#!/usr/bin/env python3
"""Gate scratch FP32 AdamW against FP64 math and the actual foreach optimizer."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T
from adamw_fused_fp32 import build_kernel, reference, scalar_values
from mlp_fused_bf16_probe import timing


def expected64(p, g, m, v, values):
    p, g, m, v = (x.double() for x in (p, g, m, v))
    lr, b1, b2, om1, om2, decay, eps = values.double().unbind()
    nm = (1 - om1) * m + om1 * g
    nv = (1 - om2) * v + om2 * g.square()
    update = (nm / b1) / ((nv / b2).sqrt() + eps)
    out = p * decay - update * lr
    envelopes = ((p * decay).abs() + (update * lr).abs(),
                 m.abs() + g.abs(), v.abs() + g.square())
    return (out, nm, nv), envelopes


def check(name, actual, expected, envelopes, enforce=True, inputs=None):
    results = []
    for field, a, e, envelope in zip(("weight", "first_moment", "second_moment"),
                                    actual, expected, envelopes):
        a = a.detach().cpu().double()
        error = (a - e).abs()
        u = 2.0 ** -24
        gamma = 32 * u / (1 - 32 * u)
        bound = 4 * gamma * envelope + 32 * 2.0 ** -149
        relative = float(error.norm() / e.norm().clamp_min(1e-30))
        row = dict(field=field, finite=bool(torch.isfinite(a).all()),
                   bound_failures=int((error > bound).sum()),
                   relative_rms=relative, max_abs=float(error.max()))
        row["passed"] = row["finite"] and not row["bound_failures"] and relative < 2e-6
        if not row["passed"]:
            index = int((error / bound).flatten().argmax())
            row["worst_bound"] = dict(
                flat_index=index, actual=float(a.flatten()[index]),
                expected=float(e.flatten()[index]),
                error=float(error.flatten()[index]),
                bound=float(bound.flatten()[index]),
                envelope=float(envelope.flatten()[index]))
            if inputs is not None:
                row["worst_bound"]["inputs"] = {
                    key: float(value.flatten()[index])
                    for key, value in zip(("p", "g", "m", "v"), inputs)}
        results.append(row)
    record = dict(name=name, enforced=enforce, passed=all(r["passed"] for r in results), results=results)
    print("ADAMW_FP32_NUMERICAL " + json.dumps(record), flush=True)
    if enforce:
        assert record["passed"], record
    return record


def optimizer_for(p, g, m, v, step, lr, beta1, beta2, wd):
    parameter = torch.nn.Parameter(p)
    parameter.grad = g
    opt = T.MuonAdamW([dict(kind="adamw", params=[parameter], lr=lr,
                           betas=(beta1, beta2), eps=1e-10, weight_decay=wd)])
    opt.state[parameter] = dict(step=step - 1, exp_avg=m, exp_avg_sq=v)
    return parameter, opt


def gates(kernel, compiled, rows, cols, cases, math_mode=0):
    generator = torch.Generator().manual_seed(1063 + rows + cols)
    compiled_eligible = True
    for case in cases:
        p = torch.randn(rows, cols, generator=generator) * .03
        g = torch.randn(rows, cols, generator=generator) * .003
        m = torch.randn(rows, cols, generator=generator) * .001
        v = torch.rand(rows, cols, generator=generator) * 1e-5
        step, lr, b1, b2, wd = 17, .02, .8, .96, .01
        if case in ("first", "sparse", "tiny"):
            m.zero_(); v.zero_(); step = 1
        if case == "first":
            p.zero_()  # expose update error without a larger old-weight term
        elif case == "zero":
            m.zero_(); v.zero_(); g.zero_(); wd = 0.0
        elif case == "decay":
            g.zero_(); step = 1000
        elif case == "sparse":
            g[torch.rand(rows, cols, generator=generator) < .97] = 0
        elif case == "tiny":
            p.zero_(); g.mul_(1e-24)
        elif case == "late":
            step, lr, wd, b1, b2 = 3100, 0.0, .001, .9, .95
        elif case == "large":
            g.mul_(1e6); v.mul_(1e12); m.mul_(1e6)
        values = scalar_values(lr, step, b1, b2, wd)
        expected, envelopes = expected64(p, g, m, v, values)
        scalar_device = values.repeat(128, 1).to("neuron")
        pd, opt = optimizer_for(*(x.to("neuron") for x in (p, g, m, v)), step, lr, b1, b2, wd)
        with torch.no_grad():
            opt.step()
        state = opt.state[pd]
        assert state["step"] == step
        check(f"foreach:{rows}:{cols}:{case}", (pd, state["exp_avg"], state["exp_avg_sq"]),
              expected, envelopes, enforce=False, inputs=(p, g, m, v))
        # Native precision is measured independently, not used to relax the
        # candidate's predeclared analytic bounds.
        for name in ("compiled", "nki_functional", "nki_inplace"):
            inputs = tuple(x.to("neuron") for x in (p, g, m, v))
            with torch.no_grad():
                if name == "compiled":
                    outputs = compiled(*inputs, scalar_device)
                else:
                    outputs = kernel[2](*inputs, scalar_device, 2, name == "nki_inplace", math_mode)
            # The native compiled sqrt path failed the original real-shape
            # analytic bound in wave54. Keep that bound and disqualify this
            # control if it fails, while still running independent NKI gates.
            record = check(f"{name}:{rows}:{cols}:{case}", outputs, expected, envelopes,
                           enforce=name != "compiled", inputs=(p, g, m, v))
            if name == "compiled":
                compiled_eligible &= record["passed"]
            assert torch.equal(inputs[1].cpu(), g), "gradient input was mutated"
            if name == "nki_inplace":
                for mutated, result in zip((inputs[0], inputs[2], inputs[3]), outputs):
                    torch.testing.assert_close(mutated.cpu(), result.cpu(), rtol=0, atol=0)
            else:
                for original, value in zip((p, m, v), (inputs[0], inputs[2], inputs[3])):
                    assert torch.equal(value.cpu(), original), "functional call mutated input"
            if case in ("zero", "late"):
                assert torch.equal(outputs[0].cpu(), p)
    return compiled_eligible


def benchmark(kernel, compiled, rows, cols, steps, compiled_eligible, math_mode=0):
    generator = torch.Generator().manual_seed(553)
    initial = (torch.randn(rows, cols, generator=generator) * .03,
               torch.randn(rows, cols, generator=generator) * .003,
               torch.zeros(rows, cols), torch.zeros(rows, cols))
    b1, b2, wd = .8, .96, .01
    native_p, native = optimizer_for(*(x.to("neuron") for x in initial), 1, .02, b1, b2, wd)
    native_count = 0

    def foreach():
        nonlocal native_count
        native_count += 1
        native.param_groups[0]["lr"] = .02 / (1 + .001 * native_count)
        native.step()

    arms = {"actual_foreach": foreach}
    candidates = ["nki_functional", "nki_inplace"]
    if math_mode:
        candidates.insert(0, "nki_control_functional")
    if compiled_eligible:
        candidates.insert(0, "compiled")
    for name in candidates:
        state = list(x.to("neuron") for x in initial)
        staging = torch.empty(128, 7, dtype=torch.float32)
        device_scalars = torch.empty_like(staging, device="neuron")

        def make_call(arm, tensors, host, device):
            count = 0

            def call():
                nonlocal count
                count += 1
                values = scalar_values(.02 / (1 + .001 * count), count, b1, b2, wd)
                host.copy_(values.expand_as(host))
                device.copy_(host)
                if arm == "compiled":
                    new = compiled(*tensors, device)
                else:
                    mode = 0 if arm == "nki_control_functional" else math_mode
                    new = kernel[2](*tensors, device, 2, arm == "nki_inplace", mode)
                if arm != "nki_inplace":
                    tensors[0].copy_(new[0])
                    tensors[2], tensors[3] = new[1], new[2]
            return call

        arms[name] = make_call(name, state, staging, device_scalars)
    with torch.no_grad():
        measured = timing(arms, steps, torch.neuron.synchronize)
    print("ADAMW_FP32_BENCH " + json.dumps(dict(
        rows=rows, columns=cols, elements=rows * cols, math_mode=math_mode, **measured,
        scheduled_lr_bias_updates_scalar_upload_and_weight_copy_charged=True,
        compiled_control_eligible=compiled_eligible,
        parameter_and_moment_dtype="float32", whole_optimizer=False)), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-shape", action="store_true")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--math-mode", type=int, choices=(0, 1, 2, 3, 4, 5, 6), default=0)
    args = ap.parse_args()
    print("ADAMW_FP32_MATH_MODE " + json.dumps(dict(math_mode=args.math_mode)), flush=True)
    import torch_neuronx
    torch.set_num_threads(4)
    T.USE_FOREACH_OPTIM = T.USE_TENSOR_LR_SCALARS = True
    assert not T.USE_CAUTIOUS_WD
    kernel = torch_neuronx.wrap_nki(build_kernel())
    compiled = torch.compile(reference, backend="neuron", fullgraph=True, dynamic=False)
    compiled_eligible = True
    for rows, cols in ((256, 128), (256, 2048)):
        compiled_eligible &= gates(kernel, compiled, rows, cols,
                                   ("first", "zero", "decay", "sparse", "tiny", "late", "large"),
                                   args.math_mode)
    if args.real_shape:
        compiled_eligible &= gates(kernel, compiled, 4096, 2048, ("first", "zero", "decay"),
                                   args.math_mode)
        benchmark(kernel, compiled, 4096, 2048, args.steps, compiled_eligible, args.math_mode)
    print("PASS: NKI FP32 AdamW analytic, state and no-mutation gates; "
          f"compiled control eligible={compiled_eligible}", flush=True)


if __name__ == "__main__":
    main()
