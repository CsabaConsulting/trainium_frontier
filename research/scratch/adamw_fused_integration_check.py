#!/usr/bin/env python3
"""Check repeated real-table updates, skipped gradients, caches and state restore."""
import argparse
import io
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T
from adamw_fused_fp32 import scalar_values
from adamw_fused_fp32_probe import check, expected64, optimizer_for
from adamw_fused_integration import install_optimizer
from adamw_stage_accuracy import check_stages, cpu_counterexamples


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def make_optimizer(initial, fused, math_mode=0):
    params = [torch.nn.Parameter(x.to("neuron")) for x in initial]
    groups = [dict(kind="adamw", params=params[:2], lr=.02, betas=(.8, .96),
                   eps=1e-10, weight_decay=.01),
              dict(kind="adamw", params=params[2:], lr=.002, betas=(.9, .95),
                   eps=1e-10, weight_decay=0.0)]
    opt = T.MuonAdamW(groups)
    if fused:
        install_optimizer(opt, T, math_mode=math_mode)
    return params, opt


def state_cpu(opt, p):
    state = opt.state.get(p, {})
    return dict(step=state.get("step", 0),
                exp_avg=(state["exp_avg"].detach().cpu().clone()
                         if state else torch.zeros_like(p, device="cpu")),
                exp_avg_sq=(state["exp_avg_sq"].detach().cpu().clone()
                            if state else torch.zeros_like(p, device="cpu")))


def diagnose_native_sqrt():
    # Reproduce the failing wave55 value and isolate sqrt from its input's
    # moment/bias rounding. This is diagnostic; it does not relax any gate.
    gradient_values = [-7.873063978536265e-9, -1e-9, -1e-6, -.003, 0.0, .003]
    g = torch.zeros(256, 128)
    for column, value in enumerate(gradient_values):
        g[:, column] = value
    values = scalar_values(.02, 1, .8, .96, .01)

    def stages(gradient, scalars):
        lr, b1, b2, om1, om2, _, eps = scalars[0].unbind()
        moment = gradient * om1
        variance = gradient.square() * om2 / b2
        root = variance.sqrt()
        denominator = root + eps
        update = (moment / b1) / denominator * lr
        return variance, root, denominator, update

    compiled = torch.compile(stages, backend="neuron", fullgraph=True, dynamic=False)
    actual = [x.cpu().double() for x in compiled(g.to("neuron"), values.repeat(128, 1).to("neuron"))]
    variance, root, denominator, update = actual
    for column, value in enumerate(gradient_values):
        exact = variance[0, column].sqrt()
        print("ADAMW_NATIVE_SQRT_DIAG " + json.dumps(dict(
            gradient=value, variance=float(variance[0, column]),
            native_sqrt=float(root[0, column]), exact_sqrt_of_actual_fp32_variance=float(exact),
            sqrt_relative_error=float((root[0, column] - exact).abs() / exact.clamp_min(1e-30)),
            denominator=float(denominator[0, column]), update=float(update[0, column]))), flush=True)


def diagnose_update(p, gradient, old_m, old_v, new_p, new_m, new_v, values, step, betas):
    """Preserve evidence for the original failing bound; never turn it into a pass."""
    expected, envelope = expected64(p, gradient, old_m, old_v, values)
    actual = new_p.detach().cpu().double()
    u = 2.0 ** -24
    bound = 4 * (32 * u / (1 - 32 * u)) * envelope[0] + 32 * 2.0 ** -149
    indices = ((actual - expected[0]).abs() > bound).flatten().nonzero().flatten()[:8].tolist()
    lr, b1, b2, om1, om2, decay, eps = values.double().unbind()
    for index in indices:
        old, g, m, v = [x.flatten()[index].clone() for x in (p, gradient, old_m, old_v)]
        nm, nv = [x.flatten()[index].double() for x in (new_m, new_v)]
        conditional = old.double() * decay - lr * (nm / b1) / ((nv / b2).sqrt() + eps)
        cpu_m = m + (g - m) * values[3]
        cpu_v = v + (g.square() - v) * values[4]
        cpu_weight = old * values[5] - values[0] * (cpu_m / values[1]) / (
            (cpu_v / values[2]).sqrt() + values[6])
        replicated = [x.expand(256, 128).contiguous().to("neuron") for x in (old, g, m, v)]
        baseline, opt = optimizer_for(*replicated, step, float(values[0]), *betas, .01)
        with torch.no_grad():
            opt.step()
        native_state = opt.state[baseline]
        print("ADAMW_FAILED_ELEMENT_DIAG " + json.dumps(dict(
            flat_index=index, inputs=dict(p=float(old), g=float(g), m=float(m), v=float(v)),
            metadata=values.tolist(), step=step,
            ideal_fp64_weight=float(expected[0].flatten()[index]),
            actual_weight=float(actual.flatten()[index]), original_bound=float(bound.flatten()[index]),
            actual_m=float(nm), ideal_m=float(expected[1].flatten()[index]),
            actual_v=float(nv), ideal_v=float(expected[2].flatten()[index]),
            fp64_weight_using_actual_fp32_states=float(conditional),
            conditional_weight_error=float((actual.flatten()[index] - conditional).abs()),
            propagated_state_rounding=float((conditional - expected[0].flatten()[index]).abs()),
            cpu_ordered_fp32_weight=float(cpu_weight), cpu_m=float(cpu_m), cpu_v=float(cpu_v),
            native_fresh_weight=float(baseline.detach().cpu()[0, 0]),
            native_fresh_m=float(native_state["exp_avg"].cpu()[0, 0]),
            native_fresh_v=float(native_state["exp_avg_sq"].cpu()[0, 0]))), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--math-mode", type=int, choices=(0, 2), default=0)
    args = ap.parse_args()
    print("ADAMW_INTEGRATION_MATH_MODE " + json.dumps(dict(math_mode=args.math_mode)), flush=True)
    import torch_neuronx
    torch.set_num_threads(4)
    T.USE_FOREACH_OPTIM = T.USE_TENSOR_LR_SCALARS = True
    T.USE_CAUTIOUS_WD = False
    cpu_counterexamples()
    diagnose_native_sqrt()
    gen = torch.Generator().manual_seed(7183)
    initial = [torch.randn(8192, 1024, generator=gen) * .03,
               torch.zeros(8192, 1024), torch.randn(12, generator=gen) * .03]
    params, candidate = make_optimizer(initial, True, math_mode=args.math_mode)
    native_params, native = make_optimizer(initial, False)
    ids = [id(p) for p in params]
    trajectory = [(p.double(), torch.zeros_like(p).double(), torch.zeros_like(p).double())
                  for p in initial[:2]]
    trajectory_envelope = [p.double().abs() for p in initial[:2]]
    # Exercise the same Linear BF16 cache refresh on the live LM-head Parameter.
    linear = T.Linear(1024, 8192, bias=False)
    linear.weight = params[0]
    x = torch.randn(2, 1024, generator=gen).to("neuron").bfloat16()
    compiled_projection = torch.compile(linear, backend="neuron", fullgraph=True, dynamic=False)
    last_gradients = None
    for iteration in range(1, 9):
        gradients = [torch.randn(p.shape, generator=gen) * .003 for p in initial]
        if iteration in (2, 7):
            gradients[0][torch.rand(initial[0].shape, generator=gen) < .97] = 0
        if iteration in (3, 7):
            gradients[0].zero_()
        if iteration in (4,):
            gradients[1].zero_()
        if iteration in (1, 5):
            gradients[1] = None
        if iteration == 4:
            gradients[0] = gradients[2] = None
        before = [p.detach().cpu().clone() for p in params]
        states = [state_cpu(candidate, p) for p in params]
        versions = [p._version for p in params]
        native_versions = [p._version for p in native_params]
        lr = 0.0 if iteration == 5 else .3 * (9 - iteration) / 8
        betas = (.8 if iteration < 5 else .9, .96)
        for opt, ps in ((candidate, params), (native, native_params)):
            opt.param_groups[0].update(lr=lr, betas=betas)
            for p, g in zip(ps, gradients):
                p.grad = None if g is None else g.to("neuron")
            opt.step()
        assert ids == [id(p) for p in params]
        for index, (p, g, old, state, version) in enumerate(zip(params, gradients, before, states, versions)):
            new_state = state_cpu(candidate, p)
            if g is None:
                assert torch.equal(p.detach().cpu(), old) and p._version == version
                assert new_state["step"] == state["step"]
                assert torch.equal(new_state["exp_avg"], state["exp_avg"])
                assert torch.equal(new_state["exp_avg_sq"], state["exp_avg_sq"])
                continue
            assert torch.equal(p.grad.cpu(), g), "gradient mutated"
            assert new_state["step"] == state["step"] + 1
            if index == 2:
                # This group deliberately uses the original native foreach
                # implementation. Require its exact native version behavior;
                # do not impose copy_'s contract on a different backend op.
                delta = p._version - version
                native_delta = native_params[index]._version - native_versions[index]
                assert delta == native_delta, (iteration, index, delta, native_delta)
                print("ADAMW_FALLBACK_VERSION " + json.dumps(dict(
                    iteration=iteration, parameter=index, version_delta=delta,
                    native_version_delta=native_delta, native_equivalent=True)), flush=True)
                assert torch.equal(p.detach().cpu(), native_params[index].detach().cpu())
                ns = state_cpu(native, native_params[index])
                assert all(torch.equal(new_state[k], ns[k]) for k in ("exp_avg", "exp_avg_sq"))
                continue
            assert p._version > version, ("fused Parameter version did not advance", iteration, index)
            values = scalar_values(lr, new_state["step"], *betas, .01)
            try:
                check_stages(f"integration:iteration{iteration}:parameter{index}",
                             (p, new_state["exp_avg"], new_state["exp_avg_sq"]),
                             (old, g, state["exp_avg"], state["exp_avg_sq"]), values)
            except AssertionError:
                diagnose_update(old, g, state["exp_avg"], state["exp_avg_sq"], p,
                                new_state["exp_avg"], new_state["exp_avg_sq"], values,
                                new_state["step"], betas)
                raise
            tp, tm, tv = trajectory[index]
            expected_trajectory, _ = expected64(tp, g, tm, tv, values)
            trajectory_envelope[index] += (expected_trajectory[0] - tp * float(values[5])).abs()
            trajectory[index] = expected_trajectory
            actual = (p.detach().cpu().double(), new_state["exp_avg"].double(),
                      new_state["exp_avg_sq"].double())
            errors = [float((a - e).norm() / e.norm().clamp_min(1e-30))
                      for a, e in zip(actual, expected_trajectory)]
            # The separately declared accumulated-trajectory bound is unchanged.
            # Per-update checks now account for stored-state rounding explicitly;
            # the derivation and original counterexamples remain in the ledger.
            u = 2.0 ** -24
            operations = 32 * new_state["step"]
            gamma = operations * u / (1 - operations * u)
            failures = int(((actual[0] - expected_trajectory[0]).abs()
                            > 4 * gamma * trajectory_envelope[index] + 32 * 2.0 ** -149).sum())
            assert not failures and max(errors) < 2e-6, (iteration, index, failures, errors)
            if lr == 0:
                assert torch.equal(p.detach().cpu(), old)
            print("ADAMW_TRAJECTORY " + json.dumps(dict(iteration=iteration, parameter=index,
                  updates=new_state["step"], relative_rms=errors, bound_failures=failures)), flush=True)
        with torch.no_grad():
            linear.refresh_weight_cache(torch.bfloat16)
            assert torch.equal(linear._cached_weight.cpu(), params[0].detach().cpu().bfloat16())
            cached = compiled_projection(x)
            reference = torch.nn.functional.linear(x, params[0].bfloat16())
            torch.testing.assert_close(cached.cpu(), reference.cpu(), rtol=0, atol=0)
        last_gradients = gradients
    # A skipped optimizer tick must not update moments, counters or weights.
    saved = cpu_tree(candidate.state_dict())
    weights = [p.detach().cpu().clone() for p in params]
    candidate.step(update_adamw=False)
    assert all(torch.equal(p.detach().cpu(), old) for p, old in zip(params, weights))
    for index, p in enumerate(params):
        assert candidate.state[p]["step"] == saved["state"][index]["step"]
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(candidate.state[p][key].cpu(), saved["state"][index][key])
    # Portable CPU serialization, then normal Optimizer.load_state_dict on a
    # fresh instance. Transient scalar buffers are deliberately not serialized.
    buffer = io.BytesIO()
    torch.save(dict(weights=weights, optimizer=saved), buffer)
    buffer.seek(0)
    payload = torch.load(buffer, map_location="cpu", weights_only=True)
    restored_params, restored = make_optimizer(payload["weights"], True, math_mode=args.math_mode)
    restored.load_state_dict(payload["optimizer"])
    for opt, ps in ((candidate, params), (restored, restored_params)):
        for p, g in zip(ps, last_gradients):
            p.grad = None if g is None else g.to("neuron")
        opt.step()
    for p, q in zip(params, restored_params):
        assert torch.equal(p.detach().cpu(), q.detach().cpu())
        assert candidate.state[p]["step"] == restored.state[q]["step"]
        assert set(candidate.state[p]) == {"step", "exp_avg", "exp_avg_sq"}
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(candidate.state[p][key].cpu(), restored.state[q][key].cpu())
    assert candidate._fresh_start_adamw_calls == 15
    print("PASS: repeated real-table analytic/trajectory gates, per-parameter counters, "
          "zero/sparse/None gradients, LR/beta changes, native fallback, Parameter "
          "identity/version, BF16 cache, skipped tick and bit-exact state restore", flush=True)


if __name__ == "__main__":
    main()
