"""Tiny CPU-only compile-boundary probe, not a Neuron backward test."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE_SHA = "921271d713f53a9d952633ba98a1704c11937235ce974f6b9fcd2cbded4ad858"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def metric(actual, expected):
    actual, expected = actual.detach().float(), expected.detach().float()
    return dict(relative_rms=float((actual - expected).square().mean().sqrt()
                                  / expected.square().mean().sqrt().clamp_min(1e-12)),
                actual_norm=float(actual.norm()), expected_norm=float(expected.norm()),
                actual_nonzero=int(torch.count_nonzero(actual)),
                expected_nonzero=int(torch.count_nonzero(expected)),
                exact=bool(torch.equal(actual, expected)))


def main():
    raw = subprocess.check_output([
        "git", "show", "50a901374ae6f76b35b8556d32fd72b1ece28fc6:workflow/proposals/research-v2/train.py"
    ], cwd=ROOT)
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    tree = ast.parse(raw)
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                and n.name in {"_fp8_row_reference", "_FP8Projection", "Linear"}]
    assert len(selected) == 3
    ns = dict(torch=torch, nn=nn, F=F, USE_WEIGHT_CACHE=False)
    exec(compile(ast.Module(body=selected, type_ignores=[]), "exact9212-fp8-extract", "exec"), ns)
    quantize, apply, Linear = ns["_fp8_row_reference"], ns["_FP8Projection"].apply, ns["Linear"]
    torch.manual_seed(42)
    x = torch.randn(8, 16).to(torch.bfloat16)
    w = torch.randn(16, 16) * (16 ** -0.5)
    gy = torch.randn(8, 16).to(torch.bfloat16)
    qx, sx, ex = quantize(x)
    qw, sw, ew = quantize(w)
    expected_y = ((qx @ qw.T) * sw.T * sx).to(torch.bfloat16)
    expected_dx, expected_dw = gy @ ew, (gy.T @ ex).float()

    def enclosing(x, weight):
        return ns["_FP8Projection"].apply(x, weight)

    results = {}
    for mode in ("eager_apply", "aot_eager_bound_apply", "aot_eager_enclosing_function", "aot_eager_production_Linear"):
        xn, wn = x.clone().requires_grad_(), w.clone().requires_grad_()
        if mode == "eager_apply":
            y = apply(xn, wn)
        elif mode == "aot_eager_bound_apply":
            y = torch.compile(apply, backend="aot_eager", fullgraph=True, dynamic=False)(xn, wn)
        elif mode == "aot_eager_enclosing_function":
            y = torch.compile(enclosing, backend="aot_eager", fullgraph=True, dynamic=False)(xn, wn)
        else:
            linear = Linear(16, 16, bias=False, fp8_projection=True)
            with torch.no_grad():
                linear.weight.copy_(w)
            wn = linear.weight
            y = torch.compile(linear, backend="aot_eager", dynamic=False)(xn)
        y.backward(gy)
        results[mode] = dict(forward=metric(y, expected_y), dx=metric(xn.grad, expected_dx), dw=metric(wn.grad, expected_dw))
        assert all(v["exact"] for v in results[mode].values())
        assert results[mode]["dx"]["actual_nonzero"] > 0
        assert results[mode]["dw"]["actual_nonzero"] > 0

    # Check the native-log zero-gradient signature against the exact CPU oracle,
    # without claiming that unrecorded native gradients were actually zero.
    torch.manual_seed(42)
    x = torch.randn(256, 256).to(torch.bfloat16)
    x[0].zero_()
    x[1] *= 1e-8
    x[2, 0] = 100.0
    w = torch.randn(256, 256) * (256 ** -0.5)
    gy = torch.randn(256, 256).to(torch.bfloat16)
    _, _, ex = quantize(x)
    _, _, ew = quantize(w)
    dx, dw = gy @ ew, (gy.T @ ex).float()
    archive = ROOT / "research/v2/operator/20260925-trn21-l1-fp8-native-r1/component"
    manifest = json.loads((archive / "retrieval-and-postflight.json").read_text())
    for name, metadata in manifest["files"].items():
        assert hashlib.sha256((archive / name).read_bytes()).hexdigest() == metadata["sha256"]
    failure_lines = [line for line in (archive / "component.log").read_text().splitlines()
                     if "ValueError: native/reference RMS gate failed: " in line]
    failures = {line.split("]:", 1)[0][1:]: json.loads(line.split("gate failed: ", 1)[1])
                for line in failure_lines}
    assert set(failures) == {"rank0", "rank1", "rank2", "rank3"}
    for result in failures.values():
        assert result["shape"] == [256, 256, 256]
        assert result["dx"]["relative_rms"] == result["dw"]["relative_rms"] == 1.0
        assert result["dx"]["max_abs"] == float(dx.abs().max()) == 4.875
        assert result["dw"]["max_abs"] == float(dw.abs().max()) == 326.0
    output = dict(verdict="CPU_COMPILE_BOUNDARY_PASS_NATIVE_CAUSE_UNRESOLVED", torch=torch.__version__,
                  source_sha256=SOURCE_SHA, backend="aot_eager", device="cpu", seed=42,
                  custom_backward_cases=results,
                  archived_files_hash_verified=len(manifest["files"]), native_failed_ranks=sorted(failures),
                  zero_gradient_hypothesis=dict(dx_reference_absmax=float(dx.abs().max()),
                                                dw_reference_absmax=float(dw.abs().max()),
                                                matches_native_error_maxima=True,
                                                native_zero_gradients_proven=False),
                  native_executed=False, candidate_edited=False, thresholds_changed=False)
    (HERE / "probe.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
