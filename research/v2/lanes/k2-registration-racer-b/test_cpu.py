"""Own CPU tests, with the installed nki_op AST and real local torch.library.

No TorchNeuron/NKI import, native execution, package installation or training.
All tensors are V8/C4/T8; the ordered scalar oracle is independent of the
tensor consumer substitute. An intentionally wrong consumer tests participation.
"""
import ast
from collections.abc import Callable, Iterable
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SDK = ROOT / "research/v2/operator/20260926-nki-api-readonly-proposal/results/sources"
PROTECTED = ROOT / "research/v2/lanes/k2-symmetric-consumer"
V, C, T = 8, 4, 8
CHECKS = []
RESULTS = []
FAILURES = []


def check(ok, name):
    CHECKS.append({"name": name, "passed": bool(ok)})
    if not ok:
        FAILURES.append(name)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_adapter():
    spec = importlib.util.spec_from_file_location("racer_b_adapter", HERE / "adapter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def static_checks():
    pins = {
        PROTECTED / "consumer_nki.py": "2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae",
        PROTECTED / "reference.py": "14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f",
        PROTECTED / "check_native.py": "89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850",
        SDK / "nki_hop.py": "a8b7bffbf461378b54e6f5b4ad3406b80e069a3d89122712c40df79ebef08914",
        SDK / "nki_kernel.py": "37b9c6f56388695fd5dd44a4638576461d8d075f9508cb80b0df9b04dfbd10a2",
        SDK / "nki-framework/kernel.py": "efef333640db8aae6805119ebed619f7f7114f8b7d6974114d0dc12a6c61f83a",
        SDK / "nki-framework/compiled.py": "c416d73c01fd8810509c4496758a122df941e109221fd0ee9a47d30f710b378e",
    }
    for path, expected in pins.items():
        check(digest(path.read_bytes()) == expected, "pin:" + str(path.relative_to(ROOT)))
    source = ast.parse((HERE / "adapter.py").read_text())
    original = ast.parse((PROTECTED / "check_native.py").read_text())
    classes = [n for n in ast.walk(source) if isinstance(n, ast.ClassDef)]
    original_class = next(n for n in ast.walk(original) if isinstance(n, ast.ClassDef) and n.name == "DiagnosticEmbedding")
    check(len(classes) == 1 and ast.dump(classes[0]) == ast.dump(original_class), "unchanged_original_failure_class_AST")
    calls = [n for n in ast.walk(source) if isinstance(n, ast.Call)]
    check(sum(isinstance(n.func, ast.Name) and n.func.id == "nki_op" for n in calls) == 1, "one_forward_registration_site")
    check(not any(isinstance(n.func, ast.Name) and n.func.id == "wrap_nki" for n in calls), "no_explicit_wrap")
    return {str(p.relative_to(ROOT)): h for p, h in pins.items()}


def extracted_nki_op(torch):
    from torch._library.custom_ops import custom_op
    from torch._library.infer_schema import infer_schema
    source = (SDK / "nki_hop.py").read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "nki_op")
    before = ast.dump(node, include_attributes=True)
    environment = dict(torch=torch, Callable=Callable, Iterable=Iterable,
                       custom_op=custom_op, infer_schema=infer_schema)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SDK / "nki_hop.py"), "exec"), environment)
    check(ast.dump(node, include_attributes=True) == before, "installed_nki_op_AST_unmodified")
    return environment["nki_op"], digest(before.encode())


def raw(tensor):
    return bytes(tensor.detach().contiguous().view(torch.uint8).reshape(-1).tolist())


def fingerprint(tensor):
    return {"sha256": digest(raw(tensor)), "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "stride": list(tensor.stride())}


def ordered_oracle(ids, gradient):
    # Scalar token-order recurrence, one IEEE FP32 rounding after each addition.
    result = [[0.0 for _ in range(C)] for _ in range(V)]
    for token, row in enumerate(ids):
        for channel in range(C):
            result[row][channel] = struct.unpack("<f", struct.pack("<f", result[row][channel] + gradient[token][channel]))[0]
    return torch.tensor(result, dtype=torch.float32)


def make_metadata(ids):
    positions, starts, counts = [], [], []
    for row in range(V):
        token_positions = [i for i, token_id in enumerate(ids) if token_id == row]
        starts.append(len(positions))
        counts.append(len(token_positions))
        positions.extend(token_positions)
    maximums = [max(counts[:4]), max(counts[4:])]
    positions += [T] * (T - len(positions))
    return tuple(torch.tensor(values, dtype=torch.int32).reshape(-1, 1)
                 for values in (counts, starts, positions, maximums))


def cpu_consumer(gradient, counts, starts, positions, maximums):
    """Test substitute only: per-row occurrence loop, no additional custom op."""
    padded = torch.cat((gradient.float(), gradient.new_zeros((T + 1 - gradient.shape[0], C)).float()), dim=0)
    accumulator = torch.zeros((V, C), dtype=torch.float32)
    for occurrence in range(T):
        active = (counts[:, 0] > occurrence) & (maximums[:, 0].repeat_interleave(4) > occurrence)
        addresses = torch.where(active, starts[:, 0] + occurrence, T - 1).long()
        tokens = positions[:, 0].index_select(0, addresses).long()
        tokens = torch.where(active, tokens, T)
        values = padded.index_select(0, tokens)
        accumulator = torch.where(active[:, None], accumulator + values, accumulator)
    return accumulator


def case(name, dtype, tangent=0):
    ids = list(range(T))
    if name in ("duplicates", "cancellation", "signed_zero", "unused"):
        ids = [0, 7, 0, 3, 7, 0, 3, 0]
    if name == "changed_metadata":
        ids = [6, 1, 6, 1, 1, 6, 1, 6]
    if name == "empty":
        ids = []
    values = [[float(((token * 7 + channel * 3 + tangent * 5) % 19) - 9) / 8 for channel in range(C)] for token in range(len(ids))]
    if name == "cancellation":
        for token, value in zip((0, 2, 5, 7), (2.0**24, 0.0, 1.0, -(2.0**24))):
            values[token] = [value] * C
    if name == "signed_zero":
        values = [[-0.0 if (token + channel) % 2 else 0.0 for channel in range(C)] for token in range(len(ids))]
    gradient = torch.tensor(values, dtype=dtype).reshape(len(ids), C)
    if tangent == 1:
        gradient = gradient.t().contiguous().t()
    weight = torch.arange(V * C, dtype=torch.float32).reshape(V, C) / 8 - 2
    weight[0, 0] = -0.0
    index = torch.tensor(ids, dtype=torch.int64)
    inputs = (weight.requires_grad_(), index, *make_metadata(ids))
    expected_forward = torch.tensor([weight.detach()[i].tolist() for i in ids], dtype=dtype).reshape(len(ids), C)
    return inputs, gradient, expected_forward, ordered_oracle(ids, gradient.float().tolist())


def strict_record(actual, expected, name):
    valid = isinstance(actual, torch.Tensor)
    shape_dtype = valid and actual.shape == expected.shape and actual.dtype == expected.dtype
    finite = valid and bool(torch.isfinite(actual).all())
    identical = shape_dtype and finite and raw(actual) == raw(expected)
    record = {"name": name, "equal": identical, "expected": fingerprint(expected)}
    if valid:
        record["actual"] = fingerprint(actual)
        record["nonzero"] = int(torch.count_nonzero(actual))
        record["finite"] = finite
        if shape_dtype:
            maximum = float((actual.detach().double() - expected.detach().double()).abs().max()) if actual.numel() else 0.0
            record["max_abs"] = maximum if math.isfinite(maximum) else None
    else:
        record["actual_type"] = type(actual).__name__
    RESULTS.append(record)
    return identical


def run_route(function, inputs, gradient, expected_forward, expected_dw, label):
    before = [fingerprint(x) for x in (*inputs, gradient)]
    output = function(*inputs)
    dw, = torch.autograd.grad(output, inputs[0], gradient)
    forward_ok = strict_record(output, expected_forward, label + ":forward")
    dw_ok = strict_record(dw, expected_dw, label + ":dW")
    after = [fingerprint(x) for x in (*inputs, gradient)]
    RESULTS.append({"name": label + ":inputs", "before": before, "after": after})
    check(before == after, label + ":immutable_inputs")
    return forward_ok, dw_ok


def main():
    started = time.time()
    adapter = load_adapter()
    check("torch" not in sys.modules, "adapter_import_has_no_torch")
    try:
        adapter.build()
    except RuntimeError as exc:
        check(str(exc).startswith("EXPERIMENT_DISABLED"), "disabled_refuses")
    check("torch" not in sys.modules and "nki" not in sys.modules and "torch_neuronx" not in sys.modules, "disabled_imports_no_sdk_or_torch")
    pins = static_checks()
    global torch
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    nki_op, extracted_sha = extracted_nki_op(torch)
    check(torch.get_num_threads() == torch.get_num_interop_threads() == 1, "single_thread")
    # A tiny constructor accounting cap, including repeats, is stricter than64MiB.
    check((V * C * 4 + 4 * T * 8) * 1000 < 64 * 1024 * 1024, "fixture_tensor_bound_below_64MiB")
    for dtype_name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        factory_calls = []
        def factory(*, enabled):
            factory_calls.append(enabled)
            return cpu_consumer
        candidate, old_control = adapter.build(enabled=True, dtype=dtype_name,
                                               test_kernel_factory=factory, test_nki_op=nki_op)
        check(factory_calls == [True], dtype_name + ":factory_enabled_once")
        from torch._subclasses.functional_tensor import FunctionalTensorMode
        check(candidate._abstract_fn is not None, dtype_name + ":fake_registered")
        check(FunctionalTensorMode in candidate._torch_dispatch_fns, dtype_name + ":functional_decomposition_registered")
        check(candidate._backward_fn is not None and candidate._setup_context_fn is not None, dtype_name + ":autograd_registered")
        check(not any(a.alias_info for a in candidate._opoverload._schema.arguments), dtype_name + ":six_inputs_no_alias_mutation")
        check(len(candidate._opoverload._schema.arguments) == 6, dtype_name + ":six_tensor_inputs")
        from torch._subclasses.fake_tensor import FakeTensorMode
        fake_inputs, _, _, _ = case("unique", dtype)
        with FakeTensorMode() as fake_mode:
            fake_output = candidate(*(fake_mode.from_tensor(x) for x in fake_inputs))
            check(fake_output.shape == (T, C) and fake_output.dtype == dtype,
                  dtype_name + ":fake_execution_shape_dtype")
        def ordinary(weight, ids, counts, starts, positions, maximums):
            return torch.nn.functional.embedding(ids, weight).to(dtype)
        routes = [("candidate_eager", candidate), ("candidate_aot", torch.compile(candidate, backend="aot_eager", fullgraph=True, dynamic=False)),
                  ("original_eager", old_control), ("original_aot", torch.compile(old_control, backend="aot_eager", fullgraph=True, dynamic=False)),
                  ("ordinary_eager", ordinary), ("ordinary_aot", torch.compile(ordinary, backend="aot_eager", fullgraph=True, dynamic=False))]
        for name in ("unique", "duplicates", "cancellation", "unused", "signed_zero", "changed_metadata", "empty"):
            for tangent in (0, 1) if name in ("unique", "changed_metadata") else (0,):
                inputs, gradient, expected_forward, expected_dw = case(name, dtype, tangent)
                if name == "unique":
                    check(bool(torch.count_nonzero(expected_dw)), dtype_name + ":nonzero_unique_sentinel")
                if tangent == 1:
                    check(not gradient.is_contiguous(), dtype_name + ":noncontiguous_tangent")
                if name == "cancellation":
                    ids_list = inputs[1].tolist()
                    grad_list = gradient.float().tolist()
                    grouped = ordered_oracle(ids_list[:4], grad_list[:4]) + ordered_oracle(ids_list[4:], grad_list[4:])
                    check(raw(grouped) != raw(expected_dw), dtype_name + ":oracle_rejects_grouped_sum")
                direct = cpu_consumer(gradient, *inputs[2:])
                check(strict_record(direct, expected_dw, f"{dtype_name}:{name}:{tangent}:direct"), f"{dtype_name}:{name}:{tangent}:direct_exact")
                for mode, fn in routes:
                    label = f"{dtype_name}:{name}:{tangent}:{mode}"
                    try:
                        forward_ok, dw_ok = run_route(fn, inputs, gradient, expected_forward, expected_dw, label)
                        check(forward_ok and dw_ok, label + ":strict_bytes")
                    except Exception as exc:
                        RESULTS.append({"name": label, "exception": type(exc).__name__, "error": str(exc)})
                        check(False, label + ":execution")
        # Distinct custom forward registrations ONLY for negative test doubles.
        # A correct-looking dW from a poison consumer means derivative bypass.
        inputs, gradient, expected_forward, expected_dw = case("unique", dtype)
        for fault in ("zero", "wrong"):
            def poisoned(gradient, counts, starts, positions, maximums):
                result = cpu_consumer(gradient, counts, starts, positions, maximums)
                return torch.zeros_like(result) if fault == "zero" else result + 1
            bad = adapter._register(poisoned, dtype, nki_op, f"k2_racer_b_negative::{dtype_name}_{fault}")
            for mode, fn in (("eager", bad), ("aot", torch.compile(bad, backend="aot_eager", fullgraph=True, dynamic=False))):
                label = f"{dtype_name}:consumer_{fault}:{mode}"
                try:
                    forward_ok, dw_ok = run_route(fn, inputs, gradient, expected_forward, expected_dw, label)
                    check(forward_ok and not dw_ok, label + ":consumer_participation")
                except Exception as exc:
                    RESULTS.append({"name": label, "exception": type(exc).__name__, "error": str(exc)})
                    check(False, label + ":execution")
        bads = [None, torch.zeros_like(expected_dw), expected_dw + 1,
                torch.full_like(expected_dw, float("nan")), torch.full_like(expected_dw, float("inf")),
                expected_dw.to(torch.bfloat16), expected_dw[:4], case("unique", dtype, tangent=1)[3]]
        for index, bad in enumerate(bads):
            check(not strict_record(bad, expected_dw, f"{dtype_name}:validator_negative:{index}"), f"{dtype_name}:validator_refuses:{index}")
    check(not any(name == "nki" or name.startswith("nki.") or name.startswith("torch_neuronx") for name in sys.modules), "no_sdk_import_at_end")
    report = {"status": "PASS_OWN_CPU_ONLY" if not FAILURES else "REFUSE_OWN_CPU_GATE",
              "torch_version": str(torch.__version__), "python_version": sys.version,
              "backend": "aot_eager/fullgraph/dynamic=False", "native_torch_version": "2.11 (not executed)",
              "installed_nki_op_AST_sha256": extracted_sha, "pins": pins,
              "adapter_sha256": digest((HERE / "adapter.py").read_bytes()),
              "test_sha256": digest(Path(__file__).read_bytes()), "elapsed_seconds": time.time() - started,
              "checks": CHECKS, "failures": FAILURES, "results": RESULTS,
              "native_correctness": "UNTESTED", "performance": "UNMEASURED", "production_activation": False}
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
