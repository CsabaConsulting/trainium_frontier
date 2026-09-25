"""Operator-requested descriptive supplement; not a fresh-review test or verdict.

Read retained reviewer outputs and recover the seeded input with unchanged
reviewer-authored functions. Never import the candidate or launch a native path.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
os.environ["ATEN_CPU_CAPABILITY"] = "default"  # Match archived reviewer's DEFAULT dispatch.
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "session/reviewer-output"
retention = json.loads((ROOT / "BINARY_RETENTION.json").read_text())["files"][0]
binary = Path(retention["path"])
if hashlib.sha256(binary.read_bytes()).hexdigest() != retention["sha256"]:
    raise SystemExit("reviewer binary hash mismatch")
saved = torch.load(binary, map_location="cpu", weights_only=True)
source = OUT / "reference_exactness_test.py"
if hashlib.sha256(source.read_bytes()).hexdigest() != "24422d083a090a3d92759bbda935a24a13dc73b909f003d91350f4ac4f9d6d84":
    raise SystemExit("reviewer test hash mismatch")
names = {"component_cases", "sequential_reference", "tile_coalescing_emulation"}
nodes = [n for n in ast.parse(source.read_text()).body
         if isinstance(n, ast.FunctionDef) and n.name in names]
if {n.name for n in nodes} != names:
    raise SystemExit("reviewer definitions missing")
scope = {"torch": torch}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
torch.manual_seed(42)
cases = {name: (ids, grad.bfloat16(), vocabulary, channels)
         for name, ids, grad, vocabulary, channels in scope["component_cases"]()}


def sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def sparse(tensor):
    flat = tensor.reshape(-1)
    keep = flat != 0
    if tensor.is_floating_point():
        keep = keep | torch.signbit(flat)  # Preserve negative zero byte identity too.
    positions = torch.nonzero(keep).flatten().tolist()
    rebuilt = torch.zeros_like(flat)
    for i in positions:
        rebuilt[i] = flat[i].item()
    if sha(rebuilt.reshape(tensor.shape)) != sha(tensor):
        raise SystemExit("lossless sparse encoding failed")
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "encoding": "zero filled; flat row-major [index,value] nonzero entries",
            "nonzero": [[i, flat[i].item()] for i in positions], "sha256_raw_bytes": sha(tensor)}


inputs = {}
outputs = {}
metrics = []
for failure in json.loads((OUT / "reference_results.json").read_text())["failures"]:
    name = failure["name"]
    _, kind, method = name.split("/")
    ids, grad, vocabulary, channels = cases[kind]
    actual, expected = saved[name]["actual"], saved[name]["expected"]
    if sha(actual) != failure["actual_sha256"] or sha(expected) != failure["expected_sha256"]:
        raise SystemExit(f"retained output mismatch: {name}")
    # This only authenticates recovered inputs against both retained outputs.
    # These checks are not counted as fresh-reviewer evidence.
    recovered_ref = scope["sequential_reference"](ids, grad, vocabulary, channels)
    recovered_actual = scope["tile_coalescing_emulation"](ids, grad, vocabulary, method)
    if sha(recovered_ref) != sha(expected) or sha(recovered_actual) != sha(actual):
        raise SystemExit(f"seeded input recovery failed: {name}")
    if kind.startswith("cross_tile"):
        original = saved[f"inputs/{kind}"]
        if sha(original["ids"]) != sha(ids) or sha(original["gradient"]) != sha(grad):
            raise SystemExit(f"explicit retained input mismatch: {name}")
    inputs[kind] = {"ids": sparse(ids), "bf16_upstream": sparse(grad),
                    "vocabulary": vocabulary, "channels": channels,
                    "provenance": "Recovered using unchanged reviewer component_cases, seed42; reference and emulated outputs match retained raw hashes"}
    outputs[name] = {"reviewer_emulated_actual_not_native": sparse(actual),
                     "dense_cpu_reference": sparse(expected)}
    delta = actual.double() - expected.double()
    err_l2 = torch.linalg.vector_norm(delta).item()
    ref_l2 = torch.linalg.vector_norm(expected.double()).item()
    max_abs = delta.abs().max().item()
    if max_abs != failure["max_abs_error"]:
        raise SystemExit(f"absolute error mismatch: {name}")
    nonzero_ref = expected != 0
    zero_ref_error = bool(((expected == 0) & (delta != 0)).any().item())
    relative_l2 = err_l2 / ref_l2 if ref_l2 else None
    relative_element = (delta.abs()[nonzero_ref] / expected.double().abs()[nonzero_ref]).max().item() if bool(nonzero_ref.any()) else None
    metrics.append({"name": name, "max_abs_error": max_abs,
                    "error_l2": err_l2, "reference_l2": ref_l2,
                    "absolute_rms_error": delta.square().mean().sqrt().item(),
                    "relative_l2_equals_relative_rms": relative_l2,
                    "relative_norm_status": "finite" if ref_l2 else "undefined: nonzero error with zero reference norm (infinite under extended ratio)",
                    "max_elementwise_relative_error_on_nonzero_reference": relative_element,
                    "nonzero_error_at_zero_reference": zero_ref_error,
                    "native_executed": False, "exactness_pass": False})

destination = Path(__file__).resolve().parent
data = {"authority": "Operator-derived descriptive metrics, not reviewer-authored or a new approval",
        "criterion": "Original fresh review used byte equality, atol=rtol=0; no threshold changed",
        "native_executed": False, "reference_binary_sha256": retention["sha256"],
        "seeded_input_recovery_hash_checks": 12, "cases": metrics}
(destination / "failure_metrics.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
(destination / "failure_inputs_and_outputs.json").write_text(json.dumps({"inputs": inputs, "outputs": outputs}, separators=(",", ":"), sort_keys=True) + "\n")
print(json.dumps(data, indent=2, sort_keys=True))
