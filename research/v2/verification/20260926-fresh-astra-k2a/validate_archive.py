"""Stdlib identity/structure audit; never substitutes for reviewer numerical tests."""
import hashlib
import json
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent
checks = 0


def check(condition, message):
    global checks
    if not condition:
        raise SystemExit(message)
    checks += 1


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


packet = ROOT / "packet"
check(digest(packet / "INPUT_MANIFEST.json") ==
      "9dd4f4814a2d4d7e06f7f74538d0495c49792d9108369f9f7650121348aacb2f",
      "input manifest changed")
inputs = json.loads((packet / "INPUT_MANIFEST.json").read_text())["files"]
check(len(inputs) == 10, "unexpected input count")
check({e["path"] for e in inputs} == {
    "candidate.diff", "lane-spec.md",
    "raw/component/cpu-initial.json", "raw/component/cpu-initial-optimized.json",
    "raw/component/cpu-final.json", "raw/component/cpu-final-optimized.json",
    "raw/g2/tiny-base.json", "raw/g2/tiny-default.json",
    "raw/g2/tiny-off.json", "raw/g2/tiny-on.json",
}, "unexpected packet content")
for entry in inputs:
    p = packet / entry["path"]
    check(p.stat().st_size == entry["bytes"], f"input size mismatch: {p}")
    check(digest(p) == entry["sha256"], f"input hash mismatch: {p}")

terminal = json.loads((ROOT / "session/terminal.json").read_text())
check(terminal["exit_code"] == 0, "review operation failed")
check(terminal["prior_history"] is False, "prior context reported")
check(terminal["repository_mounted"] is False, "repository mounted")
check(terminal["model_requested"] == "gpt-6-astra", "wrong selected model")
check(terminal["manifest_sha256"] == digest(packet / "INPUT_MANIFEST.json"),
      "launched manifest mismatch")
events = [json.loads(line) for line in
          (ROOT / "session/events.jsonl").read_text().splitlines()]
check(events[0]["thread_id"] == "01a0da15-9827-7cc0-8845-a39138c52543", "wrong session")
check(any(e["type"] == "turn.completed" for e in events), "turn did not complete")
check(any(e.get("item", {}).get("type") == "command_execution" and
          e["item"].get("exit_code") == 1 and
          "/work/reference_exactness_test.py > /work/reference_test.stdout" in
          e["item"].get("command", "") for e in events), "own failing test did not execute")
out = ROOT / "session/reviewer-output"
external = {Path(e["path"]).name: e for e in
            json.loads((ROOT / "BINARY_RETENTION.json").read_text())["files"]}
for entry in json.loads((out / "ARTIFACT_MANIFEST.json").read_text())["files"]:
    name = Path(entry["path"]).name
    p = out / name
    if name in external:
        check(entry["sha256"] == external[name]["sha256"], "external binary pin mismatch")
        p = Path(external[name]["path"])
    check(p.stat().st_size == entry["bytes"], f"output size mismatch: {p}")
    check(digest(p) == entry["sha256"], f"output hash mismatch: {p}")
result = json.loads((out / "result.json").read_text())
check(result["verdict"] == "REFUSE", "fresh refusal was not retained")
check(result["native_executed"] is False, "native scope changed")
check(result["authority"] == {"hardware_launch_permission": False, "merge_permission": False},
      "authority changed")
own = json.loads((out / "reference_results.json").read_text())
check(own["summary"] == {"actual_cpu_path_failures": 0, "checks": 714, "failures": 6},
      "own test summary changed")
check(own["harness_error"] is None, "own test had harness exception")
check(len(own["failures"]) == 6 and all(e["name"].startswith("emulation/") for e in own["failures"]),
      "failure classifications changed")


def sparse_hash(entry):
    dtype = entry["dtype"]
    size = {"torch.int64": 8, "torch.float32": 4, "torch.bfloat16": 2}[dtype]
    buf = bytearray(math.prod(entry["shape"]) * size)
    for position, value in entry["nonzero"]:
        if dtype == "torch.bfloat16":
            encoded = struct.pack("<f", value)[2:4]
        else:
            encoded = struct.pack("<q" if dtype == "torch.int64" else "<f", value)
        buf[position * size:(position + 1) * size] = encoded
    return hashlib.sha256(buf).hexdigest()


derived = json.loads((ROOT / "operator-derived/failure_inputs_and_outputs.json").read_text())
for inputs in derived["inputs"].values():
    for name in ("ids", "bf16_upstream"):
        check(sparse_hash(inputs[name]) == inputs[name]["sha256_raw_bytes"], "lossless input reconstruction failed")
for failure in own["failures"]:
    outputs = derived["outputs"][failure["name"]]
    for name, field in (("reviewer_emulated_actual_not_native", "actual_sha256"),
                        ("dense_cpu_reference", "expected_sha256")):
        check(sparse_hash(outputs[name]) == failure[field], "lossless output reconstruction failed")
metrics = json.loads((ROOT / "operator-derived/failure_metrics.json").read_text())
check(metrics["native_executed"] is False, "derived native scope changed")
check(len(metrics["cases"]) == 6 and all(not c["exactness_pass"] for c in metrics["cases"]),
      "derived metrics reinterpret failure")
check(not list(ROOT.rglob("*.pt")), "binary tensors must not be committed")
print(json.dumps({"status": "PASS", "stdlib_archive_checks": checks,
                  "full_reviewer_test_rerun_by_operator": False,
                  "operator_input_recovery_reexecuted_reviewer_semantic_functions": True}, sort_keys=True))
