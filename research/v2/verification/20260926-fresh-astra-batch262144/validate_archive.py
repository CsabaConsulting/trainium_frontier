"""Stdlib archive identities only; this is not the independent numerical test."""
import hashlib
import json
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
manifest_sha = "eeda2cb3b99b75d614b8f937cfe5843453f77492fca9240b8abd6e7bca7c83d3"
check(digest(packet / "INPUT_MANIFEST.json") == manifest_sha, "input manifest changed")
inputs = json.loads((packet / "INPUT_MANIFEST.json").read_text())["files"]
expected = {"candidate.diff", "lane-spec.md", "raw/identity/default-lockstep.json",
            "raw/identity/default-lockstep-exit.json"}
for mode in ("reference", "default", "batch262144"):
    expected.update(f"raw/{mode}/{name}" for name in
                    ("g2.json", "rank0.json", "rank1.json", "rank2.json", "rank3.json", "torchrun.log"))
check({e["path"] for e in inputs} == expected and len(inputs) == 22, "unexpected input set")
for entry in inputs:
    p = packet / entry["path"]
    check(p.stat().st_size == entry["bytes"], f"input size mismatch: {p}")
    check(digest(p) == entry["sha256"], f"input hash mismatch: {p}")

terminal = json.loads((ROOT / "session/terminal.json").read_text())
check(terminal["exit_code"] == 0, "review operation failed")
check(terminal["prior_history"] is False, "prior context reported")
check(terminal["repository_mounted"] is False, "repository mounted")
check(terminal["model_requested"] == "gpt-6-astra", "wrong selected model")
check(terminal["manifest_sha256"] == manifest_sha, "launched manifest mismatch")
events = [json.loads(line) for line in
          (ROOT / "session/events.jsonl").read_text().splitlines()]
check(events[0]["thread_id"] == "01a0da21-3da3-7f12-a35a-25c0be650a72", "wrong session")
check(any(e["type"] == "turn.completed" for e in events), "turn did not complete")
check(any(e.get("item", {}).get("type") == "command_execution" and
          e["item"].get("exit_code") == 0 and
          "run_recorded.py reference_test_run1" in e["item"].get("command", "")
          for e in events), "independent numerical test was not executed")
out = ROOT / "session/reviewer-output"
manifest = json.loads((out / "artifact_manifest.json").read_text())
check(manifest["input_packet_rehashed_unchanged"] is True, "reviewer input rehash mismatch")
for entry in manifest["files"]:
    p = out / entry["path"]
    check(p.stat().st_size == entry["bytes"], f"output size mismatch: {p}")
    check(digest(p) == entry["sha256"], f"output hash mismatch: {p}")
result = json.loads((out / "result.json").read_text())
check(result["verdict"] == "PASS" and result["blocking_findings_in_assigned_scope"] == [],
      "reviewer source verdict changed")
check(len(result["refusals"]) == 3 and all(r["verdict"] == "REFUSE" for r in result["refusals"]),
      "broader refusal scope changed")
check(result["test"]["exit_code"] == 0, "reviewer test failed")
own = json.loads((out / "reference_results.json").read_text())
check(own["verdict"] == "PASS" and own["failures"] == [], "own test results failed")
check(len(own["checks"]) == 352 and all(c["passed"] for c in own["checks"]),
      "own check count/results mismatch")
check(len(own["cases"]) == 11, "own case count mismatch")
check(own["criteria"]["default_identity"] == "byte-for-byte, no tolerance",
      "default identity criterion changed")
check(own["settings"]["intraop"] == own["settings"]["interop"] == 1, "thread bound mismatch")
check(own["bounds"]["parameter_elements_per_model"] == 165, "tiny model scope mismatch")
command = json.loads((out / "reference_test_run1.command.json").read_text())
check(command["exit_code"] == 0 and command["command"][-1] == "/work/reference_test.py",
      "actual test command mismatch")
check(not list(ROOT.rglob("*.pt")), "binary tensors must not be committed")
print(json.dumps({"status": "PASS", "stdlib_archive_checks": checks,
                  "full_reviewer_test_rerun_by_operator": False}, sort_keys=True))
