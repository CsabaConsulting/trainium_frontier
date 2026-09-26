"""Offline authentication of an unresolved run, never launch/recovery authority."""
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / "research/v2/operator/20260926-clean-native-trn21-launch"
checks = []


def check(ok, name):
    if not ok:
        raise RuntimeError(name)
    checks.append(name)


def read(name):
    return json.loads((RAW / name).read_text())


for name, expected in {
    "cache-entry-metadata.json": "5e867b0c87182ba228e0fe30cedc21b25874817860f2dbf995c061aab301d6cb",
    "explicit-postflight.json": "d60637b60143398cb757332eed3bf7a74a0a9cf4cfa966e9f23a314cb6c70584",
}.items():
    check(hashlib.sha256((RAW / name).read_bytes()).hexdigest() == expected, name + ":pin")

metadata = read("cache-entry-metadata.json")["remote"]
check(metadata["link"]["symlink"] and not metadata["link"]["regular"], "actual_link")
check(metadata["link"]["inode"] == 3187099 and metadata["link"]["bytes"] == 121, "link_identity")
check(metadata["literal_target"]["regular"] and metadata["literal_target"]["bytes"] == 4096, "sdk_target_metadata")
check(metadata["target_contents_read"] is False and metadata["link_unchanged"], "no_target_contents")

post = read("explicit-postflight.json")["remote"]
check(post["lock"] == {"device": 66305, "free": False, "inode": 596823}, "original_lock_still_held")
check(post["host_reusable"] is False and post["no_signals"] and post["no_host_writes"], "no_recovery_claim")
ids = post["observed_identities"]
check(len(ids) == 43, "43_observed_identities")
live = {int(pid) for pid, item in ids.items() if item["exists"]}
check(live == {3202396, 3202397, 3202398}, "only_parked_chain_live")
check(all(item["same_identity"] for pid, item in ids.items() if int(pid) in live), "parked_identities_unchanged")
check(all(not text.strip() for text in post["guardian_thread_children"].values()), "guardian_has_no_children")
census = post["census"]
check(len(census) == 267, "267_process_census")
check(not any(row["ppid"] == 3202398 for row in census), "no_guardian_child_in_census")
check(not any(any(word in row["comm"].lower() for word in ("torchrun", "neuronx-cc")) for row in census), "no_named_training_compiler")
devices = json.loads(post["checks"]["neuron_ls"]["stdout"])
check(post["checks"]["neuron_ls"]["returncode"] == 0 and all(not d["neuron_processes"] for d in devices), "device_idle")
check(post["checks"]["fuser"] == {"returncode": 1, "stderr": "", "stdout": ""}, "no_device_holders")

wire_path = RAW / "retrieval-wire.jsonl"
wire = [json.loads(line) for line in wire_path.read_text().splitlines()]
outcome = read("retrieval-outcome.json")
check(hashlib.sha256(wire_path.read_bytes()).hexdigest() == outcome["wire_sha256"], "wire_pin")
check(outcome["returncode"] == 2 and "RETRIEVAL_NONREGULAR" in outcome["receiver_error"], "retrieval_failed_honestly")
active = None
verified = 0
total = 0
for record in wire:
    if record["kind"] == "file":
        check(active is None, "wire_file_boundary")
        active, contents = record, bytearray()
    elif record["kind"] == "chunk":
        check(active is not None, "chunk_has_header")
        contents.extend(base64.b64decode(record["b64"], validate=True))
    elif record["kind"] == "end_file":
        check(active is not None, "end_has_header")
        check(len(contents) == active["bytes"] and hashlib.sha256(contents).hexdigest() == active["sha256"], "authenticated_partial:" + active["name"])
        check((RAW / "retrieved/files" / active["name"]).read_bytes() == contents, "local_partial:" + active["name"])
        verified += 1
        total += len(contents)
        active = None
check(active is None and verified == 27 and total == 230411, "27_complete_partial_files")
check(wire[-1]["kind"] == "error", "no_completeness_claim")
events = [json.loads(line) for line in (RAW / "retrieved/files/lifetime-events.jsonl").read_text().splitlines()]
check(events[-1]["event"] == "FAILED_LOCK_RETAINED", "guardian_parked")
check(any(e["event"] == "lineage_quiescent" and e.get("proof") == "waitpid_ECHILD" for e in events), "lineage_kernel_echild")
check(read("retrieved/files/terminal.json")["status"] == "UNRESOLVED", "terminal_unresolved")
print(json.dumps({"status": "UNRESOLVED_FAILURE_EVIDENCE_AUTHENTICATED", "checks": len(checks), "names": checks,
                  "observed_job_compiler_identities_gone": 40, "guardian_lock_held": True,
                  "partial_files_verified": verified, "partial_bytes": total,
                  "retrieval_complete": False, "native_pass": False, "hold_release_authorized": False}, indent=2))
