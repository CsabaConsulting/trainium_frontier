"""Authenticate the observed live process, not inferred completion or BPB."""
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "research/v2/operator"
path = BASE / "20260926-trn22-status-0537/trn22.json"
count = 0


def check(value, message):
    global count
    if not value:
        raise RuntimeError(message)
    count += 1


raw = path.read_bytes()
check(hashlib.sha256(raw).hexdigest() == "42c3441cc40d7e225ed8c1506785ff8d3b056c4927810f9acc0d0c5a7ecb3293", "snapshot pin")
record = json.loads(raw)
check(record["returncode"] == 0 and record["stderr"] == "", "successful read-only transport")
r = record["remote"]
check(r["no_changes"] is True, "read only")
check((r["lock"]["device"], r["lock"]["inode"], r["lock"]["held"]) == (66305, 832589, True), "original held lock")
check(r["ps_rc"] == r["neuron_ls_rc"] == 0, "process/device commands succeeded")
check(len(r["files"]) == 8, "eight complete files")
for name, f in r["files"].items():
    b = f["text"].encode()
    check(f["capture"] == "complete_at_observation" and len(b) == f["captured_bytes"] == f["bytes_at_stat"] and hashlib.sha256(b).hexdigest() == f["sha256_captured"], name)
prior = json.loads((BASE / "20260926-trn22-confirmed36-clean-cold-full-r2/trn22.json").read_text())["remote"]
old = {p["pid"]: p for p in prior["processes"]}
now = {p["pid"]: p for p in r["processes"]}
for pid in (3281907, 3281908, 3281909, 3281925, 3281926, 3281940, 3281955, 3281956, 3281957, 3281958):
    p = now[pid]
    check(p["exists"] and p["identity_stable"], "live identity " + str(pid))
    check(all(p[k] == old[pid][k] for k in ("starttime", "ppid", "pgid", "sid", "argv", "cwd", "environment_whitelist")), "same launch identity/env " + str(pid))
check({p["pid"] for p in r["neuron_ls"][0]["neuron_processes"]} == {3281955, 3281956, 3281957, 3281958}, "original four device workers")
markers = [int(x) for x in r["files"]["train/steps.jsonl"]["text"].splitlines()]
check(markers == list(range(3498)), "3498 sequential completed markers")
log = r["files"]["training.log"]["text"]
rows = re.findall(r"^step (\d+) \| loss ([0-9.]+).*?tok/s ([0-9,]+).*?remaining_time (\d+)s", log, re.M)
check(all(math.isfinite(float(row[1])) for row in rows), "finite printed losses")
check(rows[-1] == ("03490", "2.7453", "268,353", "83"), "latest printed evidence")
check(not re.search(r"Traceback|RuntimeError|AssertionError|FATAL", log), "no observed training error")
check("terminal.json" not in r["files"] and "fresh-public20m.log" not in r["files"], "no terminal or eval inferred")
print(json.dumps({"verdict": "ACCEPT_LIVE_LATE_TRAINING_OBSERVATION_ONLY", "checks_passed": count, "optimized": bool(sys.flags.optimize), "observed_utc": r["utc"], "completed_markers": len(markers), "printed_step": 3490, "tokens_per_second": 268353, "source_remaining_seconds": 83, "public20m": None, "training_complete": False, "submission_ready": False}, sort_keys=True))
