# Direct-wiring Neuron component gate

This is a hardware-only component gate, not training, submission, or a scientific verdict. It
uses the repository `fresh_start_launch.py`/`fresh_start_guard.py` protocol: host-wide lock,
preflight, immutable manifested workload, whole-process timeout, cleanup, and postflight.

## Trust model and frozen contract

`neuron-component-authorization-request.json` is the sole expected-value source. It contains the
complete authorized SHA-256 map and exact request constants. The harness never derives expected
hashes or argv from the files it is testing. Before candidate import it requires every fingerprint
path—including harness, runbook, proposal, authorization request, candidate, launcher, guard,
host/sync helpers and launcher-default manifest inputs—to be tracked and clean at exact HEAD, then
compares their bytes with the authorization map.

The trust chain begins at independent-Verifier rejection commit
`e512c00ffcb6abb31daa4048d0ea3b719564fc6a`. The Implementer must leave the authorization object
`pending_independent_verifier` with `hardware_command_authorized: false`. Only a later independent
Verifier may set it to `authorized`, name the reviewed implementation commit and authorize the
hardware command. Full mode rejects pending status. Static mode can test the pins but grants no
hardware authority.

Frozen invariants include candidate SHA
`c5956187de3302da4a424b17686768c8239606701e0d4c690042944cbf847537`, seed 42,
four host threads/ranks, LNC2, exact compiler flags, host `trn22` / hostname
`ip-172-31-32-253`, unique cache/report paths, and remote checkout
`/home/ubuntu/trainium_freshstart_submission_validation_20260923_direct_component`.

The report is rank-0-only, an exact label-derived child of `research/run-logs`, and created with
`O_EXCL|O_NOFOLLOW`. Existing files/symlinks, unsafe parents, overwrites, or renames fail.

## Exact authorized request and launch

Run this from the validation repository. It aborts unless the independent authorization is active,
the checkout is exact/clean, and every frozen hash matches. Its only generated values are the
unique UTC label and paths deterministically derived from that label.

```bash
REQUEST_PATH=$(python - <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

root = Path.cwd().resolve()
evidence = root / "research/submission-validation/20260923-dense6x1024-direct-review-safe"
authorization_path = evidence / "neuron-component-authorization-request.json"
authorization = json.loads(authorization_path.read_text())
decision = authorization["authorization"]
assert decision == {
    "status": "authorized",
    "hardware_command_authorized": True,
    "authorized_by_role": "independent Verifier",
    "reviewed_implementation_commit": decision["reviewed_implementation_commit"],
}
assert len(decision["reviewed_implementation_commit"]) == 40
commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True
).stdout.strip()
paths = [*authorization["expected_sha256"], str(authorization_path.relative_to(root))]
for relative in paths:
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative], cwd=root, check=True,
        stdout=subprocess.DEVNULL,
    )
for command in (
    ["git", "diff", "--quiet", "HEAD", "--", *paths],
    ["git", "diff", "--cached", "--quiet", "HEAD", "--", *paths],
):
    subprocess.run(command, cwd=root, check=True)
untracked = subprocess.run(
    ["git", "ls-files", "--others", "--exclude-standard", "--", *paths],
    cwd=root, check=True, text=True, capture_output=True,
).stdout
assert not untracked.strip(), "untracked fingerprint input"
for relative, expected in authorization["expected_sha256"].items():
    actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    assert actual == expected, f"authorized SHA mismatch: {relative}"
authorization_sha = hashlib.sha256(authorization_path.read_bytes()).hexdigest()
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
label = f"fs-20260923-dense6x1024-direct-neuron-component-{stamp}"
contract = authorization["request_contract"]
remote = contract["remote"]
report = f"{remote}/research/run-logs/{label}.component.json"
harness = (
    f"{remote}/research/submission-validation/"
    "20260923-dense6x1024-direct-review-safe/neuron_component_direct.py"
)
candidate = f"{remote}/submissions/dense6x1024_direct_review/train.py"
remote_authorization = (
    f"{remote}/research/submission-validation/"
    "20260923-dense6x1024-direct-review-safe/neuron-component-authorization-request.json"
)
environment = dict(contract["environment_static"])
environment.update({
    "NEURON_COMPILE_CACHE_DIR": f"/tmp/neuron-cache-{label}",
    "TRAINIUM_COMPONENT_RUN_LABEL": label,
    "TRAINIUM_COMPONENT_EXPECTED_COMMIT": commit,
    "TRAINIUM_COMPONENT_EXPECTED_HOSTNAME": contract["expected_hostname"],
    "TRAINIUM_COMPONENT_AUTHORIZATION_SHA256": authorization_sha,
})
command = [
    *contract["timeout_argv"],
    *contract["torchrun_argv_prefix"],
    harness,
    "--train-py", candidate,
    "--report", report,
    "--run-label", label,
    "--expected-commit", commit,
    "--expected-hostname", contract["expected_hostname"],
    "--authorization-request", remote_authorization,
    "--authorization-request-sha256", authorization_sha,
    "--seed", "42",
    "--host-threads", "4",
]
request = {
    "host": contract["host"],
    "remote": remote,
    "expected_commit": commit,
    "label": label,
    "purpose": contract["purpose"],
    "environment": environment,
    "command": command,
    "fingerprint_paths": paths,
    "authorization_request": str(authorization_path.relative_to(root)),
    "authorization_request_sha256": authorization_sha,
}
path = root / "research/run-logs" / f"{label}.request.json"
with path.open("x") as stream:
    json.dump(request, stream, indent=2)
    stream.write("\n")
print(path)
PY
)
python research/fresh_start_launch.py "$REQUEST_PATH"
```

The authorized `timeout_argv` is exactly `timeout --foreground --signal=TERM --kill-after=60s
1200s`; the authorized torchrun prefix is exactly the native-venv binary, `--standalone`, and
`--nproc_per_node=4`. The harness reconstructs the complete argv array—including every script
option—and requires equality for both launch and job manifests. It likewise requires exact
environment, request keys, fingerprint paths, source hashes, host, remote, label, cache, report,
commit and authorization SHA. Substrings, reordered flags, duplicate overrides, extra arguments,
or changed timeout/signal/grace fail.

The harness-side clean-tree/hash gate compensates fail-closed for older launchers whose remote
preflight dirty list does not include requested fingerprint paths. It runs before candidate import
or any component test. The per-label fresh cache also has an exclusive identity marker; reuse
fails. Do not invoke the guard/harness directly or relaunch an ambiguous label.

## Retrieval and completion

After the guard finishes, sync once or repeat sync only; never repeat launch.

```bash
LABEL=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["label"])' "$REQUEST_PATH")
REMOTE=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["remote"])' "$REQUEST_PATH")
python research/fresh_start_sync.py --rounds 1 --remote "$REMOTE" --prefix "$LABEL" --host trn22
python - "$REQUEST_PATH" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path.cwd().resolve()
request = json.loads(Path(sys.argv[1]).read_text())
label = request["label"]
folder = root / "research/run-logs"
paths = {suffix: folder / f"{label}.{suffix}" for suffix in (
    "launch.json", "job.json", "process.json", "postflight.json", "component.json", "log"
)}
for path in paths.values():
    assert path.is_file() and not path.is_symlink(), f"missing/unsafe receipt: {path}"
launch = json.loads(paths["launch.json"].read_text())
job = json.loads(paths["job.json"].read_text())
process = json.loads(paths["process.json"].read_text())
post = json.loads(paths["postflight.json"].read_text())
report = json.loads(paths["component.json"].read_text())
request_keys = set(request)
augment = {"started_at", "hostname", "git_commit", "source_dirty",
           "preflight_free_bytes", "source_sha256"}
for manifested in (launch, job):
    assert set(manifested) == request_keys | augment
    assert {key: manifested[key] for key in request} == request
assert process["label"] == label and process["command"] == request["command"]
assert post.get("protocol_version") == 2
assert post.get("source_commit") == request["expected_commit"]
assert post.get("label") == label and post.get("remote") == request["remote"]
assert post.get("command_returncode") == 0 and post.get("host_clean") is True
assert not post.get("guard_error") and not post.get("failures") and not post.get("interrupted_signals")
assert report.get("passed") is True
assert report.get("global", {}).get("all_ranks_passed") is True
assert report.get("global", {}).get("identity_digest_all_ranks") is True
assert report.get("global", {}).get("replicated_parameter_digest_all_ranks") is True
assert report.get("distributed_cleanup_succeeded") is True
assert report["identity"]["authorization"]["sha256"] == request["authorization_request_sha256"]
assert paths["log"].read_text().count("DIRECT_NEURON_COMPONENT ") == 1
print(json.dumps({"label": label, "completed": True, "report": str(paths["component.json"])},
                 sort_keys=True))
PY
```

Missing postflight means incomplete, not pass. Any timeout, signal, dirty tree, hash/argv/env/request
mismatch, failed rank, nonzero exit, dirty postflight, or missing result marker is failure.

## Numerical and repeated-execution gates

Every comparison requires finite candidate/reference tensors, a nonzero reference above the
declared minimum, a nonzero candidate, bounded candidate/reference norm ratio, bounded L2
error/reference ratio and elementwise tolerance. Static adversarial checks reject zero, half-scale,
permuted, corrupted and nonfinite fixtures.

| Quantity | atol | rtol | min ref L2 | max error/ref L2 | candidate/ref L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP output | 0.015625 | 0.01 | 1 | 0.01 | 0.98–1.02 |
| MLP input gradient | 0.00025 | 0.02 | 0.000001 | 0.02 | 0.90–1.10 |
| MLP up/down gradient | 0.002 | 0.02 | 0.00001 | 0.02 | 0.90–1.10 |
| CE loss | 0.00002 | 0.00002 | 1 | 0.0001 | 0.999–1.001 |
| CE logit gradient | 0.000002 | 0.0002 | 0.00001 | 0.0005 | 0.99–1.01 |
| Optimizer parameter | 0.0000001 | 0.000001 | 0.001 | 0.000001 | 0.99999–1.00001 |
| Optimizer update delta | 0.0000001 | 0.00001 | 0.00000001 | 0.0001 | 0.99–1.01 |

The BF16 production-trigger MLP executes twice with a real weight update/cache refresh before the
second fused node. Both CE lane grids execute twice through real wrappers/nodes. OwnedOptimizer
performs two different updates, checks full parameters and deltas against an all-rank reference,
requires exact replicated-parameter digests, complete state and second-step round-trip.

## Required local verification

Static success requires the tracked pending authorization request, exact clean HEAD, exact hashes,
all pins and a unique report. Negative tests must change one item at a time and exit nonzero:
dirty harness at unchanged HEAD; dirty/mutated runbook, proposal or authorization request; candidate,
launcher or guard hash; authorization SHA/status; seed/threads/topology/compiler/cache/commit/host;
report outside/name/existing/symlink; exact environment; any argv element/order/extra argument;
timeout duration/foreground/signal/kill grace; request host/label/cache/report/path; and manifest
schema/source hashes. Existing report content must remain unchanged.

A component pass establishes neither full-model timing, checkpoint replay, BPB, reviewer acceptance,
nor submission readiness.
