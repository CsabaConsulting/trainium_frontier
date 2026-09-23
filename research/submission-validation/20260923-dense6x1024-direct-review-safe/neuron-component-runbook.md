# Direct-wiring Neuron component gate

This hardware-only gate does not authorize training or submission. It runs the three approved
component checks through the repository's `fresh_start_launch.py` and `fresh_start_guard.py`
protocol: one host-wide lock, clean preflight, bounded workload, unconditional cleanup, and a
postflight receipt. The governing approval is `proposal.json` in this directory.

## Frozen contract

- Candidate: `submissions/dense6x1024_direct_review/train.py`
- Candidate SHA-256: `c5956187de3302da4a424b17686768c8239606701e0d4c690042944cbf847537`
- Seed/host threads: exactly `42` / `4`.
- Topology: exactly four ranks, `WORLD_SIZE=LOCAL_WORLD_SIZE=4`, LNC2.
- Compiler flags: exactly `--optlevel=1 --auto-cast matmult --auto-cast-type bf16`.
- Host: `trn22`, expected hostname `ip-172-31-32-253`.
- Immutable remote checkout: `/home/ubuntu/trainium_freshstart_submission_validation_20260923_direct_component`.
- Every run gets a generated unique label, report, request, and `/tmp/neuron-cache-<label>`.
  Reuse, pre-existing paths, symlinks, wrong commits/hostnames/arguments, or missing guard identity
  fail before candidate import.
- The report is created by rank 0 using `O_EXCL|O_NOFOLLOW` at exactly
  `research/run-logs/<generated-label>.component.json`; the harness never creates its parent,
  overwrites, or renames over a file.

The harness records argv; candidate/harness/runbook/proposal hashes; commit, label, hostname,
environment, cache identity, fresh-start receipts, per-rank stage results, and cross-rank identity
agreement. The process receipt and postflight are necessarily produced by the guard after the
harness exits; they must be retrieved and checked before interpreting the component result.

## Exact manifested launch

Run from the validation repository containing the reviewed commit. The only variable fields are
generated here: the UTC-derived unique label and the paths derived from it. `open(..., "x")`
prevents request reuse.

```bash
REQUEST_PATH=$(python - <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

root = Path.cwd().resolve()
commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True
).stdout.strip()
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
label = f"fs-20260923-dense6x1024-direct-neuron-component-{stamp}"
remote = "/home/ubuntu/trainium_freshstart_submission_validation_20260923_direct_component"
expected_hostname = "ip-172-31-32-253"
cache = f"/tmp/neuron-cache-{label}"
report = f"{remote}/research/run-logs/{label}.component.json"
harness = (
    f"{remote}/research/submission-validation/"
    "20260923-dense6x1024-direct-review-safe/neuron_component_direct.py"
)
candidate = f"{remote}/submissions/dense6x1024_direct_review/train.py"
request = {
    "host": "trn22",
    "remote": remote,
    "expected_commit": commit,
    "label": label,
    "purpose": (
        "Standalone direct-wiring Neuron component gate: repeated fused MLP, repeated adaptive "
        "CE one/two lane, and two-step explicit OwnedOptimizer semantics. No training or verdict."
    ),
    "environment": {
        "PATH": "/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin",
        "NKI_VENV_PATH": "/home/ubuntu/workspace/native_venv",
        "NEURON_LOGICAL_NC_CONFIG": "2",
        "NEURON_CC_FLAGS": "--optlevel=1 --auto-cast matmult --auto-cast-type bf16",
        "NEURON_COMPILE_CACHE_DIR": cache,
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1",
        "TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION": "1",
        "TRAINIUM_COMPONENT_RUN_LABEL": label,
        "TRAINIUM_COMPONENT_EXPECTED_COMMIT": commit,
        "TRAINIUM_COMPONENT_EXPECTED_HOSTNAME": expected_hostname,
    },
    "command": [
        "timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "1200s",
        "/home/ubuntu/workspace/native_venv/bin/torchrun", "--standalone", "--nproc_per_node=4",
        harness,
        "--train-py", candidate,
        "--report", report,
        "--run-label", label,
        "--expected-commit", commit,
        "--expected-hostname", expected_hostname,
        "--seed", "42",
        "--host-threads", "4",
    ],
    "fingerprint_paths": [
        "submissions/dense6x1024_direct_review/train.py",
        "research/submission-validation/20260923-dense6x1024-direct-review-safe/neuron_component_direct.py",
        "research/submission-validation/20260923-dense6x1024-direct-review-safe/neuron-component-runbook.md",
        "research/submission-validation/20260923-dense6x1024-direct-review-safe/proposal.json",
    ],
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

`fresh_start_launch.py` acquires the host lock, validates the immutable clean checkout and exact
commit, runs the standard preflight, fingerprints the four scoped files, writes launch/job
receipts with exclusive creation, and starts `fresh_start_guard.py`. The guard injects
`TRAINIUM_FRESH_START_RUN_ID=<label>`, enforces the whole-process 1,200-second timeout, and always
runs cleanup/postflight. Do not invoke the harness or guard directly and do not relaunch an
ambiguous label.

The expected diagnostic envelope is 480 seconds for the two production-shape MLP executions,
300 seconds for both twice-executed CE grids, and 420 seconds for two optimizer updates plus
collectives/state round-trip. These are not retry authorities; the outer timeout is the gate.

## Retrieval and completion gate

After the remote guard has finished, retrieve the exact generated prefix. Repeating sync is safe;
repeating launch is forbidden.

```bash
LABEL=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["label"])' "$REQUEST_PATH")
REMOTE=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["remote"])' "$REQUEST_PATH")
python research/fresh_start_sync.py --rounds 1 --remote "$REMOTE" --prefix "$LABEL" --host trn22
python - "$REQUEST_PATH" <<'PY'
import json
from pathlib import Path
import sys

root = Path.cwd().resolve()
request = json.loads(Path(sys.argv[1]).read_text())
label = request["label"]
folder = root / "research/run-logs"
post_path = folder / f"{label}.postflight.json"
process_path = folder / f"{label}.process.json"
report_path = folder / f"{label}.component.json"
log_path = folder / f"{label}.log"
for path in (post_path, process_path, report_path, log_path):
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"missing/unsafe receipt: {path}")
post = json.loads(post_path.read_text())
process = json.loads(process_path.read_text())
report = json.loads(report_path.read_text())
if not (
    post.get("protocol_version") == 2
    and post.get("source_commit") == request["expected_commit"]
    and post.get("label") == label
    and post.get("remote") == request["remote"]
    and post.get("command_returncode") == 0
    and post.get("host_clean") is True
    and not post.get("guard_error")
    and not post.get("failures")
    and not post.get("interrupted_signals")
    and process.get("label") == label
    and process.get("command") == request["command"]
    and report.get("passed") is True
    and report.get("global", {}).get("all_ranks_passed") is True
    and report.get("global", {}).get("identity_digest_all_ranks") is True
    and report.get("global", {}).get("replicated_parameter_digest_all_ranks") is True
    and report.get("distributed_cleanup_succeeded") is True
):
    raise SystemExit("FAIL: component result or fresh-start cleanup contract did not pass")
marker = "DIRECT_NEURON_COMPONENT "
if log_path.read_text().count(marker) != 1:
    raise SystemExit("FAIL: expected exactly one rank-0 result marker")
print(json.dumps({"label": label, "completed": True, "report": str(report_path)}, sort_keys=True))
PY
```

Missing postflight means “not yet complete,” never “pass.” A timeout, signal, dirty host, nonzero
exit, receipt mismatch, report mismatch, failed rank, or missing marker is a failed gate.

## Predeclared numerical and execution gates

Every tensor requires finite candidate/reference values, a nonzero reference norm above its
minimum, a nonzero candidate norm, a bounded candidate/reference norm ratio, a bounded L2
error/reference ratio, and elementwise absolute/relative agreement. This prevents all-zero or
attenuated tensors from passing. MLP tolerances cover BF16 matmul rounding; CE and optimizer
remain FP32-tight. The independent CPU reference in `independent-verifier.json` found exact MLP
output/input/up/down-gradient equality; these are finite datatype allowances rather than
observed-error-derived tuning.

| Quantity | atol | rtol | min ref L2 | max error/ref L2 | candidate/ref L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP output | 0.015625 | 0.01 | 1 | 0.01 | 0.98–1.02 |
| MLP input gradient | 0.00025 | 0.02 | 0.000001 | 0.02 | 0.90–1.10 |
| MLP up/down gradient | 0.002 | 0.02 | 0.00001 | 0.02 | 0.90–1.10 |
| CE loss | 0.00002 | 0.00002 | 1 | 0.0001 | 0.999–1.001 |
| CE logit gradient | 0.000002 | 0.0002 | 0.00001 | 0.0005 | 0.99–1.01 |
| Optimizer parameter | 0.0000001 | 0.000001 | 0.001 | 0.000001 | 0.99999–1.00001 |
| Optimizer update delta | 0.0000001 | 0.00001 | 0.00000001 | 0.0001 | 0.99–1.01 |

The production-trigger MLP runs forward/backward twice. Between executions it applies a real,
deterministic gradient update and refreshes both BF16 linear caches; the second fused autograd
node and changed/matching caches are required. CE runs twice at 128 rows/one lane and twice at
512 rows/two lanes; each wrapper call, selected grid, direct node, and later execution is counted.
OwnedOptimizer performs two different deterministic schedules/gradient updates and compares both
full parameters and update deltas against an all-rank reference, then proves omission-free state
and exact round-trip after the second update. All stages and identities must agree on all ranks.

## Local static and negative checks

Static mode performs no Neuron operation. It still requires all pins and a unique in-tree report,
and it proves every declared metric accepts identity while rejecting zero, half-scale, permuted,
single-element-corrupted, and nonfinite fixtures. Use a locally generated `fs-` label/report/cache,
current full commit, and `hostname`; remove only those generated test reports afterward.

Required negative invocations change exactly one field from the successful static command and
must exit nonzero: seed, host threads, LNC, `WORLD_SIZE`, `LOCAL_WORLD_SIZE`, canonical candidate
path, compiler flags, cache path, commit, hostname, label-derived report path, existing report,
and symlink report. The existing-report test must additionally prove its SHA-256 is unchanged.
These local checks do not replace real Neuron.

A passing component report is not a scientific verdict: it establishes neither full-model
compilation/timing, checkpoint replay, validation BPB, reviewer acceptance, nor submission
readiness.
