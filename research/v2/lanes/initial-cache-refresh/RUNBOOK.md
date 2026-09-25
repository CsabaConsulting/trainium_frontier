# Pending-gate handoff (no launch authority)

Production sources/hashes and architecture commit are in IMPLEMENTATION.md.
Separate proposal.json files pin the full native argv and unique labels.
Neither candidate is hardware-ready until independent review and full-size
CPU G2 pass. Operator alone stages and runs through the existing reviewed
manual route, one job/host with fresh role/STOP/idle/device/lock/disk/source
checks. No new launcher is provided. Never use the test import path natively.

## Reproduce tiny self-checks locally

From this packet's repository root, with the existing CPU Python runtime:

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/poojith-devan/venv/bin/python -B research/v2/lanes/initial-cache-refresh/test_initial_cache.py --run-tiny
python3 -B research/v2/lanes/initial-cache-refresh/test_initial_cache.py
python3 -O -B research/v2/lanes/initial-cache-refresh/test_initial_cache.py
```

For the architecture variant, prefix each command with
`CACHE_TEST_VARIANT=architecture` and
`CACHE_TEST_SOURCE=/absolute/path/to/exact-3e79a7c1-train.py`.
The run command extracts its pinned original93ae reference from Git and
runs base/default/OFF/ON sequentially. Source bytes are checked; no full
model is allocated because tiny argv uses width128 component flags.
Do not rerun concurrently with another laptop tensor job. Raw eight receipts
are committed; rerun into an isolated review worktree, not over these records.

The support driver reuses workflow/tools/preflight_dryrun.py. It adds
observations of all gradients/state/cache connectivity and checkpoint bytes;
it does not replace any update. All tensor jobs ran with normal Python.
The static/receipt assertion suite passed both normal and optimized Python;
this is not a claim of separately repeating all eight tensor jobs under -O.

## Exact-full-size CPU G2: Operator after independent source review

Use the existing CPU route on an idle >=64GiB machine, sequentially, no
Neuron imports/execution. For each source/profile run the exact base,
candidate default and candidate ON. Use real prepare.py, not --stub-prepare
or test_support. Existing production source defaults and all native-profile
argv must match the corresponding proposal; append no initial-cache flag
to the original base. The G2 driver itself appends CPU/no-compile overrides.

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python g2_observe.py --train-py EXACT_SOURCE --prepare-dir REAL_PREPARE_DIR --steps 3 --nproc 4 --micro-rows 2 --micro-tokens 256 --timeout 3600 --out UNIQUE_RECEIPT -- EXACT_PROFILE_ARGV
```

Keep the observer beside its repository-relative support dependencies, or
use the normal existing G2 driver's audited staging arrangement; do not
copy this single file to a random directory and change its imports. The
standard G2 `--compare` checks losses/final state/ownership; additionally
compare each rank's `cache_observations.steps` and rank0 checkpoint payload
and file SHA exactly, just as the committed tiny tests do. Initial ON caches
must have verified connectivity and unchanged Parameter.grad; every gradient
key/missing status/update must match. Full-size expected model is6x1024
RecursiveON or8x1024 standalone; synthetic2x256 still is NOT a production
131072-token hardware update. Native/compiler/NKI behavior remains pending.

## Native interpretation

Use the proposal's full1800 command, exact source hash and existing environment;
do not execute it from this packet automatically. Source gates, host choice,
control/result binding and conditional architecture retry are Supervisor tasks.
Preserve exact argv/env, raw log, all-rank graph_stats, actual startup/charged
through-exit, clean exits and checkpoint. Fresh public20M/context1024 remains
the quality evaluator. No extra tiny native warmup outside accounting.

Ask whether model forward/backward specializations reduce from two to one,
not whether total NEFF count becomes one. Report every graph's timestamp,
nodes and actual available compilation times. Compile_us=0 is unknown, not
a hit. Without comparable startup/cache provenance, no causal startup-speedup
claim is justified. A true-cold full-budget gate is still required before
submission. No native first50-update identity promise: actual elapsed-based
LR may change when startup gets faster, with its formula unchanged.
