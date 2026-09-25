# Scoped guarded launches — Supervisor release, 26 September 01:06 IST

The latest single-launcher rule and 01:24 IST tooling timebox are in force.
The Operator alone may stage and launch each exact packet below ONCE after
fresh preflight. This is not permission to re-arm any old queue, change science
or submit. Both old queue processes were absent in 19:34:26 UTC evidence.

## Guard review

Exact operator repair: `65f848af`, canonical `cdf17ee7`.
Supervisor read the complete shared guard, dispatch wrapper and both driver
changes, ran the 15 tests in normal/optimized Python, and independently checked
23 source/argv/environment/hash/lock/descriptor bindings. All passed.
The separate Verifier independently reproduced the original driver-death
lock-release bug, then tested each repaired driver's actual call expression:
contender refused before and after driver death while the immediate child
lives; allowed only after child exit. Its 13 own tests passed in both modes.
The original HOLD remains archived, not erased.

Both drivers authenticate the same inherited exclusive host lock, check fresh
ps AND neuron-ls under it, and pass its descriptor to the immediate job child.
Dispatch retains an outer-command exit receipt even if lock acquisition fails
before driver entry. No retry, alternate lock or release/check gap exists.
This does not prove inheritance into every deeper grandchild of the unchanged
CT harness. Any crash requires explicit worker/compiler/device quiescence
before reuse; every next launch independently refuses a non-idle host.

Shared pins:

- Lock on each host: `/home/ubuntu/.cache/trainium_freshstart/device0.lock`.
- Guard: `adafb54a98527570a76af19ec2785e68496bf1027c51c4127664435c8c963737`.
- Detached receipt wrapper: `c71474bc920f1bdba8f577d072497c4b94ca2c0318c29d39a37555da57ea041a`.
- Source/CPU reviewer acceptance: `20260926-fresh-review-supervisor-acceptance.md`.

## trn21 — P diagnostic first

- Label `fs-20260926-profile-recursive6ef7-trn21-r1`.
- Directory `research/v2/operator/20260926-trn21-profile-recursive6ef7/`.
- Proposal SHA `5d50b247b1f38fb0e1ecd99b3689dddfe8a1ada888029f5caa76b5436d5b3d6f`.
- Binding SHA `d915209e958cf1bc02477d8c02b5ebe23d119758fcb2df11d651acf876c5b832`.
- Driver SHA `cd7a0f2835c6f8e7d41f7b77ea71c0ee035b0d6168949e975e98f69ba52ff2f3`.
- Exact6ef7 source and eaad6aff wrapper; unchanged prepare/profiling API pins.
- Exact bound science/environment, full chip LNC2/four ranks, seed42, budget1800,
  22 steps, capture zero-based step20, rank0 device capture. Warm cache is
  intended, not assumed a cache hit. Timeout2640+60grace; no eval/next job.
- Collect all rank receipts/traces/errors. This is diagnostic only: no BPB,
  fair throughput, update-identity, all-rank dtype or MFU claim from exit0.

## trn22 — cold architecture startup only

- Label `fs-20260926-ct-arch3e79-cold-trn22-r1`.
- Directory `research/v2/operator/20260926-ct-architecture-startup-only/`.
- Proposal SHA `6507dea15fe091eb7c87922049327a79927e21c49a8d6ae053f4be76e8b07906`.
- Binding SHA `ffc82b6a96779b6e76f4fa360e4276f76f79fa4de217445075a8c439a31514f7`.
- Driver SHA `03ba9c6b200a0c5ff34b5ac3a32fc90940d3f24fc2776671f304a8e057904d86`.
- Exact3e79 source; unchanged rehearsal7a55cb87 and prepare5807c46e pins.
- Exact bound five-step launch; full chip LNC2/four ranks, seed42, budget1800.
  Brand-new empty six-cache sandbox, compiler URL equals its new directory.
  Timeout1800+60grace. No public eval and no automatic full run.
- Retrieve five sequential boundaries, uncapped fifth-boundary startup,
  capped allowance, charged seconds, step timings and all exits. Full run
  remains separately gated by observed cold startup <=900 seconds. Three
  post-compilation step samples are not a robust throughput benchmark.

## Both hosts

Retain owned queue pauses. Confirm old queue PIDs truly exited, no other
torchrun/training/eval/compiler/launcher or device holder, source/driver/helper
pins, no STOP/breaker, exact packet root freshness, and proposal disk headroom.
Do not lower gates, reset breakers or overwrite staging. The guarded dispatch
must itself acquire the lock and rerun idle checks immediately before spawn.
If anything fails, archive exact command/error/receipt, do not launch, and
move on under the user's bounded recovery/timebox rule. Never release the lock
immediately after spawning. Preserve current checkpoints/logs and report actual
launch UTC plus each diagnostic outcome. No AWS action or submission authority.
