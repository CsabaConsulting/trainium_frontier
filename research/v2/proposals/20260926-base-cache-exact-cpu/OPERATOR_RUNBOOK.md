# Full-parameter CPU qualification only — disabled pending independent review

Supervisor selected frozen **bcbd148d / b635c8b0** for qualification, not source
adoption, native launch or a full run. This packet changes no train.py. Four
fresh CPU arms run serially: exact6ef7 reference, candidate default, explicit
OFF, ON. All use RecursiveON, seed42. No tensors ran during preparation.

The proposed binding is trn22 solely to reuse the previously reviewed hostname,
prepare location and full-size CPU pattern. **No host is reserved.** Do not
interrupt architecture training/eval, a fallback, or any Operator assignment.
Different host/path bindings require an updated packet/hash review, not editing
the frozen request in place. Do not use these commands on the laptop.

## Pinned source and reused support

`CPU_PLAN.disabled.json` binds full source commits/hashes, all four commands,
their canonical JSON hashes, request/driver hashes and exact source locations.
Read its complete JSON before staging. No branch HEAD inference is allowed.

| Artifact | SHA256 |
|---|---|
| Reference6ef7 from a4aa4351 | `6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3` |
| Frozen candidate from b635c8b0 | `bcbd148dd51b5a014289cbecf3f0e684e35cf537e9588c878e15b265fcbfd522` |
| Unchanged standard G2 | `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8` |
| Real unchanged prepare.py | `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb` |
| Current unchanged idle guard | `adafb54a98527570a76af19ec2785e68496bf1027c51c4127664435c8c963737` |
| Current unchanged dispatch | `c71474bc920f1bdba8f577d072497c4b94ca2c0318c29d39a37555da57ea041a` |

The architecture CPU pattern was independently reviewed in
`research/v2/verification/20260926-architecture-cache-cpu-packet-review/`.
Its old outer lock alone is insufficient now. The new per-arm driver retains
its CPU environment, resource/STOP/source checks and command behavior, adding
the **existing** guard before execution and passing its authenticated lock FD
to the immediate G2 child. The second legacy process scan excludes only the
authenticated current driver's PID (necessary because it is now named
`host_driver.py`). The only post-run addition streams the checkpoint file's
SHA256; it does not load tensors or instrument gradients. All changes are
listed in the plan and require independent review; old acceptance is not
silently inherited. No new launcher, queue or approval system is introduced.

## Scope and resources

All62 parameter tensors /117,453,580 parameters, six layers, width1024,
vocabulary8192, nominal context1024/pack32 remain. Actual synthetic forwards
are2x256 per rank, four ranks, three updates:2048 physical tokens/update, not
the production131072. Source grad accumulation stays1. The native venv's
existing CPU Torch runtime is used; no package install or test_support path.

Unchanged G2 shims: BF16 CPU compute, Neuron CE installation skipped,
CPU/no-compile/three-step/out-dir overrides, synthetic tokenizer and rank-local
synthetic data, input slicing, deterministic clock advancing1ms/call. Real
prepare is imported; **no --stub-prepare or --real-data**. Existing custom MLP
backward and ownership updates execute on CPU. This is not Neuron execution,
native correctness, actual-data training, timing or official-budget proof.

Retain the prior64GiB MemAvailable /6.5GiB disk floors and one OMP/MKL/OpenBLAS
thread per worker. They are reviewed resource floors, not measured peakRSS.
Use new private **disk-backed** TMPDIRs; each retained checkpoint is about470MB.
The laptop already failed the full-size RAM bound and is prohibited. Historical
full6ef7 CPU runs took about20s each on a much larger host, not a runtime promise.
The unchanged G2 timeout is3600s. No cleanup, retry or smaller-model substitution.

## Sole Operator recipe after exact assignment

1. Independent source/packet review and a distinct Supervisor CPU-only
   assignment must precede staging. Confirm current STOP/breaker/role state,
   no competing work, and absent proposed root
   `/home/ubuntu/trainium_base_initial_cache_bcbd_cpu_20260926_trn22_r1`.
2. Stage only the exact Git blobs to the plan's `reference/train.py`,
   `candidate/train.py` and `preflight_dryrun.py` paths. Verify the real prepare
   hash in its declared existing directory. Create each request's absent
   arm root and empty disk-backed `tmp`; place that arm's exact request and
   `host_driver.py`, plus unchanged pinned `idle_guard.py` and
   `dispatch_driver.py`. Verify all hashes. No Downloads/submission changes.
3. For **one arm at a time**, use its exact `detached_dispatch_command` in
   the plan through the existing detached manual route. Dispatch makes exactly
   the recorded `/usr/bin/flock -n .../device0.lock /usr/bin/python3 -I -B
   <arm>/host_driver.py` call. The driver authenticates the inherited exclusive
   lock and fresh `ps` **and** `neuron-ls` evidence before its CPU job. Any
   failed/ambiguous/busy check refuses and logs. No direct driver bypass.
4. Wait for terminal/outer receipts and all job-owned children to exit; retain
   the shared lock through the synchronous job. Confirm child/torchrun exit0,
   no timeout, G2 PASS, all4 rank reports and expected dimensions/source/argv.
   After reference, compare each candidate with it using the unchanged exact
   G2 `--compare` command in the plan, before moving to the next arm. Require
   `identical:true`, all12 losses,62 final parameter digests and ownership plan.
5. Retain raw G2/rank/torchrun/driver logs, requests, actual launch environment,
   pre/postflight, under-lock/outer/terminal receipts, comparisons and
   `checkpoint-identity.json`. Retain checkpoint files on host with URI/SHA;
   no checkpoint download/commit. All four checkpoint SHA256 values should
   match; strict finite deterministic reload and `[1,256,8192]` output shape
   must pass in each G2 report. A mismatch is a failed identity check requiring
   diagnosis, not permission to relax the criterion.

The unchanged G2 inner torchrun boundary does not propagate the lock FD to
arbitrary deeper children and its timeout is not a process-tree cleanup proof.
On any failure, stop the sequence, retain evidence, and require explicit
descendant/device quiescence before any later assignment. Do not launch the
next arm merely because the outer process exited. No automatic retry exists.

## Evidence limits and cold follow-up

Standard G2 final parameter hashes are16 hexadecimal characters, not full
tensor byte dumps. It records all12 losses,62 final rank0 parameter digests,
cross-rank agreement,62/62 changed-parameter counts, ownership plan and
checkpoint reload/determinism/shape. The new read-only driver records a full
checkpoint file SHA256. The standard harness does **not** record intermediate
gradient tensors, per-step parameter/optimizer digests or checkpoint-logit
digests. Do not label these unavailable outputs as checked. Historical tiny
all-gradient/optimizer/cache-connectivity evidence stays separately scoped at
`research/v2/lanes/initial-cache-refresh/`; no intrusive observer is added here.

Independent numeric acceptance is still required after retrieval. Place
receipts under `results/<arm>/` and use the packet's stdlib result-audit mode,
in addition to the unchanged G2 comparator. No PASS exists in this packet yet.

`COLD_PROPOSAL.disabled.json` is a **separate five-step startup-only proposal**,
target<=600s, seed42/full chip/MAX1800 and real accounting. It is not the old
full-run proposal and is not launch-authorized. It needs a separately reviewed
current guard/cold-cache/driver binding after CPU acceptance; no automatic
full run/eval follows. Faster startup can alter elapsed LR/WD, so CPU fixed-clock
identity is not a native first50-step identity promise. No performance claim.

Night-mode decision,26September01:49IST /25September20:19UTC: prepare this reversible qualification
packet only, because Supervisor selected the existing source. Undo: decline
or rebind the packet before execution; no production/host state was changed.
