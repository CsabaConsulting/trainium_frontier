# Independent mechanical review: Recursive architecture full trn21 packet

Disposition: **PASS_MECHANICAL_DISABLED_FULL_BINDING_ONLY**. This is not source adoption, native evidence, quality acceptance, submission approval, or permission to stage/launch. No host, provider, tensor, training, evaluator, or candidate-source operation was performed.

## Frozen target and reproduction

Reviewed commit `7172b012c0e10e8f429c8b48755c1b03495a63c6`, packet `research/v2/proposals/20260926-architecture-recursive-full-trn21/`.

| Artifact | SHA-256 |
|---|---|
| Source, commit `fe0969fd311d654e143dc6113614416c8970a424` | `75b944f7caf7f16a8392e720c135d7a6436f4c1ef980a6bb0c9ac15bafc9f70f` |
| Disabled binding | `c8503bf7aeb06cc4f59d289eba73def9ebd4a59e46dd804a8a16b74ebc425035` |
| Driver | `0b2888e84670f394847783b93256d3cfa98e9328113072db36aa7215562d6213` |
| Proposal | `b2ad09dda0ebaa2e23198b38427641d9a721ebdc9a18c3f6b741322523a8c3c7` |
| Manifest | `a8a5a02db8bf9a58e1aac24f5c180ca0c4a8256a5ada47b3e3fda0500d826020` |

Own `audit.py` executes **107 checks**, independently of the implementer's test labels. Both ordinary Python and `python3 -O` passed; exact target fragments are compiled with `optimize=0` to preserve the production driver's assertions. Raw results are `RESULT-normal.json` and `RESULT-optimized.json`. Reproduction from this directory: `python3 audit.py` and `python3 -O audit.py`. The script reads immutable Git objects from `/home/poojith-devan/Desktop/Trainium`; it does not import a model or execute a host command.

All ten manifest entries and source/prepare/harness/common-guard/dispatcher pins authenticate. The earlier preparation EOF-copy failure remains in the target manifest; this review did not erase it.

## Exact mechanical equivalence and guards

The entire driver is byte-identical to the reviewed `af53137fb8a8dc547572a1552a6bbf91d090e2b6` trn21 pure-architecture driver after only the documented label/root, source SHA, and descriptive-string substitutions. Training argv differs only by appended `--recursive-schedule`. All other science, evaluation command, cache paths, environment, thresholds, and timeouts match the parent packet. There is no startup-adjusted-schedule flag or new `train.py` patch.

The new label is `fs-20260926-arch75b9-recursive-trn21-r1-full`. Package command and provenance agree on the exact source and four-rank command, seed 42, LNC2, 1024 context, matrix LR 0.015, initial cache refresh, and Recursive ON. The source batch default is not overridden. Training compiler flags remain `--optlevel=1 --auto-cast matmult --auto-cast-type bf16`; evaluation remains `--optlevel=1`.

The sole outer path remains `/usr/bin/flock -n /home/ubuntu/.cache/trainium_freshstart/device0.lock /usr/bin/python3 -I -B .../host_driver.py`. The authenticated live lock descriptor is passed to the immediate timeout child. Both process-list and device-idleness checks occur under this same lock before training and again before evaluation, with the evaluation descriptor required to match. The exact guard rejects direct invocation without the authenticated flock. No optimized production entry is approved; the explicit phase environments contain no `PYTHONOPTIMIZE` or `PYTHONPATH` injection.

Own positive/negative fixtures exercised valid and changed source/prepare/harness pins, STOP and disk-floor refusals; busy-before-train/eval, training failure, missing/false budget evidence, missing checkpoint, changed lock FD, and evaluation failure. Exact guard parser fixtures reject another torchrun, malformed process data, ambiguous device JSON, and an occupied device. Both idle checks still run when the process check is busy. Only local temporary-file lock semantics and mocked control flow were exercised, not a remote preflight.

Existing limitations are unchanged: the deeper rehearsal child can close inherited descriptors, so driver death alone is not proof that every descendant stopped. There is no new automatic process-quiescence postflight in the driver. Operator must preserve/retrieve terminal evidence and establish actual PID/PGID/SID/compiler quiescence before any later work. Old queues must remain paused/exited; this packet creates no automatic next job, retry, second launcher, or breaker reset.

## Budget and evaluation

The exact unchanged rehearsal (`00f2ca7f`, SHA `7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b`) was read, not replaced. Its observer records the uncapped fifth rank-0 boundary and process wall through exit, then requires `wall_seconds - min(fifth_boundary_seconds, 900) <= 1800`. Excess startup above 900 seconds is charged. Four independent boundary fixtures pass/refuse correctly: `(2699.9,1200)` passes at 1799.9; `(2700.1,1200)` fails at 1800.1; `(1900,100)` passes at 1800; `(1900.1,100)` fails at 1800.1 seconds.

Source 1800 seconds is unchanged. Outer train 2730 seconds and evaluation 1800 seconds, each with 60-second kill grace, are safety timeouts, not additional official training budget. The driver refuses evaluation after a failed training exit, absent/failed through-exit budget receipt, or missing checkpoint.

Evaluation is a fresh process using unchanged `prepare.py`, device Neuron, batch 1, context 1024, and exactly 20,971,520 public tokens. It receives the retained checkpoint after the second common-lock idle check. A driver completion remains `COMPLETE_PENDING_INDEPENDENT_REVIEW`; finite metrics, exact token count, checkpoint identity, exits, and actual budget must still be authenticated from the resulting raw artifacts.

The pinned separate cold-five receipt is for this exact source: five boundaries, exit 0, six empty SDK caches, fifth `831.4215557309799` seconds and wall `835.84370373399` seconds. This review only authenticates its binding and arithmetic; it does not replace the existing independent cold audit. The proposed full run reuses the parent's existing research cache roots and is **not a fresh-cache full run**. There is no cache deletion/reset or inferred cache-hit claim.

## Honest comparator provenance

Using immutable ledger/adoption snapshot `3a976019fec9fcf401a5139ba18f67a7e2c0ccac`:

| Role in this packet | Actual historical label and score | Historical classification retained |
|---|---|---|
| Raw parent contrast | `fs-20260926-arch3e79-primed-trn21-r1-full`, **0.9931291951444726** | `architecture_arm`, c5c base, DISCARD; not an adopted control |
| Decision comparator | `fs-20260925-l3-timer-dense-trn21-r1-full`, **0.9900560897366009** | `schedule_arm`, c5c base; actual same-host confirmed 6ef7 result |

The packet does not substitute the later Recursive repeat (0.9899528600562257), another seed, a c5c control, or the trn22 architecture score. It does not relabel either historical row or fabricate a new `same_host_control`. These proposed contrasts do not themselves make a future row eligible for the unchanged selector. No selector decision or ledger mutation was performed here.

## Remaining gates

Binding, proposal, and review all retain stage/launch/adoption/submission authorization as false. Prior source, full-CPU, and cold qualifications are separate evidence; this review grants none of them retroactively. A scoped Supervisor assignment and the sole Operator's fresh under-lock preflight are still required, followed by complete raw train/eval/postflight retrieval and independent result/decision review. No quality result is yet available for this future run.
