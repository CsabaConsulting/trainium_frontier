# Warm S0 native validation — pending, not submission-ready

This package specifies one full-budget **warm-cache** S0 run on trn22, using the immutable S0
`train.py`. It is not a cold submission replay, a quality-arm win, launch authorization, or upload
approval. The request remains `dispatch_authorized=false`; the unchanged generic launcher does not
enforce that field or the enhanced free-space projection by itself. A separate Supervisor/Verifier
decision and a fresh Operator preflight must enforce both before any dispatch.

## Immutable source and bare training command

- Host: `trn22` / `ip-172-31-32-253`; unique label `fs-20260925-s0-warm-native-trn22-r1`.
- Science checkout is a unique path named in `request-template.json`; root `train.py` is an exact byte
  copy of `submissions/candidates/S0/train.py`, SHA256
  `ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb`.
- `prepare.py` remains `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  The legacy generic launcher/guard/host/sync/backup scripts are exact unedited copies from the
  separately used A0 core; their hashes are pinned in the request.
- Training command after the bounded outer wrapper is exactly
  `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py` with **no flags
  after `train.py`**. S0's source defaults are seed 42 and 1,800 seconds. Compiler flags remain
  `--optlevel=1 --auto-cast matmult --auto-cast-type bf16`.
- The S0 `--out-dir` default is `out`, so the candidate checkpoint is `<checkout>/out/final.pt`.
  `NEURON_COMPETITION_R1_STEP_LOG` points to the unique label's step-boundary log. The boundary
  file is **not JSONL** despite its `.steps.jsonl` suffix: it contains only fsynced decimal step
  integers, no loss or timestamp. Training stdout supplies losses and in-file budget summary.

## Cache and disk constraints

The six train-cache paths in the request are the historical trn22 **116188** run's actual
`training-rehearsal.json` paths (commit `68ecfeb1`, source run
`fs-20260924-current-base-control-trn22-r1`). They are legitimate same-host training caches, but
their **current existence, inode identity and size are unverified by this package**. The Operator
must recheck all six by anchored read-only inventory immediately before launch. Missing paths are a
blocker; no fallback to `/tmp/neuron_cache`, `/var/neuron-cache/research`, newly made caches, or a
different host's cache. Bounded compiler misses inside the same existing directories may happen
because S0 source cleanup can change cache keys; this run must not be called cold or guaranteed
cache-hit.

The prior trn22 116188 dense run consumed `1,854,160,896` root bytes between preflight and
postflight. The immutable operational floor is `6,979,321,856` bytes (6.5 GiB). Require at least
`8,833,482,752` free bytes at the **live pre-dispatch check**; the generic launcher only enforces
the 6.5-GiB floor. If free space is lower, do not dispatch. Historical late-September trn22 free
space was below this projection, so this package is **disk-conditional pending the human volume
decision**. No deletion, gate lowering, cache clearing, or volume action is authorized here.

## Required sequence (Operator only after independent authorization)

1. Reauthenticate trn22-only G1 PASS and the exact bare S0 G2 PASS and 116188-vs-S0 `identical:true`
   receipts. A0 or G1 work using the device must be complete, with one job per host and an idle
   shared lock. Check no global STOP/breaker. Independently review this package and bind an
   authorization record to the **exact external request SHA256**; editing the request requires
   re-review.
2. Stage the exact science commit into the unique **absent** remote path. Do not edit or reuse an
   old checkout. Verify clean HEAD/origin, every fingerprint path/hash, hostname, exact S0 bytes,
   `prepare.py`, absent output/log/step files, idle processes and device holders, available memory,
   free inodes, the canonical device lock, all six cache identities, and free root bytes
   `>= 8,833,482,752`. The expected source commit is a separate science-parent commit; the child
   request itself is external to that pinned checkout to avoid self-hash circularity.
3. Start `research/a0_boundary_observer.py` from a **separate** reviewed checkout before the first
   optimizer-step boundary. Bind its `--label`, `--guard-script`, `--process-json`,
   `--boundary-file` and `--postflight-json` to the exact remote root and label. Save stdout and
   exit code to a unique external receipt with no overwrite. Its identity/timing proof is needed
   alongside the guard; it cannot establish training correctness alone.
4. Dispatch this request once through unchanged `research/fresh_start_launch.py`. The guarded
   command has a 3,600-second outer timeout; rc 124, missing five boundaries, no checkpoint, or
   dirty postflight fails. Do not relaunch a failed label; diagnose and use a new one.
5. After quiescence, retrieve and hash complete launch/job/process/log/postflight artifacts,
   boundary log and `<checkout>/out/final.pt`. Compare **training stdout** step-0/1 losses to the
   same-host 116188 anchor `9.0112` and `8.7822` from archived `training.log`. Later logged losses
   may be compared at matching steps, but a sparse stdout log is not a complete 100-step lockstep.
   The boundary file must be the consecutive integer sequence `0..N-1`, and the external timing
   observer must have started before its first line. Require the in-file budget summary **and**
   external boundary-arrival times, not one alone. Treat any discrepancy as a correctness gate,
   not an optimization result.
6. In a separately authorized fresh guarded process, hash/weights-only-load the checkpoint,
   assert finite tensors, run a causal/batch-independence test on the same trained model, and run
   the trusted evaluator with the G1-proven exact argv:
   `/home/ubuntu/workspace/native_venv/bin/python prepare.py eval-public --train-py
   <absolute-S0-train.py> --checkpoint <absolute-S0-final.pt> --device neuron --batch-size 1
   --seq-len 1024 --eval-tokens 20971520 --timeout-seconds 1200`.
   Require exactly 20,971,520 tokens, finite BPB, no evaluator timeout, and an intact checkpoint.
7. Seal all receipts, source hashes, logs, checkpoint hash, fresh20M metrics and any failures in
   the repo; preserve the checkpoint/cache/checkout. Do not enter a KEEP/adoption in the quality
   ledger for this safety run. The historical trn22 `0.992968300659387` is descriptive only: that
   run failed its postflight disk gate and is **not** an advancement control.

Still required after this warm run: independent Neuron 100-step equivalence if sparse historical
stdout is insufficient, S0 native evaluator-loader equivalence, **two cold** bare-command replays
with startup and public20M checks, and the independent reviewer audit. Only the human submits.
