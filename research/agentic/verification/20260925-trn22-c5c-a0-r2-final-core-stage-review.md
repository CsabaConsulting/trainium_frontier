# Independent verifier receipt: trn22 c5c A0 r2 final-core stage

Reviewed 2026-09-24 23:27–23:30 UTC by a Verifier independent of the retry-package
Implementer and the staging Operator. Reviewed Operator commits
`cd46fc7a52748cca73dd6a65814ccfb0d242845c` and
`81ad58667077dad1258237fb85436aa6df86c90c`; the latter is a one-line
metadata clarification. I performed repository/static checks only. I did not
connect to the host, start the observer, launch training/evaluation, reset a
breaker, alter AWS, adopt a model, or submit.

## Verdict

**PASS for the immutable, non-dispatchable staged core; NATIVE DISPATCH HOLD.**
This stage contains the exact proposed science command and archived Operator
preflight snapshot. It is not an executable final request or one-attempt
authorization, and it produces no BPB result.

The committed staged request is
`research/v2/authorizations/fs-20260925-v2-c5c-a0-control-trn22-r2.request.staged.json`,
SHA-256 `8a22424d58afe1826ec6973f8ff9d294235ce8e6c499699d00b58dbd44528f0a`.
The companion Operator snapshot is
`research/v2/authorizations/fs-20260925-v2-c5c-a0-control-trn22-r2.operator-preflight.json`,
SHA-256 `0ad7051c6e6eb543b95271f763eec8a4dbd8e580805b3266332364a71a7f9a06`.
Both parse as JSON; `git diff --check` passed for both Operator commits, which
touch only these two authorization artifacts. The canonical JSON hash of the
nested `launch_template` (`jq -cS`) is
`30a53ec19f56ace0c99123139dbf0d07a942921473589ee91ac6c751c0145d95`.

## Bound checks

- The pinned source commit `e65a3097a5b00f6178188a935bb2acb9a5042c6c`
  exists locally. Its Git blobs independently hash to root `train.py`
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`,
  unchanged `prepare.py`
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`,
  and the six launcher/guard/tool SHA-256 values in the staged request. Those
  values match the archived source-stage receipt
  `research/v2/g2-probes/fs-20260925-v2-c5c-a0-control-trn22-r2/operator-stage-hold.json`
  (SHA-256 `a476312c1b9b24e785b72aac1ef78d165fc65b36fa193ea65530a1d188dc9aba`).
  The later Operator snapshot reports the same clean pinned checkout and
  unchanged checkpoint/cache identities. This is an archived snapshot, not
  independent live host verification at dispatch time.
- The exact Track-A suffix in the nested command equals the retry gate
  contract's `train_argv` and the independently reviewed CPU G2 r3 receipt's
  `argv`. The command is bounded by `timeout 3600s`, uses `env -i`, full-chip
  `NEURON_LOGICAL_NC_CONFIG=2`, four `torchrun` ranks, compiler
  `--optlevel=1 --auto-cast matmult --auto-cast-type bf16`, seed `42`, and
  max-train-seconds `1800`. Every declared environment assignment occurs once
  in the command. The only extra assignments are `HOME`, `PATH`,
  `NKI_VENV_PATH`, `PYTHONUNBUFFERED`, and the unique r2 output, step-log and
  run-ID paths. No `--seed`, `--max-train-seconds`, `--num-steps`, CPU or
  `--no-compile` override appears in the native command.
- All six warm-cache environment paths exactly match the unchanged retry
  contract and its prior identity inventory. The Operator snapshot reports
  the retained G1 checkpoint at 469,831,009 bytes and SHA-256
  `1b1d1a3f0ec6b5580312b760f5fee4f702c3fe97c9ddd01ed811472736aadf1b`,
  all six cache directories with the earlier device/inode identities, five
  default training shards by metadata only, clean Git status, no matched
  train/eval/guard process, no Neuron holder or shared-lock holder, and absent
  r2 output/log/step/observer receipt paths. Latest recorded root-free space
  was 58,452,881,408 bytes, above the 10,737,418,240-byte prelaunch gate;
  `MemAvailable` and free inodes were above their stated gates. Recheck all
  of these immediately before native dispatch.
- The trn22 G1 calibration review authenticates a fresh-process public-20M
  known-checkpoint replay, not a new control. The exact-environment four-rank
  CPU G2 r3 numeric review reports PASS and bit-identical comparison to the
  prior c5c CPU anchor; it is not native training or BPB. The earlier CPU G2
  r2 label failed due to an absent step-log parent and is consumed. The failed
  native A0 r1 label is separately consumed with zero boundaries and no BPB.
  The archived human acknowledgement and independent review support only the
  narrowly re-armed trn22 disk breaker; no launch authority follows from it.
- The external observer script's staged SHA-256
  `a6f437fe8a9d0a14ec3025bc822f1afa12e3f355fca730e4ae0db7ae00263592`
  matches the archived r1 script bytes. Its command binds the new r2 label,
  exact process/step/postflight paths, and exact guard script. It was not
  started in this stage. The wrapper has no top-level `host` or `command`,
  says `dispatch_authorized=false` and
  `final_executable_request_exists=false`, and cannot be passed directly to
  the pinned `fresh_start_launch.py`, which requires those top-level fields.
  The nested template also has `dispatch_authorized=false`.
- The Operator reported a subsequent read-only check finding this clean
  checkout's `research/run-logs` **parent absent** at stage; its raw result
  was not part of the two committed snapshot files and is not an independent
  live finding of this Verifier. I checked the exact pinned
  launcher/tool Git blobs: `fresh_start_launch.py` calls
  `fresh_start_host.preflight(r, label)` before opening its launch manifest,
  job and log files or spawning the guard; `preflight()` first records
  `<label>.preflight.json` through `write_json()`, which executes
  `path.parent.mkdir(parents=True, exist_ok=True)` before opening the file.
  Therefore this known launcher creates the absent parent before the native
  child could append the step log. The earlier CPU G2 bypassed that launcher
  and failed at synthetic step 0 for precisely this missing-parent reason.
  This source-order proof applies only to the pinned launcher; an absent or
  unwritable parent would make the launcher fail **before** child spawn. No
  manual host directory creation was performed or authorized by this review.

## Minimal safe continuation

1. A Supervisor must ratify the still-draft retry proposal as a completed
   one-factor control proposal. An Operator may mechanically materialize a
   **new** executable-but-unauthorized request from the exact reviewed nested
   template. Bind the completed proposal and this stage receipt, preserve
   every science/environment/cache/path/label byte, and commit the new file.
   Do not rename or overwrite the staged wrapper or reuse r1.
2. **A second independent exact-SHA review of that materialized final request
   is required.** Compare its canonical launch core against the hash above,
   verify no extra command/environment drift and a fresh Operator preflight,
   including the exact pinned launcher's ordered parent-creation behavior
   above; do not silently substitute a direct G2-style invocation or an
   unreviewed manual directory creation. The known launcher records source
   hashes but does not itself enforce the request's
   `expected_sha256` map or the 10-GiB gate, so the Operator must enforce both.
3. Only after that PASS may the Supervisor issue a separate, hash-bound,
   one-label/one-attempt authorization. The Operator must repeat STOP, breaker,
   source, cache, checkpoint, host, free-space, inode, memory, process, device,
   lock and unique-path checks **immediately before** launch; start and verify
   the read-only observer first while the step file is absent; then execute
   once with continuous root-space monitoring and authenticated guard stop
   before the immutable 6.5-GiB floor. Any failed check returns to HOLD.

The staged snapshot, CPU G2, and this receipt do not authorize native training,
fresh public evaluation, checkpoint deletion, adoption, or submission.
