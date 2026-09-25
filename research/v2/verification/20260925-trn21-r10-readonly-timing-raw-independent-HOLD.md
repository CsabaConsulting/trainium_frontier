# Independent raw-result review: trn21 r10 read-only timing — HOLD

**Verdict:** `HOLD_NO_OPEN` for timing feasibility. The one assigned, fixed-result
attempt is consumed. This receipt grants no source-stage, native, training,
evaluation, submission, cleanup, breaker-reset, or AWS authority. I made no
host contact and changed no source, request, assignment, or raw evidence.

## Exact evidence binding

- Executed timing package: `codex/20260925-trn21-r10-timing-enabled-r3-r1`
  at `c238fe2478f81799c367279958c3db3e0ac21e4e`; the pre-result package
  was clean at `490bb4e533d41e095cea3a69eb041a6c42ff6085`. The only files
  added by `c238fe24` are the raw result and Operator terminal receipt.
- Raw result:
  `research/v2/authorizations/trn21-r10-stage-readonly-timing-r1-hold/OPERATOR_RAW_TRN21_R10_TIMING_RESULT.json`,
  SHA-256 `9573fae597fcef03e49419cbe6886541d3109b476ad89af26f4887806b51e34f`.
  Operator terminal receipt SHA-256
  `9259457a5b92c74012763aef2dc5990315d2333219ad8f341a36399d22046f74`.
- Exact canonical authority bytes revalidated locally: enabled request
  `5b607110086abe10d03f669b34143329143e4fbd840255208964f6701fc9e173`,
  independent final review
  `938efa5d5f9f768d097399367e0e2aa90bfb19c0a5d543c278b9deaf0fbc0f21`,
  and Supervisor one-attempt assignment
  `f0b8a520e27966d70053f285408d9d6dfb84aa3bef03df35e94d00549884ac6c`.
  Invoker SHA-256 was
  `30d9609e9766f9cb8484c54f9d0d9d49c167ec0b54e1145bff94aa30f1004345`;
  collector SHA-256 was
  `83836cc6f0a6c2697226ba916b4d915e1f9a7b8f4eada8d5e05bd5470311aaf4`.
- Reconstructed, without execution, the pinned SSH endpoint
  `ubuntu@ec2-16-26-246-98.ap-southeast-4.compute.amazonaws.com`, remote
  command SHA-256
  `397f78eba7431a01baeab549649950dd21584d619417a33b0c1135a819a71c81`,
  and canonical stage/timing payload SHA-256
  `ce1eb0393237e41dbc73159cb8c8741c19d982b921f4c46734c509d794621b77`.
  These match the raw result. The recorded host key is pinned ED25519
  `SHA256:G4eS9Bib/yRt+bpWWOUV0SgF/pqTfhLITafrA2ToCfc` with known-hosts
  SHA-256 `e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3`.
- Both local execution-root records are identical: clean
  `/home/poojith-devan/Desktop/Trainium-trn21-r10-stage-r3-active-root` at
  `7205abf5469edc616c4e4b58caee27bc1957bbd4`, `train.py` SHA-256
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`,
  and `prepare.py` SHA-256
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  Both active Desktop controller records have state SHA-256
  `560daf07441133a14854bc108af484fb7a9c1f9f360379b830ca0bc85facb6d0`,
  absent checked local STOPs, untripped global/trn21 breakers, and false
  launch flags.

## Raw outcome and no-open boundary

The result is canonical JSON. Its base64 stdout decodes to exactly 1,353
bytes (SHA-256
`f9bf74ed71410fb5160888afa37433050ed2fa35b1e52646b572f869533ffb1d`)
and exactly three canonical JSONL records, byte-for-byte equal to `parsed`:
`start`, trial 1, `end`. Stderr is empty; remote exit code is 0, timeout is
false, and parse error is null. UTC ordering is local start
`11:10:05.768382`, remote start `11:10:12.850551`, trial
`11:10:14.012074`, remote end `11:10:14.012127`, local end
`11:10:14.258617` on 2026-09-25.

The remote full read-only preflight took `1.4248434960609302` seconds and
reported the exact trn21/e65 source, target, bundle, label, and lock; root
free space was `57,896,984,576` bytes with `31,563,301` free inodes. Trial
1 of 3 took `1.1614667139947414` seconds and returned
`RuntimeError: final label scans too old for first O_EXCL`. The terminal is
`READONLY_TIMING_HOLD_NO_OPEN`, `completed_trials=1`, and `actual_open=false`.
No successful per-trial activity, label, or space ages were emitted; their
margins are **unknown**.

The pinned timing source calls only the stage's read-only lock, `_probe`, and
`_final_no_write_continuity`. The latter throws at its two broad `/home/ubuntu`
label-scan duration check before its second activity check, its freshness
return, any hypothetical clock-only open point, and every bundle/target
writer. Thus this exact path did not call bundle `O_EXCL`, create a target,
run training/evaluation, or write a stage marker. The raw
`no_host_write_or_marker=true` flag is consistent with this source path, not
an independent proof that no third party changed the host.

Separate point-in-time read-only snapshots at
`origin/codex/20260925-dual-host-readonly-1021@4b106484` have raw SHAs
`694102330ebb1a41c9bc5d47589cb517593a50be9962bed2745d00921700dad4`
(before) and `38416da210cf9d3841033085dc98393279ea7bf779b2813108243dd09b74987f`
(after). Their decoded stdout hashes match the embedded hashes. Both report
the same trn21 boot ID, mount namespace, clean e65 checkout and c5c/prepare
hashes, no matched process or Neuron/device/host-lock holder, no checked host
STOP, unchanged `31,563,301` free inodes, and the same safe local controller
state. Root free bytes decreased by `24,576` between observations; the
snapshots do not attribute that change and do **not** explicitly stat this
r10 bundle/target. Therefore they support no observed material drift, not a
byte-for-byte unchanged host or an independent exact-path postflight proof.

## Diagnosis and next decision

The immediate bottleneck is the **two broad label scans**, not the second
activity check: the code rejects when their combined duration exceeds the
existing 1.0-second limit, before it can execute that second activity check.
The failed trial's entire pre-open sequence was 1.16147 seconds, so the scans
alone occupied **more than 1.0 second, over 86%** of that measured interval.
The individual scan times and all final freshness margins were not recorded.

A separately proposed and reviewed *bounded, read-only* retry could
plausibly help if scan time varies or caches warm; this single failure cannot
establish that it would. It must rerun the full activity, both scans, STOP,
space, and original one-second freshness gates for each try, stop within a
fixed trial/time cap, and use a new exact authorization/result path after
fresh host checks. Do not relax the one-second gate, reuse this consumed
label/result, or infer source-stage/native authority from a future timing
pass.
