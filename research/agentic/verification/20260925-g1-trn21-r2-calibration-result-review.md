# Independent result review: trn21 G1 r2 public-20M calibration

Reviewed 2026-09-25 00:03 UTC. Operator evidence commit:
`f5180bab` on `codex/20260925-g1-trn21-r2-evidence`.
The archived Operator receipt is
`research/v2/g1-calibrations/fs-20260925-g1-public20m-calibration-trn21-r2/operator-receipt.json`,
SHA-256 `a0896536999087e19f1f2e781b79763b9c6ec9adb9b971d452a1d16e095c2c12`.

**Verdict: PASS for trn21's exact fresh-process public-20M evaluator calibration
only.** The retained checkpoint reproduced its known score. This is not a new
model result, an advancement control, a scored-training authorization, a
submission-readiness finding, or permission for cleanup, AWS action, or a
breaker reset. I did not implement or dispatch this run, edit the model or
evaluator, read validation contents, or mutate the host. The Supervisor may
record trn21 G1 PASS subject to the remaining independent A0 gates.

## Authority and one-shot provenance

- The separate Supervisor authorization file has SHA-256
  `af6cd24d0b3443ea7ed10801b7eab6e977d94dbeb3451ff5f629495c46c26ce0`.
  Direct raw diff against pending SHA
  `921498a53b010006649a4570f10513d43988a1f0ea770031f1c739158519ff91`
  changes **only** `status`, `launch_authorized`, and `authorization`.
  The staged request, stage receipt, and independent stage review still match
  their bound SHA-256 values `aba839eb...`, `ca417a41...`, and `bf3c78a7...`;
  the independent corrected guard-package review is `d230c312...`.
- Archived `dispatch.json` SHA-256 is
  `95d0bcf78f45596e7f5a0eaaf92a9fcf1eb268c82087fe5b061ead00c796f761`.
  Its embedded request is the authorized request plus the exact request-file
  hash. The remote `.job.json` and `.launch.json` are byte-identical at SHA-256
  `55100fa554a1f5ded237e9d0367234352861782a8af270b6f111841553a8a7cd`.
  After removing only launcher-added runtime metadata and request-file hash,
  their JSON equals the authorized request. This binds the actual one-label
  `timeout 1800s` guarded command, nested trained source, retained checkpoint,
  1,024 context, 20,971,520-token eval, 1,200-second eval timeout, historical
  compiler flags, and six warm-cache paths. The fresh Operator preflight
  recorded the r2 label absent before the single dispatch; no relaunch is
  recorded. The dispatch returned 0.
- The proven launcher preflight passed at root free space 58,684,768,256 B,
  far above the unchanged 9,126,805,504-B floor-plus-reserve gate. The
  staged checkout was at trained commit `e7f11c38dc3b5e68b373322e10cfef580f1e8d36`
  with clean required tracked paths. Its nested `train.py` SHA-256 was
  `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`
  and trusted `prepare.py` was
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  The retained 469,831,009-B `final.pt` remained at SHA-256
  `5c42f143837fc6d6a64dd1101f53887bdf40d984e752ad8a9ed99cc026a758b9`.
  The six cache directories are path/presence-checked, not content-SHA-bound.

## Monitor, evaluation, and postflight

- All **seven** archived remote artifacts independently re-hash to the seven
  SHA-256 values in the Operator receipt. The monitor event file SHA-256 is
  `1c40e539b0bb3d0a82b30a8432ec8d316dccebef52208297d073a627d03f89f8`.
  Its first event binds the exact remote job SHA. It contains one
  `monitor_ready`, one evaluator start (PID 2835282), 74 logged disk samples
  spanning samples 1 through 730, and one `monitor_finished`: 733 actual
  one-second samples, minimum free space **58,668,208,128 B**, evaluator
  return code 0, no stop/error event or stopped reason. The minimum exceeds
  the 9,126,805,504-B stop threshold by 49,541,402,624 B.
- The raw fresh-process `prepare.py eval-public` JSON reports finite
  `total_nats=57,379,240`, `num_tokens=20,971,520`, `num_bytes=83,462,186`,
  `eval_seconds=718.4807509109378`, `eval_over_timeout=false`, and
  `val_bpb=0.9918353324459732`. Independent arithmetic
  `57,379,240 / (83,462,186 × ln 2)` gives exactly
  `0.9918353324459732`, equal to the retained checkpoint's historical
  reference and inside the 0.0002 absolute tolerance. This is a measurement
  reproducibility result, not a quality improvement.
- The guard/postflight returned 0 with no error, interrupted signal, cleanup
  action, survivor, or deleted artifact. Two stable quiescence checks at
  23:54:17.949801 and 23:54:31.618732 UTC were 13.669 seconds apart; the
  Operator recorded all three owned PIDs absent, zero matching jobs and
  Neuron processes, silent device/lock/checkpoint `fuser`, and stable exact
  source/checkpoint and seven artifact hashes on both checks. The kernel log
  records a TPB reset request at device attach at 23:41:19, completed at
  23:41:29, followed by a normal 23:53:29 device release and no later kernel
  error. As in the prior trn22 G1 review, this is a run-specific attach
  observation, not a general hardware-health claim.
- An additional **read-only** SSH audit after retrieval used the pinned trn21
  endpoint, existing key by path, BatchMode, IdentitiesOnly, and strict host
  key checking. It independently confirmed hostname `ip-172-31-34-85`, exact
  staged commit, no required tracked-path diff, exact nested source,
  evaluator, monitor and checkpoint SHA-256 values, root free space
  58,675,847,168 B, silent `/dev/neuron0` `fuser`, and an empty
  `neuron-ls` process list. No key contents or secret values were accessed.

Do not append this calibration to `research/v2/ledger.jsonl`, call it a new
candidate win, or automatically dispatch A0 from this receipt. A0 still needs
its own exact-environment G2, native/source/safety gates, fresh host preflight,
and separate one-label authority.
