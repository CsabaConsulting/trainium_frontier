# trn21 G1 r2 — one-label evaluator guard, pending authorization

**Status: HOLD.** `request.guard-pending.json` has `launch_authorized=false`. This package
does not authorize a host launch, change the staged source, produce a new model score,
or authorize A0. The staged source and retained checkpoint have a separate independent
PASS review at `research/agentic/verification/20260925-g1-trn21-r2-staged-request-review.md`.
This new guard package needs its **own independent Verifier receipt** before the
Supervisor can issue a distinct one-label authorization.

## Exact one-shot design

- The staged Git root remains
  `/home/ubuntu/trainium_freshstart_20260925_g1_public20m_calibration_trn21_r2`
  at trained commit `e7f11c38dc3b5e68b373322e10cfef580f1e8d36`. The trusted
  `prepare.py`, nested trained `train.py`, retained checkpoint and six historical
  warm eval caches are pinned in the JSON request. No source, evaluator, model,
  checkpoint or data is edited or copied.
- The proven `research/fresh_start_launch.py` at SHA-256
  `06eefe82e04a810474614295228e1cef84923373e653a9c62f97f5dca3456b94`
  authenticates the committed runtime closure and source bytes, obtains the host
  lock, and starts its owned-process guard. `dispatch_one_label.py` adapts only
  the already configured SSH transport to the pinned trn21 endpoint with
  strict host-key checking. It refuses the pending request and any authorized
  copy with non-authorization field drift.
- The guard runs `guarded_eval.py` from a separately hash-verified,
  non-overwriting path outside the staged checkout. Its sole child has exactly
  the same `prepare.py eval-public` argv as `request.staged.json` after removing
  that request's outer `timeout` prefix. The same historical environment and
  six cache paths are used; there is no second scorer or alternate data path.
- `guarded_eval.py` samples root free bytes every 1 second while its child is
  alive. The hard floor is 6,979,321,856 bytes (6.5 GiB); it sends TERM to its
  **own verified** evaluator process group as soon as free bytes fall below
  9,126,805,504 (floor plus 2 GiB reserve). The child has a parent-death TERM
  signal, and the proven outer guard owns final descendant cleanup and
  postflight. Samples and stop events are fsynced to the unique-label
  `*.disk-monitor.jsonl`. This reserve is intentionally stricter than the
  historical launcher's 8-GiB gate; neither threshold is lowered.

## Independent review and authorization boundary

1. A distinct Verifier compares the two JSON requests: the original evaluator
   suffix, environment, hashes, cache paths, disk threshold and label must
   match. Review the monitor's PID-birth/UID/marker check, parent-death behavior,
   signal path and fail-closed tests. A static PASS is *not* dispatch authority.
2. The Supervisor alone creates `request.guard-authorized.json` from the
   pending JSON, changing **only** `status` to
   `AUTHORIZED_ONE_LABEL_G1_EVALUATOR`, `launch_authorized` to `true`, and
   `authorization` from null to an object with exactly these keys:
   `decision=AUTHORIZE_ONE_LABEL_G1_EVALUATOR`,
   `pending_package_sha256=<SHA-256 of unchanged request.guard-pending.json>`,
   `stage_review_sha256=bf3c78a7db65ea7829386129969e7bb06cc700f11c2d4b2f251c448f04878a21`,
   `guard_package_verifier={path:<absolute reviewed receipt path>,sha256:<its SHA-256>}`,
   and `supervisor_authorized_at_utc=<UTC ISO-8601 timestamp>`. The adapter
   refuses an authorized copy with *any* other field changed. The new file's
   exact SHA-256 is recorded before dispatch. No environment or command field
   is edited during authorization.
3. A distinct Operator completes a **fresh** live preflight: expected hostname,
   UID/HOME, clean exact staged HEAD and source/runtime hashes, checkpoint type/
   size/SHA/unheld status, all six cache directories, fixed tokenizer-cache
   hashes, allowed environment **names**, no train/eval/G2/Neuron or lock holder,
   closed applicable breakers, no existing r2 label artifact, and root free
   bytes at least 9,126,805,504. A snapshot from staging/review is not enough.
   A failure is HOLD; do not clean, delete, reset a breaker, or substitute a
   source/cache. The source checkout's old untracked logs stay untouched.
4. The Operator installs only the exact `guarded_eval.py` bytes to the request's
   unique external `monitor.script_path`. The target must be absent before
   creation and, after writing, a regular UID-1000 file, mode not group/world
   writable, with SHA-256 equal to `monitor.script_sha256`. Do not put it in the
   clean staged Git checkout, replace an existing target, or edit it on-host.
   Recheck disk/idle/label after installing. The script's path is not a credential.
5. Invoke `dispatch_one_label.py --request <the authorized JSON> \
   --operator-checkout <local checkout containing the exact e7 launcher/sync> \
   --ssh-key-path <existing configured key path>`. The adapter authenticates
   the *path and permission* of the key without reading or printing key bytes.
   It creates an exclusive local dispatch receipt and invokes the proven
   launcher once. If this transport cannot be reproduced with the exact
   module hashes, HOLD rather than patching a live tool or falling back to
   another launcher. Any attempted evaluator dispatch consumes r2.
6. The Operator watches the guard, monitor events and disk while active. A
   monitor stop/error, timeout, missing event, source/checkpoint hash drift,
   unexpected process, or disk-floor crossing is a failed gate and requires
   a new label plus diagnosis. Do not trigger A0 automatically.
7. After the guard exits, require **two** stable quiescent checks at least 10
   seconds apart: guard/timeout/evaluator/descendants absent; `fuser` and
   Neuron process count empty; same hashes for log, guard postflight, monitor
   events, checkpoint and source; no file change between samples. Retrieve
   full compact artifacts with SHA-256 confirmation, parse the actual JSON,
   require rc 0, finite `val_bpb` and `total_nats`, exactly 20,971,520 tokens
   and 83,462,186 bytes, `eval_over_timeout=false`, and recomputed BPB within
   0.0002 of `0.9918353324459732`. The monitor must report no stop and a
   nonzero sample count. Archive Operator evidence and obtain a **separate
   result Verifier** before trn21 G1 can be marked PASS. This is calibration,
   not a new science result or model-ledger row.

No command in this package is to be run on either host by the Implementer.
