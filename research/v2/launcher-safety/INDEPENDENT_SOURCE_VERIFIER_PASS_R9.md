# Independent Verifier PASS: disabled trn22 A0 r9 launcher source

**Verdict: PASS for the disabled operational launcher source repair only.** The
reviewed subject is `e046de16826bfc3b954136e3c60804aea0554ecc`, the sole
child of r8 `d3f39dd8cd8bfa16e25306d4d06293e188fd2cff`, fetched from
`origin/codex/20260925-a0-r8-launcher-contact-repair`. The remote branch
resolved to that exact subject before I created this separate clean Verifier
worktree. The independent r8 HOLD is `b197f86cdea3b9123b453e277e46a5b849eddc9c`
and its receipt SHA-256 is
`13d739de53ec60eb7c783785653697344f7d9d17542d5d261bd16df628efaa7d`.
I did not edit the candidate launcher, contact either EC2 host, run SSH,
authorize dispatch, transfer source, start training, or evaluate a checkpoint.

## Source findings

- The only production-source change from r8 is `research/fresh_start_launch.py`,
  SHA-256 `e6d6d4bcc391d575f6c82bc4b0643b3a04c99b783bc0cc532bc92e65e0b12b9c`.
  Its local and embedded remote validators compare the complete command list
  against the same exact trn22 A0 profile after checking the complete ten-key
  environment map. The command fixes absolute `/usr/bin/timeout`, its finite
  `--kill-after=60s 3600s` options, absolute `/usr/bin/env -i`, ordered
  assignments, absolute native-venv `torchrun`, four ranks, and all fixed
  training flags through `--matrix-lr 0.015`. Output and step-log assignments
  must match the request label and checkout. A request PATH, extra environment
  key, shell/Python wrapper, altered flag, alternate executable or trn21 host
  is rejected before a local dispatch receipt or fake SSH, and before remote
  timeout probing, checkout, host lock, Git, preflight, or guard contact.
- Both paths call `/usr/bin/timeout --version` with a fixed minimal environment
  and a five-second subprocess deadline. A non-GNU banner, nonzero exit,
  stderr, missing binary or hung check refuses before the local receipt/fake
  SSH or remote checkout/lock. This closes the r8 request-controlled PATH
  fake-timeout witness. As the runbook states, a version banner cannot prove
  safety against a privileged actor replacing the absolute system binary
  between check and execution; that is outside this request-controlled source
  gate.
- Explicit `sys.flags.optimize` and inherited `PYTHONOPTIMIZE` checks remain
  before local request handling and before remote JSON/host effects. The
  accepted inner `env -i` argv assigns no `PYTHONOPTIMIZE`; exact argv equality
  excludes the r8 shell-mediated optimized-child witness. These gates use no
  Python `assert` statements.
- The local and embedded remote source fingerprint sets exactly equal the
  frozen disabled A0 request's eleven approved source paths. Missing,
  duplicate or extra private paths remain early refusals. The 6.5-GiB
  operational floor, at-least-10-GiB prelaunch gate, four `require_live_pair`
  checks around preflight and guard spawn, interlock pass write, no-spawn
  handling and guard call remain in the reviewed source. The r8-to-r9 diff
  leaves `research/fresh_start_guard.py`, `research/fresh_start_host.py`,
  `research/fresh_start_monitor_interlock.py`, and both paired disk-monitor
  production files unchanged.

The r9 runbook's five listed file SHA-256 values match the candidate bytes.
`train.py` remains
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`;
`prepare.py`, including evaluation/scoring, remains
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
The frozen r3 request is unchanged, `dispatch_authorized=false`, pins older
source, and still uses relative `timeout`/`env` names that r9 refuses.
`git diff --check` passed for the candidate and this evidence worktree.

## Offline verification record

| Check | Result |
| --- | --- |
| Supplied launcher suites (`test_fresh_start_launch_nonassert`, `test_fresh_start_launch_bounded_r8`, `test_fresh_start_launch_exact_r9`) | 24/24 PASS in normal Python, `-O`, and inherited `PYTHONOPTIMIZE=1` |
| Independent `test_independent_r9_adversarial.py` | 4/4 PASS in each of those three modes; SHA-256 `fada0af323c736635b25303ac4db8157da9f8458e00e4769e25cadccab1e9221` |
| Independent command/environment matrix | 30 mutations, each checked at both local and embedded remote gates under normal and optimized compilation; all refused before the respective contacts/writes |
| Independent GNU-version failures | Five local and three remote malformed/nonzero/stderr/hung/missing cases failed closed before the receipt or checkout |
| Independent optimized-interpreter matrix | `-O` and inherited `PYTHONOPTIMIZE=1`, `0`, or empty refused before timeout probing, receipt, or remote effects |
| Retained root-free, paired-monitor, r4 cleanup, early-refusal and interlock suites | 59/59 PASS in normal Python; expected synthetic HOLD diagnostics only |

The independent tests use disposable local files and fake calls only. The
positive profile probe enables a request **in memory for fake contact only**;
no repository request or authority state was changed.

## Disposition

This PASS clears only the disabled r9 source repair for subsequent independent
binding work. It is not an enabled request, host preflight, native launch,
checkpoint, BPB result, or leaderboard authority. An exact r9 source capsule,
transfer/checkout request and reviews, source-matched G2 disposition, fresh
host checks, live paired monitors/observer, and a separate one-shot native
assignment remain required. Existing r3/r5/r6 bindings remain unusable for
this source.
