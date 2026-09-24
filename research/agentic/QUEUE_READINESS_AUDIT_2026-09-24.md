# Queue-readiness audit — 2026-09-24

Role: Researcher / queue auditor only. This report creates no proposal, request,
authorization, or launch authority. No host was accessed.

## Executive verdict

**No high-value Trainium job is ready to execute now.** One exact `116188cf...`
Trn21 control request remains technically launch-ready after fresh Operator preflight,
but it is a duplicate of two valid same-host exact-byte runs and should not consume a
host slot. Every scientifically useful next device action needs more than mechanical
dispatch: A1 G1 needs a new adjudication/finalization chain and independent review;
the retained-checkpoint calibration/screen branch is dirty and unverified at its latest
core; all other current-base ideas lack an approved, independently verified request.

The standing autonomous authorization permits a deterministic Operator to execute a
complete independently verified proposal. It does not turn stale, consumed, dirty,
pending, or low-value requests into queue entries.

## Evidence baseline

- Current submission/research insurance bytes: `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`.
- Current Track-A / A1 production bytes: `9191e85835d8cad50923f32d44ad90c047a9e2d3d4621ce196b11ba66cd66bae`.
- `prepare.py`: `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- Exact `116188cf...` Trn21 cold runs: evidence commits `7fe770b2185f594cfd66fead05aee2051ce1793b`
  and `28b89bfc6d65be9188f8d8a78accd6187e7627b7`, public20M BPB
  `0.9920722144854677` / 3,635 steps and `0.9918353324459732` / 3,654 steps.
- The exact-byte Trn22 workload at `68ecfeb168ddd095ed14f1908fee7c0284dbc97f`
  is descriptive only: its terminal protocol state is
  `FAIL_POSTFLIGHT_OPERATIONAL_DISK_FLOOR`.
- Exact Track-A `35c9 + MATRIX_LR 0.015` host controls are valid at
  `b0b23cbc2e6d16e8bc8d88be9cb042140494206d` (Trn21, BPB
  `0.9866973734526626`, 3,636 steps) and
  `499f753ca554c5c732c35ab4eab7e96488d5e355` (Trn22, BPB
  `0.9873495157460374`, 3,634 steps). Candidates must use their same-host control.
- A1 G0 five-update evidence is archived at
  `37aa3644e63e20a1427cfa69b44fd6de7e25cd81`. Corrected comparator r4
  `994d302b0036c56aaab63797f0561797bc4cf431` and independent receipt
  `8add2585d30b11ac2c50fb2701957657a1a6f30d` validate that archive. The
  corrected G0 result SHA-256 is
  `7198d39b9c21f1eb7e47f84e52398ca85d239ad564a36e6d524674d83355ab4a`.

## Technically ready, but rejected from the active queue

### Exact `116188cf...` Trn21 control r1

This is the only unconsumed device request found with a complete independent verifier
receipt and immutable request material. It remains conditional on the Operator's fresh
idle-device, host-lock, source, memory, and 6.5-GiB post-staging disk preflight.

- Proposal lineage: `experiment/current-base-dual-control-20260924`, verifier tip
  `f0643224ce27c77632dc7dc4367eb72816b04b42`, audited finalization
  `fb1727673381797b96201e0ef05173e41cbf8c88`, runtime core
  `c025d216c2c07210449651099a982fe5656169fd`.
- Host/label: `trn21` / `fs-20260924-current-base-control-trn21-r1`.
- Candidate: `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`.
- Proposal SHA-256: `12aefdde688514597219fc3902e2d544e01196acd179b08b80ef457298c0808a`.
- Authorization SHA-256: `5772ab97e83bc78b552d8f2cd5786634b819e5fe4067e072ece1e63c66fd6e00`.
- Request SHA-256: `0379b78fd35a675b1737df0be888c8d3eeed77be64307fc0f47e31760ee715bf`.
- Job SHA-256: `697ea53ab7e81cf61e8320c73eae14bb32f8cf2011f1e593f4faf9693e20407e`.
- Request contract SHA-256: `a5ede7049f36330cdb46ef7ad285de9272f95f6d704a02a4a9575d10c6cddcb2`.
- Operator entry point, if it were selected:

  ```text
  python research/fresh_start_launch.py \
    research/submission-validation/20260924-current-base-dual-control/trn21-request.json
  ```

- Child command frozen by the request:

  ```text
  timeout --foreground --signal=TERM --kill-after=60s 4200s \
    /home/ubuntu/workspace/native_venv/bin/python \
    research/submission-validation/20260924-current-base-dual-control/control_job.py \
    --arm trn21 \
    --authorization-sha256 5772ab97e83bc78b552d8f2cd5786634b819e5fe4067e072ece1e63c66fd6e00 \
    --sandbox-root /home/ubuntu/trainium_freshstart_20260924_current_base_control_trn21_r1/out/fs-20260924-current-base-control-trn21-r1 \
    --core-commit c025d216c2c07210449651099a982fe5656169fd \
    --request-contract-sha256 a5ede7049f36330cdb46ef7ad285de9272f95f6d704a02a4a9575d10c6cddcb2
  ```

Queue verdict: **SKIP as a duplicate, not READY-to-spend**. Two valid cold Trn21
full-budget runs of the exact bytes already exist. A third same-host control does not
resolve A1, S0, package review, or a candidate comparison. The paired Trn22 arm is
consumed and cannot be relaunched under its old label; its postflight failure also does
not justify spending Trn21 on an unpaired repeat.

## Jobs needing only mechanical request materialization

**None.** There is no current device experiment for which the only missing action is
substitution of hashes into an otherwise accepted request. In particular, A1 G1 has a
scientific adjudication gap, and the calibration branch has uncommitted runtime-core
repins plus no independent receipt over the latest bytes.

One useful **local-only** activity is close to mechanical and needs no Trainium slot:
stage exact `116188cf...` plus launch-command SHA-256
`b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd`
for the final offline upload-package/reviewer verification described by the S0/current
submission audit. That is verification/package work, not an experiment or authority
to submit.

## Blocked jobs, ordered by scientific value

### 1. A1 G1 native five-update observation arm — highest priority

Scientific value: it is the shortest causal gate that can unblock the selected A1
table path. It proves G0/G1 observer non-perturbation and captures detailed table
gradient/state evidence needed before G2.

Pinned intended identities:

- accepted package/comparator commit `994d302b0036c56aaab63797f0561797bc4cf431`;
- independent G0 comparator receipt `8add2585d30b11ac2c50fb2701957657a1a6f30d`;
- hybrid source SHA-256 `908431d7e1da8a9110e6e6005de5ecbe3dba13d798bf2f37f2443404dd8d7d94`;
- ownership manifest SHA-256 `8119df41c8a9ecbfc3f9e69067c7e6ac162c2d5d8ba0a15635265304dcf865a2`;
- host/label: `trn21` /
  `fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r1`;
- eventual launcher command:

  ```text
  python research/fresh_start_launch.py \
    /tmp/fs-20260924-v2-a1-native5-g1-hybrid-observation-on-trn21-r1.request.json
  ```

Blockers: no committed G1 request, finalizer, authorization, or independent verifier
receipt exists. The r4 comparator has no durable G0/G1-only adjudication mode, so a
separately reviewed adjudicator is required before G2 can be drafted. The G1 drafting
worktree currently contains untracked implementation files; those are work in
progress, not evidence or authority. This is **substantive implementation and
verification**, not mechanical materialization.

### 2. A1 G2 and full A1 training

G2 is locked until authenticated G1 evidence and a clean postflight produce a distinct
request and independent authorization. Full A1 remains locked until the complete
G0/G1/G2 exact comparator and model/checkpoint gates pass. No command is authorized.
Running G2 or a 1,800-second A1 arm early would destroy the designed causal sequence.

### 3. Retained-checkpoint 2M calibrations and LR0.0125 screen

Proposal: `20260924-lr015-trn21-2m-calibration-and-lr0125-screen`; candidate screen
SHA-256 `3b7c826933070c0881b05373b15a5f4d71e49d2d1d32f84f933fa0ab1e9640d8`;
sole host `trn21`. The two evaluator-only profiles bind retained exact-`116188`
checkpoints `503dbfda...` and `5c42f143...`; the screen is a one-factor
`MATRIX_LR 0.015 -> 0.0125` full run.

Blockers: the branch head is `ff167881671976c818b65b5cb79e9a1dc3deb48d`
with seven uncommitted request/job/test repins. No independent verifier receipt covers
those final bytes. The proposal still says pending independent verification. Even
after repair, calibration must complete and be authenticated before the screen.
Queue value is below A1: exact `116188` already has two full public20M measurements,
while LR0.0125 is a speculative scalar sweep and the 2M result cannot authorize
adoption.

### 4. S0 cleanup

S0 remains `PATCH / HOLD`. Only provenance hash
`ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb`
is recorded; canonical source bytes, proposal, independent semantic/reviewer checks,
Neuron lockstep, and two cold full runs do not exist. It is not an idle-fill job. The
fast path is offline final-package verification of `116188cf...`, not importing or
running S0.

### 5. A2, local-conv ordering, and QKV-norm CSE

- A2 is conditional on a valid A1 result and must not be run first.
- Dense local-conv-before-attention ordering is researched but still requires a new
  human-approved proposal, implementation, independent causality/NKI-eager gates, and
  same-host comparison.
- `--qkv-norm-cse` is the only duplicate-cleared flags survivor, but it still needs
  native identity and at least a 1% phase gain before a full run. It has no authorized
  request.

These are proposal-stage hypotheses, not mechanically launchable idle fill.

### 6. Track B2 and old-base queue entries

Track B2 touched-row sparse n-gram optimization was rejected before hardware because
the candidate used unsupported dynamic sparse/duplicate/deferred-BF16 paths. Native
component-probe requests remain deliberately non-dispatchable. Do not treat them as
diagnostic idle fill.

All old dense640 / `9a4caa7b` queue items and consumed c595 cohort requests are
obsolete for current selection. Their controls, labels, and quality thresholds do not
bind `116188`, `35c9`, S0, or A1. Replaying them would spend hardware without answering
a current decision.

## Both-host scheduling conclusion

- **Trn21:** reserve for the eventual independently authorized A1 G1, because exact
  G0 is Trn21-bound and the comparator requires matched records. Do not fill it with a
  third `116188` control.
- **Trn22:** no current high-value request is authorized. Do not manufacture symmetry
  by replaying `116188`, old c595/640 arms, or A1 G1 on the wrong host. Use the idle
  period for local research, package verification, and finalization review.

## Queue state

1. Ready and worth executing: **none**.
2. Technically verified but rejected as duplicate: exact `116188` Trn21 control r1.
3. Mechanical request materialization only: **none**.
4. Next device target after new gates: **A1 G1 on Trn21**.
5. Best immediate non-device work: finalize and independently verify the G1
   adjudication/request chain; in parallel, complete offline `116188` package review.

Any future Operator action still requires a fresh protocol preflight. This audit is
not launch authorization.
