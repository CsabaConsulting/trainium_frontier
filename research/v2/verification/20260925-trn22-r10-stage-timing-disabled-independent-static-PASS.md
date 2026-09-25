# Independent trn22 r10 read-only timing package review — static PASS, dispatch HOLD

Reviewed 2026-09-25 10:13 UTC by a Verifier separate from the Implementer.
Exact disabled subject: `f91f24d946da8b9305a7df7ffb0b71abc565beba` on
`origin/codex/20260925-trn22-r10-stage-timing-disabled`. This is **PASS for
the disabled, read-only source only**. It is not an enabled timing request,
host feasibility measurement, source-stage authority, native result or BPB
result. No SSH/EC2 contact, marker, output receipt, transfer, checkout,
cleanup, training, evaluation, AWS action or submission occurred in this
review.

## Exact-byte and safety findings

- The diagnostic binds the corrected r10 stage package at
  `020bfa7ebd048f947d5207cbedaf028a29a4b159`, which is an ancestor of
  this subject. In this tree the stage remote, operator and HOLD request
  hash respectively to `da647804d439e738484532a06b3d7dcf122b3fce808d6dce342a3d25eaaab49f`,
  `7c04b0288de04a514b3da0a6a9c01f525551f2cfc92f69d4ba63f8941aa2d0b5`
  and `625b4a53c0f67fba0a7b4ef5a6705f64f3935eb8f5841ef5e2212f6c8e820b6e`.
  The timing remote and collector hash to
  `a9bc253680403883dacc601b35cb1935c81672c512fc9a04cff9ce12e7944ed4`
  and `4d35b139e2d1cd44e96afb25458b10415de94dd211791ed4172e2893f95e21ab`;
  the disabled contract matches those bytes and sets dispatch and every
  remote-write/training permission false.
- The remote code opens the existing host lock and source files read-only,
  uses read-only process/Neuron/Git/label/disk checks and stdout JSONL, and
  has no bundle/target create, payload write, unlink, process signal or
  source-stage writer call. The local collector's one `O_EXCL` write is its
  **local** raw-evidence file after distinct enabled request, independent
  timing review and Supervisor assignment; no such artifacts are checked in.
  Both direct entrypoints exit 1 on the absent enabled request before SSH
  or local output creation. Its SSH command uses pinned host identity,
  `stdin=/dev/null`, isolated Python and only `probe`/`timing` operations.
- The 20-trial clock model follows the r10 remote stage's three activity
  checks, three broad label scans and two byte/inode-floor checks in the
  same order. It records actual helper intervals and age arithmetic, stops
  a trial at the modeled one-second failure boundary, and flushes each
  complete JSONL trial. Timeout/partial output is retained but cannot be
  classified as timing PASS; a distinct post-probe is attempted. The
  would-open is deliberately a monotonic sample: it does **not** measure
  actual `O_EXCL` syscall delay, payload/fsync or same-UID races, and it is
  not a stage preflight.
- The supplied offline suite passed **19/19** under normal Python and
  **19/19** under `python3 -O`; `git diff --check` passed. Synthetic tests
  cover stale/slow boundaries, STOP/activity/space/label arrival, partial
  JSONL and timeout retention, source pin mutation and disabled refusal.

## Separate final-enablement blocker and next action

The timing collector checks the **active**
`/home/poojith-devan/Desktop/Trainium/research/v2/state.json`, but the
currently pinned source-stage operator `020bfa7e` checks the different
`/home/poojith-devan/Documents/ChatGPT/Trainium/research/v2/state.json`.
Read-only local inspection found these are distinct files/commits; at that
observation their SHA-256 values were `07543c62ce760da02e711be225119563f7fa59cd4ed111d3756fab69052b20e8`
and `9e859fdf29b684dff4040fe27f25cef28a40203bdc321e65475ab6b69a38716b`.
The currently equal trn22 breaker bits do not prove future synchronization.
**Do not finally enable the r10 source-stage writer at 020bfa7e** until its
controller-state authority is repaired or proved synchronized in a separately
reviewed exact package. A repair changing that stage operator's bytes/commit
will make this timing package's stage pin stale; rebind and independently
review the timing package before any eventual one-shot timing dispatch.
Even then, a separately reviewed enabled timing request, fresh live checks
and Supervisor assignment are required. This static PASS supplies none of
them.
