# Independent HOLD addendum — r10 source-stage controller root

Reviewed 2026-09-25 by an independent Verifier. **Do not enable or dispatch
either r2 e65→r10 source-stage operator.** This addendum supersedes any
operational inference from the disabled-source static PASS receipts below;
their findings about the other exact bytes remain source-only findings. No
host/SSH/EC2 contact, marker, transfer, checkout, cleanup, training,
evaluation, breaker reset, AWS action or submission occurred in this review.

| Subject | Exact prior disabled-source static PASS | Operator SHA-256 |
| --- | --- | --- |
| trn21 r2 `3e84658933cf7f1c9758df894dd57fb265aba366`, `research/v2/authorizations/trn21-c5c-a0-r10-incremental-stage-r2-hold/operator_once.py` | `research/v2/verification/20260925-trn21-r10-incremental-stage-r2-independent-static-PASS.md` at `5e0e0434c5d6fd8146b7d32d987a93d2cab27ff8`, receipt SHA-256 `da237898187a471b616763af6eeb3c0227f772e701001572971ce7bb9436f474` | `cdfdd5c731a73dd190268108a6234579f7547379006d51f1bcac5ef160cdacf7` |
| trn22 r2 `020bfa7ebd048f947d5207cbedaf028a29a4b159`, `research/v2/authorizations/trn22-c5c-a0-r10-incremental-stage-r1-hold/operator_once.py` | `research/v2/verification/20260925-trn22-r10-incremental-stage-r2-independent-static-PASS.md` at `aff8762e1ee310527ecfe2068ee03de869bc4325`, receipt SHA-256 `97ffe8d25ae2be54fa85b8794641343babfbd008d162fbd14931c225872baa83` | `7c04b0288de04a514b3da0a6a9c01f525551f2cfc92f69d4ba63f8941aa2d0b5` |

Both exact operators hardcode
`CONTROL_ROOT = Path("/home/poojith-devan/Documents/ChatGPT/Trainium")`
(trn21 line 28, trn22 line 32), then load
`CONTROL_ROOT / "research/v2/state.json"` for the local breaker/launch
gate (trn21 line 107, trn22 line 116). The active controller instead updates
`/home/poojith-devan/Desktop/Trainium/research/v2/state.json`. Read-only
inspection found distinct regular files/inodes and these exact tracked
contents:

| State file | Last modifying commit | SHA-256 |
| --- | --- | --- |
| Documents/ChatGPT/Trainium/research/v2/state.json | `f8604f4b007ceac80d95e037d18395b0d56ff2f5` | `9e859fdf29b684dff4040fe27f25cef28a40203bdc321e65475ab6b69a38716b` |
| Desktop/Trainium/research/v2/state.json | `7f1206bd454416eb2d8de5a8cd212a7a49bd691e` | `07543c62ce760da02e711be225119563f7fa59cd4ed111d3756fab69052b20e8` |

The currently equal breaker values do not establish a synchronization
invariant. A subsequent active Desktop breaker/launch-state change could be
missed by either r2 operator; a package-root STOP check, even if it reaches
Desktop, does not repair the stale `state.json` read. This is a final
enablement blocker, independent of the prior source-stage race analysis.

Minimum correction: bind each operator's breaker/launch read to the active
Desktop controller (or prove an equally authoritative synchronized state),
check relevant local STOP locations, and independently test the adversarial
case where Documents still says safe while Desktop says tripped. Re-review
each corrected exact disabled package before any enabled request or host
contact. Rebind any downstream exact-hash timing package to the corrected
stage bytes and review it again. Fresh live preflight, separate Supervisor
assignment and all ordinary stage gates still apply; this HOLD grants no
dispatch authority or BPB claim.
