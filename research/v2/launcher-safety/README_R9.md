# r9 exact-A0 launcher-source repair — DISABLED / HOLD

This is a distinct Implementer-only source successor directly atop r8
`d3f39dd8cd8bfa16e25306d4d06293e188fd2cff`. The independent r8
HOLD (`research/v2/launcher-safety/INDEPENDENT_SOURCE_VERIFIER_HOLD_R8.md`)
was published on its separate evidence branch at
`b197f86cdea3b9123b453e277e46a5b849eddc9c`: its receipt SHA-256 is
`13d739de53ec60eb7c783785653697344f7d9d17542d5d261bd16df628efaa7d`,
and its adversarial test SHA-256 is
`4b1cc6165963304f66d9d3f49077be79b5079b351c36445e1dfe643bc321c7b4`.
The r8 source accepted a request-controlled fake `timeout` via `PATH` and a
shell-wrapped `PYTHONOPTIMIZE` child. Its source and HOLD remain historical;
this r9 diff requires a new, different independent source review. No SSH,
AWS, host, transfer, checkout, guard, monitor, training, evaluation,
assignment, or submission action was performed here.

## Exact local review surface

The only production-source edit is `research/fresh_start_launch.py`. These
hashes are SHA-256 of actual r9 file bytes; this runbook has no self-hash.

| File | SHA-256 |
| --- | --- |
| `research/fresh_start_launch.py` | `e6d6d4bcc391d575f6c82bc4b0643b3a04c99b783bc0cc532bc92e65e0b12b9c` |
| `research/test_fresh_start_launch_nonassert.py` | `8553a462d307bca3b42783bbac7c81d752878bf935dd50808d3f6e98289715d5` |
| `research/test_fresh_start_launch_bounded_r8.py` | `3fc45944bba8960096626a3f569a5348cad80e8dc6711ab30b05ae221c740128` |
| `research/test_fresh_start_launch_exact_r9.py` | `efa6d43c1ddcb418934215aad561e73cb50a6a459f017c037723c9c1e7a657e2` |
| `research/v2/disk-monitor/test_r4_early_refusal_implementation.py` | `545c7d3d0d310fe27dc5e5c8c700b9b1d6728feb343ce557999680323ccdf6a8` |

The scored model `train.py` is unchanged at
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`;
`prepare.py`, including evaluator/scoring, is unchanged at
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
No tokenizer, data, optimizer, guard, host-preflight, interlock, or monitor
production source is changed. The parent-to-candidate Git diff is the review
surface, not a claim of independent approval.

## Narrow A0 command boundary

The local launcher and the independently embedded remote script now accept
only the exact trn22 dense c5c A0 training argv profile. The command begins
with `/usr/bin/timeout --foreground --signal=TERM --kill-after=60s 3600s`,
then absolute `/usr/bin/env -i`, the exact reviewed assignment order and
values, absolute native-venv `torchrun`, LNC2/four ranks, and the fixed A0
`train.py` flags including `--matrix-lr 0.015`. The output and step-log
assignment paths are derived from—and compared against—the request's label
and checkout. The environment map must equal the reviewed A0 map exactly:
the frozen cache paths, seed 42, 1,800-second training budget, LNC2 and
`--optlevel=1` compiler flags. An extra `PATH`, `PYTHONOPTIMIZE`,
`PYTHONPATH`, or other key is refused. Only the already-reviewed inner
`env -i` PATH assignment is accepted at its exact value. No shell, Python
wrapper, second `env`, alternate executable, timeout option, train flag,
additional assignment, or trn21 host profile is legal. Other future
experiments require separately reviewed argv/environment profiles.

Both entry points call the absolute `/usr/bin/timeout --version` with a
minimal fixed environment and require a GNU-coreutils banner before the
local dispatch receipt/SSH or remote checkout/host lock. A missing,
non-GNU, malformed or hung version check fails closed. This detects the
reported request-PATH fake-timeout bypass but is not a proof against a
privileged actor replacing a system binary between version check and guard
execution. The absolute executable paths and exact argv eliminate the
reported shell-mediated child-optimization bypass. Existing explicit
non-assert optimization, fingerprint, disk, label and paired-monitor gates
remain. The old disabled r3 request remains unchanged and has relative
`timeout`/`env` tokens, so this source refuses it; a later enabled request
must be newly prepared and reviewed with the absolute argv and r9 source
hash. No request is enabled by this branch.

## Offline regression and remaining gates

The local tests use disposable request files and fake SSH, Git, GNU-version,
preflight, interlock and guard contacts; no real A0 process is launched.
They include the r8 fake-PATH and shell-child witnesses, non-GNU absolute
timeout, extra/mutated environment, wrong host, label/checkout mismatch,
and changed training flags. An in-memory adaptation of the frozen disabled
A0 command proves the legal profile reaches only fake contact. The three
launcher suites pass **24/24** in ordinary Python, `-O`, and inherited
`PYTHONOPTIMIZE=1`. The five retained r4 root-free/paired-monitor/cleanup/
early-refusal/interlock suites pass **59/59** in ordinary Python. The old r4
early-refusal fixture expects a post-bootstrap no-spawn receipt and is not
an optimized-Python acceptance test: under `-O`, the newer launcher correctly
refuses before that stage; r9's explicit optimized-mode tests cover this
entry boundary. These are Implementer checks, not self-verification.

Disposition remains **DISPATCH HOLD**. A separate Verifier must review these
exact bytes and issue an independent source verdict. If that passes, new
exact-source capsule/manifest, transfer-only request/review/assignment,
clean host checkout, G2 disposition, fresh full host preflight, live paired
monitors/observer and a new native one-shot request/assignment remain
separate gates. The frozen 446b/r8 source and r3/r5/r6 request bindings
cannot be flipped or silently reused. No training result or BPB claim follows
from this source repair.
