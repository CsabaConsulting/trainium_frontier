# Independent static PASS: disabled trn21 c5c A0 r10 native-control draft

Reviewed 2026-09-25T11:55:44Z by a Verifier, offline only. Subject is the
Implementer commit `726eca9fa7cd04ae4b23778ed2a98b01f8140a56` on
`codex/20260925-trn21-a0-r10-native-disabled`; its seven added files are under
`research/v2/authorizations/trn21-c5c-a0-r10-native-disabled/`. The subject
worktree was clean. This is a **STATIC PASS FOR THE DISABLED PACKAGE ONLY**.
It is not a source-stage result, G2 carry-forward, final native-request review,
Supervisor assignment, machine preflight, training or evaluation authorization.
No host was contacted and no marker, staging, training, evaluation, cleanup or
AWS action was performed.

## Exact reviewed bytes and bindings

| Artifact | SHA-256 |
| --- | --- |
| `NATIVE_REQUEST.DISABLED.json` | `f4612c71fd4cd1a429026b7a01d23c95de0652323a7dca34130b25d8a42454ce` |
| `BASE_BINDING.STATIC_ONLY.json` | `eb37d356edfc6e846a342e47f4e067c0d49dc30503199279bf320ae2bd651630` |
| `PAIR_BINDING.STATIC_ONLY.json` | `6613a520b85d219ec48c32be2368b039953d4b20fefa18f334dd273f8cedc766` |
| `PUBLIC20M_EVAL.DISABLED.json` | `444bb99406793ec4050a7d24dc0900adf1f21cc7da5d9d1f51e5e46608a715ca` |
| `GATE_CONTRACT.json` | `b18cd0af7f69e9343534e1e5412fce53c2fce9509a0eec8eac0b81925831a06b` |
| `RUNBOOK.md` | `d1530f40bf0401bf82f10942e70ca4b2ca8e9a680b5459eacb08ab1bdb9c7cca` |
| `test_disabled_package.py` | `33fd2f7373e52ffd799430830bdf70113c3d77116aa2378aba456f739dbbe219` |

The completed proposal at `96fcec98a538673772febed826a653a4478cd9b0`
has SHA-256 `1f616cde93b726553684b13c4c0ac2081e51bc289d90fdaff2d7ac18920b7e9d`;
its independent static review at `dd9891c6d4dedfaa9d5762d400fc379c18670dcf`
has SHA-256 `0b425319baac572fd0ee52e4c2cc9a60b1a2d071122d659707466dc9434a1aba`.
The current proposal-only ratification has SHA-256
`1c5180d49e1122a8fdf992ac8735aefcbb71b92dfa924fffba335e33905b235b`
and `launch_authorized=false`; the old e65 ratification does not cover r10.

The proposed scored source is the frozen Git commit
`22bfbab5a77d6feb670d2d737504f8b70b08ddc1`, not this documentation
checkout: `train.py` SHA-256 `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`,
unchanged `prepare.py` SHA-256
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`,
and r10 `research/fresh_start_launch.py` SHA-256
`1be19bc31d2bdcde8abe9fe90cc1ac884d6efd5f7be197d7bdbfe64c257dcff7`.
I independently recomputed those Git-blob hashes. The other eight frozen
source hashes are checked against Git blobs by both normal and `-O` package
tests. The observer script SHA-256 is
`a6f437fe8a9d0a14ec3025bc822f1afa12e3f355fca730e4ae0db7ae00263592`.
The local documentation checkout's `train.py` is **not** c5c and must not be
used as the execution root.

## Static findings

- The request is `DISABLED_*`, `dispatch_authorized=false`; the eval plan is
  `DISABLED_*`, `evaluation_authorized=false`, with no checkpoint hash; both
  monitor bindings are `STATIC_ONLY`. The contract denies host, native and
  eval actions. Only proposal selection/ratification is marked complete;
  every later authority gate remains false. No one-shot monitor-authorization
  file is in the package. The frozen launcher's request validator rejects
  both the disabled request and a missing dispatch field at its first gate,
  before receipt creation or SSH.
- The request matches the ratified proposal byte-for-byte for its ordered
  43-token child command and ten-key effective environment: trn21 only,
  label `fs-20260925-v2-c5c-a0-control-trn21-r1`, private r10 checkout,
  native `torchrun --standalone --nproc_per_node=4`, LNC=2, seed 42,
  1800-second training budget, width 1024, `--no-eval-public`, and final
  `--matrix-lr 0.015`. The six cache paths are trn21's proposed existing
  train caches, not trn22 or eval caches; live identities remain unproved.
- Request SHA binds the base and pair monitor drafts; base SHA binds the pair.
  The base's `base-unused` journal names are not the paired wrapper's live
  role journals. The three disk thresholds are 10,737,418,240 bytes at
  prelaunch, 9,126,805,504 bytes for paired-monitor stop, and 6,979,321,856
  bytes as the immutable floor. Polling is 0.25 seconds. The boundary
  observer is explicitly required before guard spawn, but its live host
  identity is a future gate.
- The separate fresh-process public-20M/context-1024 eval plan matches the
  trn21 G1 command template and expected 83,462,186 public bytes. It is
  disabled, has no checkpoint identity, and cannot score this control yet.

## Independent checks

- `git diff --check 726eca9f^ 726eca9f`: PASS; subject tree clean.
- Package unittest discovery: **6/6 PASS** under normal Python and **6/6
  PASS** under `python3 -O -B`.
- Extracted the exact frozen local and embedded-remote command validators
  without running their entrypoints. Each accepted the exact request and
  rejected six independent command mutations (removed `--no-eval-public`,
  changed rank/budget/seed/cache, reversed final flags) and three environment
  mutations (LNC, seed and cache). The frozen request validator rejected
  disabled and missing dispatch authority before any external action.

## Remaining HOLD gates for any future run

The current trn21 r10 stage draft is a HOLD, not an accepted source checkout;
the matching trn22 stage and same clean HEAD must also be proven. The local
`/usr/bin/timeout` compatibility problem noted in the runbook is not waived.
Require an independently accepted r10 source stage, G2 carry-forward or a
fresh numeric review, a *new* enabled request/binding hash chain and its
independent review, exact fresh dual-host preflight, disk/cache/observer/lock
and protected-artifact checks, unique unused label, and a separate one-shot
Supervisor assignment. Later training and public-20M evaluation each need
their own terminal/checkpoint/readback reviews. A changed package byte or
execution root requires re-binding; this static PASS must never be treated
as dispatch or evaluation authority.
