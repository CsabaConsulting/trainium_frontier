# Independent Verifier HOLD: trn22 A0 r8 launcher source

**Verdict: HOLD for the disabled operational launcher source gate.** The exact
subject is `d3f39dd8cd8bfa16e25306d4d06293e188fd2cff`, directly atop r7
`e74f99958067a10de1467c8801ad601fadd46684`. This was an offline review
in a separate clean sparse worktree. I made no SSH, AWS, EC2, transfer,
checkout, monitor, training, or evaluation contact and did not edit the
candidate source. Independent tests and this receipt are on a separate
Verifier evidence branch.

## Blocking counterexamples

1. **The validated `timeout` name can resolve to an unbounded executable.**
   Both `bounded_timeout_command()` copies accept
   `command=["timeout","--kill-after=1s","1s","true"]`; both environment
   gates accept a request `PATH` beginning with an isolated directory that
   contains an executable named `timeout`. The local `main()` then wrote a
   synthetic dispatch receipt and reached fake SSH. The embedded remote
   script reached one fake guard spawn. A separate harmless child-process
   probe with the same argv and PATH executed the fake marker
   `FAKE_TIMEOUT_NO_DEADLINE`, proving that the deadline options were never
   interpreted by GNU `timeout`. The launcher merges request environment at
   `research/fresh_start_launch.py:250`; the guard later calls
   `subprocess.Popen(request['command'], env=env)` at
   `research/fresh_start_guard.py:111`. Neither the local command gate at
   `research/fresh_start_launch.py:305` nor the remote copy at line 59 binds
   the executable. Thus a syntactically bounded request may be operationally
   unbounded before the promised source gate refuses it. The temporary fake
   executable ran only on this workstation, with no training or host contact.

2. **A child optimization assignment can be hidden in a shell argument.**
   `command=["timeout","--kill-after=2s","2s","sh","-c",
   "export PYTHONOPTIMIZE=1; python3 -c 'import sys; print(sys.flags.optimize)'"]`
   passes both launchers' argv-prefix check, reaches fake local SSH and fake
   remote guard, and a harmless local execution prints `1`. The r7/r8
   launcher-interpreter checks themselves still work, but the stated child
   optimization exclusion is incomplete. The eventual exact request review
   must not rely on this parser to exclude shell-mediated optimization.

The independently reproducible witnesses are in
`research/v2/launcher-safety/test_independent_r8_adversarial.py`. Passing
tests in that file describe both safe refusals and these accepted unsafe
paths; their green result is **not** a launcher PASS.

## Repaired r7 witnesses and scope checked

- `timeout 0`, `0s`, zero/missing `--kill-after`, negative and oversized
  durations refuse before local receipt/fake SSH and before remote host-lock
  directory, Git, preflight, no-spawn receipt, manifest, job, or fake guard.
- Off-list private path, `research/scratch`, missing, extra, duplicate,
  traversal and absolute fingerprint lists refuse at the same early gates.
  The local and remote constants exactly match the frozen A0 request's eleven
  source paths. The r8 fingerprint repair closes the two r7 HOLD witnesses.
- Real `-O`, `-OO`, inherited `PYTHONOPTIMIZE` (including empty and `0`),
  optimized compilation, and a direct child assignment have passing refusal
  checks. The full frozen A0 command, including the GNU-style
  `timeout --foreground --signal=TERM --kill-after=60s 3600s env -i` prefix,
  passes the parser and reaches only fake contact. This establishes no host
  safety or native result.
- The r3 request remains `dispatch_authorized=false` with old expected commit
  `e4272b4d...`; its base and pair bindings are `STATIC_ONLY` and pin old
  request/source hashes. There is no r8-bound enabled request or one-shot
  authority. The in-memory `dispatch_authorized=true` positive probe is a
  test fixture only.

The five-file r7-to-r8 diff is exactly the launcher, two launcher tests, one
r4 early-refusal fixture, and `README_R8.md`; `git diff --check` passes.
`train.py` SHA-256 remains `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`,
`prepare.py` (evaluator/scorer) remains
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`,
and the tokenizer archive remains
`12fa2c4a56d9f26d46e8ea0a57f69d78145930eecb0a765c9bfc085176af8a54`.
Their Git blobs, and guard/host/interlock/monitor source blobs, are unchanged
from r7. The r8 launcher SHA-256 is
`dcf9fdc0e1d8ecc459e6687b87511f00a5747d3fff19dfdfeae0a6b57d818c07`.

## Offline test record and disposition

| Check | Result |
| --- | --- |
| Supplied launcher suite, normal / `-O` / inherited `PYTHONOPTIMIZE=1` | 11/11 each, PASS |
| Supplied r8 bounded/fingerprint suite, same three modes | 7/7 each, PASS |
| Independent adversarial suite, same three modes | 7/7 each, including the two HOLD witnesses |
| Five retained root-free/paired/r4 monitor and interlock suites | 59/59, PASS |
| Direct harmless fake-`timeout` and child-Python probes | fake executable selected; optimized child observed |

Keep r8 source and all r5/r6/r3 requests disabled. A separate Implementer
must make the timeout executable identity non-bypassable and close the claimed
child-optimization gap, then seek a fresh independent source review. Only
after a source PASS may separate roles build and review a new exact-source
capsule, transfer/request bindings, G2 disposition, host preflight and
one-shot authority. This static HOLD is not native-launch authority.
