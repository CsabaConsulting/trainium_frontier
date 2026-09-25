# Independent Verifier receipt: trn22 c5c A0 r5 offline package

Date: 2026-09-25 IST

Role: Verifier only; not Implementer, Supervisor, or Operator

Verdict: **HOLD — no host contact, source transfer, checkout, monitor arm, native dispatch, or evaluation**

This receipt reviews the seven-file DISABLED draft in
research/v2/authorizations/trn22-c5c-a0-r5-native-hold/ at Implementer
commit 9f5459e670f3feb8422222e49d0462ea81c16974. It was produced on
the separate branch codex/20260925-trn22-c5c-a0-r5-package-independent-review.
The scored-source target is the distinct frozen commit
446b657695f086f9b484fa601c9a3b61c2495912. No host, SSH, Neuron,
training, evaluation, or protected artifact was touched in this review.
This is not self-approval of an Implementer package.

## Decisive blocker: disabled local launcher is not an unconditional no-contact gate

The package RUNBOOK says the frozen launcher checks
dispatch_authorized=false before it can create a local dispatch receipt or
contact SSH. In frozen research/fresh_start_launch.py, line 174 implements
that refusal with a Python assert. Lines 175–188 use more asserts for
request validation, line 190 writes the local dispatch receipt, and line 195
calls subprocess.run for SSH. Python optimization removes those asserts.

I compiled the exact frozen main() AST with optimize=0 and optimize=1,
passed the exact disabled r5 request, and replaced subprocess.run with a
fake function that raises FAKE_HOST_CONTACT_REACHED. With optimize=0 the
disabled request was refused before receipt or contact. With optimize=1 the
launcher wrote the local dispatch receipt and reached the fake SSH seam.
The independent regression test is
research/v2/verification/test_trn22_c5c_a0_r5_static_negatives.py.
There was no real SSH call. This establishes a local launch-safety bypass,
not a claim that the remote interlock would spawn a guard or training.
It falsifies the draft's unconditional no-contact claim, so a static PASS
would be unsound even though every package authorization field is false.

Do not flip this draft in place or dispatch it. A distinct successor must
put an explicit non-assert optimization/environment gate before invoking
the frozen launcher, so an optimized interpreter or unsafe environment
cannot make a disabled request reach a receipt or remote side effect.
It must carry a negative optimized-interpreter test. The other
assertion-based local validations need equivalent protection; if the
launcher itself changes, rebind its exact source hash and reassess G2
identity. Obtain another independent review before any authority decision.

## Exact-byte findings that do not cure the HOLD

The draft request, base binding, pair binding, and disabled evaluation plan
have the following independently computed SHA-256 values. The base embeds
the request hash, the pair embeds both preceding hashes, and the contract
pins all four. These are draft-byte bindings, not a one-shot authorization.

| File | SHA-256 |
| --- | --- |
| NATIVE_REQUEST.DISABLED.json | 086b78e315050d3a0a44014e03c2ab28d4279eb5f9c0f2f4882447ed2ab461be |
| BASE_BINDING.STATIC_ONLY.json | 4f3af43cae356f58802232bb47c8c3a12b323e7ffaa140b40025216f13dc3b9f |
| PAIR_BINDING.STATIC_ONLY.json | d45e2c680a15c8ba58858d2ddb9b285c4868cb739cf4d394172e70b3d7bbfce1 |
| PUBLIC20M_EVAL.DISABLED.json | 9714c20f8040d3bd9bee0879280db2353c06f9060a11e35ddcf78a3cf4652888 |
| GATE_CONTRACT.json | c36fa90380431cf1260d93f3c4e42b435a7279cd295af8d249e5f9e562ae125d |
| RUNBOOK.md | ae64fc116e4dc3474b786bf431944b8808d7d178b76e20dc244eff66eaff4a9c |
| test_disabled_package.py | 1fd8a53fc9c87c60d60dddd21f1ff6bb708219d8c4cb5a47d2ecc3895f3aaffe |

All eleven production source SHA-256 values in GATE_CONTRACT.json, request,
and base binding matched blobs read directly from frozen 446b6576:

| Frozen source path | SHA-256 |
| --- | --- |
| train.py | c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586 |
| prepare.py | 5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb |
| research/fresh_start_backup.py | c978fd20b6620d0d80c25f8982974e8c4b2ad009edd9da376d784f5cecc3ead3 |
| research/fresh_start_guard.py | df902becb2faf009d04d7ecfaee3791df67e822e90ec6112adbc39be81599ecc |
| research/fresh_start_host.py | b34c368c12175615514e63daebc296ec09332ef801e176e0d19cc3edc045fed3 |
| research/fresh_start_launch.py | 9488efa918119439a33cbe68cb5638bd1af3b15f1f8f61ec790a216e2bc1e3f4 |
| research/fresh_start_monitor_interlock.py | f2fcecd427ea0c4fd5396799fe845d327c2e1384b05e1ef9445b8fb3cf9d2b4d |
| research/fresh_start_sync.py | ea428d33f4de18376950abbd81cfda33d3a969f75b60e3648ff552bea04fc60c |
| research/run_experiment.py | 88243a74f0f080929f93d0b88ddedbe6f5351ad37bf272229d9ae6d26efd5d2d |
| research/v2/disk-monitor/root_free_guard_monitor.py | 91dae71468d5d029857371ccd2c960769b71874e855bb058130e738b91bb2c60 |
| research/v2/disk-monitor/paired_root_free_guard_monitor.py | 9e2d6ed26d918131c3e12b9937093e465b39aa895a9bf2be7339de8ae5094344 |

The local r4 capsule was 104,154,875 bytes, mode 0600, one link, and
SHA-256 767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a.
Its member manifest SHA-256 was
7ec8798c3e6d0284a4514a15a0caeb17bc5eb2cb5df935fb7dd6e8f3aabee42a.
Neither local hash proves transfer or host checkout.

The r5 label is new relative to its parent commit:
fs-20260925-v2-c5c-a0-control-trn22-r5. The request deliberately retains
the prospective r4 checkout path, while r5 output, run-log, monitor, and
observer receipts have r5-specific names. Its ordered A0 training argv
matches archived r3 G2, including matrix LR 0.015. The child uses
seed 42, NEURON_LOGICAL_NC_CONFIG=2, the same compiler flags and six
warm training-cache strings, a 1,800-second training budget, bounded
timeout, and four-rank full-chip torchrun. No checkpoint or result is
claimed. The request's environment is reset with env -i; this does not
protect the outer Python launcher from optimization.

G1 calibration evidence is the accepted fresh-process public-20M
context-1024 result: 20,971,520 tokens and 83,462,186 public bytes.
The r3-to-r4 G2 disposition is numeric-only carry-forward, conditional on
exact staged source, ordered argv, seed, Neuron configuration, compiler
flags, budget, and all six cache strings. Those static strings match r5,
but there is no accepted r4 host checkout or native-safety proof. A host
identity mismatch would void carry-forward.

The disabled evaluation plan has evaluation_authorized=false,
checkpoint_sha256=null, a new r5-public20m label, fresh_process=true,
20,971,520 tokens, context 1024, six warm eval-cache strings, and a
1,200-second evaluator limit inside a bounded command. It is not an
evaluation request, cannot use a 2M screen as the decision, and requires
a separately accepted training terminal receipt, actual checkpoint hash,
fresh preflight, assignment, and independent review.

## Prior zero-step step-log parent failure: independently attacked

The earlier r2 CPU G2 failed at record_step_boundary(0) because its
NEURON_COMPETITION_R1_STEP_LOG parent was missing; frozen prepare.py opens
the path for append without creating its parent. The r5 draft also points
under the prospective checkout's research/run-logs. Direct execution of
the frozen function reproduced FileNotFoundError for a missing parent and
success once the parent existed.

Here the frozen 446b Git tree actually tracks research/run-logs with
5,560 file children. The r4 capsule manifest and the capsule tar itself
contain the parent directory (mode 0700) and those children. Therefore
complete reconstruction of the reviewed tree should provide the parent
without manual in-checkout mkdir or a dirty-tree exception. This is
offline source/capsule proof, not proof that a future trn22 checkout has
the directory. Final checkout and prelaunch acceptance must explicitly
lstat the real parent as a directory (no symlink, correct owner/mode),
check the r5 step file absent, and separately certify the clean exact
tree before generated run artifacts. If reconstruction omits that
tracked parent, HOLD rather than creating an unreviewed repair.

## Remaining gates and offline checks

The r4 transfer is under an independent HOLD. No accepted transfer,
clean host checkout, r5 completed proposal/ratification, final enabled
request, one-shot assignment, or r5 host result exists in this review.
The r2 ratification covers its r2 label only. research/v2/state.json has
launch_authorized=false. Every required gate in this draft's contract is
false. Before any successor could run, the protocol still requires full
both-box preflight and source sync, clean exact checkout, idle Neuron and
locks, protected assets and six cache identities, absent label/outputs,
at least 10,737,418,240 root-free bytes, live authenticated paired
monitors with stop at 9,126,805,504 and immutable floor 6,979,321,856,
and read-only boundary observer before guard. These were inspected as
requirements, not performed. Terminal and postflight retrieval would be
separate later duties.

The draft poll interval is 0.25 seconds against the one-second maximum.
Its observer script pin
a6f437fe8a9d0a14ec3025bc822f1afa12e3f355fca730e4ae0db7ae00263592
matches both the repository observer and the archived r1 observer bytes.

Offline test record:

| Check | Result |
| --- | --- |
| Supplied disabled-package unittest | 9/9 PASS |
| Frozen monitor/interlock/host test files, direct execution | 72/72 PASS |
| Independent negative tests in this branch | 3/3 PASS; optimized bypass reproduced |
| Seven in-memory mutations of supplied-package objects | 7/7 rejected |
| Direct frozen source SHA verification | 11/11 match |
| Request → base → pair binding and disabled eval pins | PASS |
| Frozen capsule run-log parent and children | directory mode 0700; 5,560 files |
| Ordered G2-to-r5 training command and effective settings | PASS; 10/10 seed/config/budget/cache fields |
| train.py and prepare.py e65a3097-to-446b diff | no changes |
| Implementer commit diff whitespace check | PASS |

The source-monitor tests passed by executing each test file directly.
A combined local Python 3.14 unittest invocation using slash-path arguments
returned six import errors because it treated those paths as module names;
that invocation is not counted as a passing suite. All work was CPU/static
and offline, with temporary files and a fake host-contact seam only.

**Disposition:** HOLD the exact seven-file disabled r5 package. Do not infer
launch authority from correct hashes, successful static tests, or this
receipt. A distinct reviewed successor and every ordinary role/host gate
remain necessary.
