# Independent static PASS — disabled trn22 c5c A0 r10 native draft

Date: 2026-09-25 IST

Role: independent Verifier only

Subject: Implementer commit `8bdc727bf4f72955473bbc19c79272e46e5f8650` on
`origin/codex/20260925-trn22-a0-r10-native-r1-disabled`, seven files under
`research/v2/authorizations/trn22-c5c-a0-r10-native-disabled/`.

**Verdict: PASS for the exact disabled offline package; native dispatch and
evaluation remain HOLD.** This receipt is not a final enabled-request review,
Supervisor assignment, source-stage result, G2 source-identity decision, host
preflight, native result, or BPB measurement. I used an isolated clean verifier
branch and did not contact either host, run SSH, consume a marker, transfer or
delete anything, arm a monitor, train, evaluate, reset a breaker, change AWS
state, adopt a candidate, or submit.

## Exact bindings and disabled authority

| Frozen draft artifact | Independently checked SHA-256 |
| --- | --- |
| `NATIVE_REQUEST.DISABLED.json` | `3885415c561649bf4e8fe09e948e2793aec718b483566a4575ed0591d4b0977e` |
| `BASE_BINDING.STATIC_ONLY.json` | `4cc37ad06f23b0a17cf1225418e842870aeb3de6ed4086454ee23dfb6c6a0d51` |
| `PAIR_BINDING.STATIC_ONLY.json` | `9c067b8e97179f4ae3b6d4e410816a5e8826b94c8bda42ebb96bad05321b4700` |
| `PUBLIC20M_EVAL.DISABLED.json` | `45822174f7bd3842f89e2056cff821826de3e8ec7c81f56c3109e21c44da2ec2` |
| `GATE_CONTRACT.json` | `837ed3064d807c45d1174c4fe99af04c9b7fae399e45d9a8d673488a164e883d` |
| `RUNBOOK.md` | `a35df1bded2fb269b76aaeae306612121a4f7f8ea8b0f7cf6dd15fbd8eb884ab` |
| `test_disabled_package.py` | `8303f081c6834b6b76816e00317fad6c416278b9991af0bc4681c8d2da720d7f` |

The contract pins the request, base binding, pair binding, and evaluation-plan
hashes above; the base binds the exact request hash, and the pair binds the
exact request and base hashes. The contract's host, native, and evaluation
authorization fields are `false`; the native request has
`dispatch_authorized=false`; the evaluation plan has
`evaluation_authorized=false` and `checkpoint_sha256=null`; both monitor
bindings say `STATIC_ONLY`. All twelve required gates remain `false`, their
future result/assignment hashes are null, and the package contains no
one-shot authorization. The checked-in controller state has
`launch_authorized=false`, controller `enabled=false`, and only the legacy
one-off launcher identified as active. No second active launcher is proposed.

I executed the frozen local launcher `main()` against this exact disabled
request with a fake contact seam and empty receipt directory. Normal Python
refused the Boolean dispatch gate before a receipt or contact. Optimized
compilation, actual `python -O`, and inherited `PYTHONOPTIMIZE=1` also refused
before contact. The embedded remote script was run only in a local Python
child: disabled normal mode and optimized modes exited before its timeout,
checkout, lock, manifest, or guard paths. This closes the r5 assertion-bypass
failure for this frozen source/package combination. It proves no-contact
refusal for these offline cases; it does not authorize an enabled request.

## Science, safety, and evaluation findings

The proposed scored checkout is the distinct
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r10_stage_r1`.
Its frozen source is
`22bfbab5a77d6feb670d2d737504f8b70b08ddc1`, separate from the
Implementer package commit. The eleven source hashes in the contract,
request, and base binding match Git blobs at that commit. Key pins are:

| Frozen source | SHA-256 |
| --- | --- |
| `train.py` | `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586` |
| `prepare.py` | `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb` |
| `research/fresh_start_launch.py` | `1be19bc31d2bdcde8abe9fe90cc1ac884d6efd5f7be197d7bdbfe64c257dcff7` |
| `research/fresh_start_guard.py` | `df902becb2faf009d04d7ecfaee3791df67e822e90ec6112adbc39be81599ecc` |
| `research/fresh_start_monitor_interlock.py` | `f2fcecd427ea0c4fd5396799fe845d327c2e1384b05e1ef9445b8fb3cf9d2b4d` |
| `research/v2/disk-monitor/root_free_guard_monitor.py` | `91dae71468d5d029857371ccd2c960769b71874e855bb058130e738b91bb2c60` |
| `research/v2/disk-monitor/paired_root_free_guard_monitor.py` | `9e2d6ed26d918131c3e12b9937093e465b39aa895a9bf2be7339de8ae5094344` |

The `e046de16` r9-to-r10 source diff changes only the launcher, its tests,
and launcher documentation; it does not edit `train.py` or `prepare.py`.
The package commit itself adds only the seven draft files listed above.
The r10 source Git tree contains `research/run-logs` for the step-log parent;
its presence in a future host checkout still needs direct verification.

The request's full ordered child argv is the bounded absolute GNU `timeout`,
absolute `env -i`, fixed native `torchrun --standalone --nproc_per_node=4`,
and the unchanged dense c5c Track-A flags ending in `--matrix-lr 0.015`.
The effective science environment pins LNC=2, seed 42, 1,800 training
seconds, `--optlevel=1` BF16 compiler flags, and all six named prior-trn22
warm cache paths. Both local and embedded remote exact-profile validators
accepted the frozen request and rejected twelve independent child-command
or environment mutations apiece, including an extra flag, altered timeout,
relative executable entry, world size, seed, budget, compiler flags,
cache, `PATH`, and `PYTHONOPTIMIZE`. The r10 launcher does not hard-code the
trn22 label or checkout; this draft's exact request, hash chain, and the
future final enabled-request review must bind them. No unreviewed profile
substitution is accepted by this receipt.

The request and monitor drafts preserve the 10,737,418,240-byte prelaunch
root-free gate, 6,979,321,856-byte operational floor, and
9,126,805,504-byte paired-monitor stop threshold. The base monitor polls at
0.25 seconds against the one-second maximum. The frozen interlock checks
the fresh 10-GiB free-space value, mutual live monitor samples, request and
binding hashes, and one-shot authorization before guard spawn. The external
boundary observer is pinned to SHA-256
`a6f437fe8a9d0a14ec3025bc822f1afa12e3f355fca730e4ae0db7ae00263592`
and is required before guard launch, but its host bytes and liveness have
not been checked. STOP, breakers, one-run-per-host, clean matching Git HEAD
on both hosts and local science checkout, idle process/device/host lock,
protected checkpoint and six cache identities, label absence, and disk
projection are future live Operator/final-request gates. The disabled draft
does not perform them. It permits no cache cleanup, checkpoint pruning,
protected deletion, or AWS state action; the frozen guard's prior-run
cleanup is inventory-only.

The old trn22 r2 proposal/ratification is for the r2 label and e65 checkout;
the earlier four-rank, three-step G2 is CPU-numeric evidence on e65. Neither
binds this r10 label or proves the staged r10 checkout. The r10 source-stage
r3 static PASS is disabled-only and has no independently accepted host-stage
result. The active controller's 10:21 UTC dual-host snapshot is point-in-time
planning evidence, not launch preflight.

The disabled evaluation plan matches accepted trn22 G1 settings: a separate
fresh process, trusted unchanged `prepare.py eval-public`, Neuron batch size 1,
context 1024, exactly 20,971,520 public tokens, expected 83,462,186 public
bytes, and a 1,200-second evaluator timeout inside a 1,800-second outer
bound. A 2M screen cannot establish this control. The plan has no actual
checkpoint hash or evaluation assignment and remains disabled.

## Offline checks and remaining gates

| Check | Result |
| --- | --- |
| Implementer disabled-package suite | 6/6 PASS under normal Python and `-O` |
| Frozen r10/r9/r8/nonassert launcher suites | 31/31 PASS under normal Python and `-O` |
| Frozen interlock, root monitor, paired monitor, containment/host-cleanup suites | 68/68 PASS under normal Python and `-O` |
| New independent disabled/refusal/profile tests | 3/3 PASS under normal Python, `-O`, and inherited `PYTHONOPTIMIZE=1` |
| Frozen source fingerprints and upstream evidence hashes | 11/11 source blobs and cited evidence match |
| Package and r9-to-r10 source diff whitespace checks | PASS |

The new independent regression test is
`research/v2/verification/test_trn22_c5c_a0_r10_native_disabled_independent.py`.
The monitor-suite `HOLD` lines are expected synthetic negative cases; each
suite exited zero. No Neuron or host test was run.

Before any native attempt: independently accept an exact final r10 source
stage and its Operator result; bind the actual staged bytes/argv/environment
to an independent r10 G2 carry-forward decision or perform a new G2; complete
and ratify an r10-label proposal; prepare a new enabled request and monitor
bindings with an independent final review; perform fresh complete dual-host
and local-source preflight including STOP/breakers, protected assets, disk,
cache, label, lock and Neuron checks; prove two live monitors and the observer
before guard; then issue a separate one-label one-attempt Supervisor
assignment. After a run, retrieve full logs, boundary/monitor/guard receipts,
budget and finite checkpoint evidence, prove unconditional quiescence and
postflight, and obtain independent terminal review. Only a separately
reviewed and assigned fresh public-20M evaluation bound to the actual
checkpoint may establish a same-host A0 control.
