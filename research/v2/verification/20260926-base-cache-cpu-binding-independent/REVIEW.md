# Existing base-cache CPU packet: independent mechanical review

## Narrow repair re-review

Repair `2cf504d9b763e98e7130447d52fc93799e26db50`: **PASS for the exact
trn22 CPU-only mechanical binding**. No remaining blocking defect found in
this bounded review. It is eligible for a separate scoped Supervisor/Operator
CPU assignment; this receipt itself performs no staging or launch and does
not authorize trn21 relocation, native/cold/full execution or adoption.

The repair captures the parent PID immediately before the unchanged shared
guard authenticates it, requires that same parent at `checks()` entry, and
excludes only ownPID plus that exact authenticated PID. Another torchrun or
host_driver is still refused; a changed parent is explicitly refused. Shared
guard/dispatch, production sources, all four request bytes, commands,
resource floors and CPU workload are unchanged. All four driver repairs
have identical ASTs apart from their literal ROOT.

Independent replay:74 checks pass normally and with `-O`, including the
same original self/parent fixtures on all four repaired drivers, competing
torchrun, unrelated driver, wrong excluded PID, changed actual parent and
capture/authentication/check ordering. Original default invocation still
reproduces all four failures against immutable5a3c5358 (54 checks); its raw
receipt below is retained unchanged. Supplied static tests also pass, but
were not substituted for independent checks. No tensor/host/native work ran.

| Repaired binding | SHA-256 |
|---|---|
| CPU_PLAN.disabled.json | `b0c2a4abbce771a344e886d8a6da0723d42314997db245bdccb6dcd5b39202bc` |
| SHA256SUMS | `00fe8964313a08558d30b039b55142d092e60613e7b49f5ed7891d922e4ab17c` |
| reference driver | `45c41cb51b52fdcb5917b923e165019831c46a611c83af834564f329d553fffc` |
| default driver | `2126c47932910afe9422f20d6a78105a06f02e3208e48cda32fba819f645f9d2` |
| OFF driver | `0be8c1cc6164c71d260ffa6257dc8e1888de92f40b4e4f4c40de42cc1ff04dd9` |
| ON driver | `c42212bae82271d14a59306671881e9a2f2b8940d71fdd705636654266f08529` |

The full-parameter CPU numeric gate has not run. Its results, independent
numeric acceptance and fresh isolated source review remain pending. All
scope and descendant-lifetime limits documented below remain applicable.

## Preserved original verdict

Original packet `5a3c5358473e3463b19aadeafeac6212a1b24f96`: **HOLD**.
CPU plan SHA256 `2b08cbea8ed011e1be418e0faf7bb62ac9e394f211105557dbd45d5d51731f09`.
This is a CPU diagnostic binding review, not the separately required fresh
isolated source review, native correctness, startup, performance or adoption.

## Blocking finding

Each of the four `host_driver.py` files, lines50–60, runs a second legacy
`/proc/*/cmdline` workload scan. It excludes only `os.getpid()`. The required,
authenticated outer `/usr/bin/flock ... <arm>/host_driver.py` parent contains
`host_driver.py` in its argv; the scan therefore classifies that very parent
as another workload and raises `OTHER_WORKLOAD`. This occurs after the shared
idle guard passes and before any G2 child can start. Every prepared arm is
unlaunchable under its required invocation. It is a false refusal, not a lock
bypass or a model numerical failure.

Independent `test_packet.py` executes the exact immutable legacy-scan AST
from all four drivers with ownPID100 and authenticated outerPID99. Each yields
`active=[99]`, `OTHER_WORKLOAD:[99]`. Adding another torchrunPID200 yields
`[99,200]`. The 54 binding/reproducer assertions pass normally and under `-O`;
that means the failure was reproduced, **not** that this packet passes.
Raw structured outputs are retained in `original-result.json`.

Required narrow repair: exclude exactly the previously authenticated driver's
outer parent PID as well as the driver itself. Do not exclude all ancestors,
all Python/flock processes or process names, and do not remove the shared
guard, fresh ps/neuron-ls checks or inherited-FD propagation. Re-run this same
fixture with only self+authenticated parent accepted and every independent
training/driver process still refused. Root and Implementer were notified
immediately; Verifier has not edited the candidate packet.

## Remaining binding inspection

The six source/support pins independently match: exact adopted6ef7
`6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`,
existing bcbd candidate
`bcbd148dd51b5a014289cbecf3f0e684e35cf537e9588c878e15b265fcbfd522`,
standard G2
`6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`,
real prepare
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`,
shared guard
`adafb54a98527570a76af19ec2785e68496bf1027c51c4127664435c8c963737`,
and dispatch
`c71474bc920f1bdba8f577d072497c4b94ca2c0318c29d39a37555da57ea041a`.
Candidate is the already-frozen four-line initial-refresh addition to6ef7;
this packet adds no production patch.

All four request, command and driver hashes match the plan. Reference,
candidate default, explicitOFF andON requests differ only in appropriate
source and final refresh flag. All use the standard G2, seed42/RecursiveON,
four CPU ranks/three updates, real prepare import, synthetic2x256 tokens,
no `--stub-prepare` or `--real-data`, and retained checkpoints.

The G2 source was read without importing it: appended `--device-type cpu`
and `--no-compile` force CPU/gloo and replace the million-step launch limit
with three steps; BF16 compute, skipped Neuron CE install, fixed1ms clock,
synthetic tokenizer/data and forward input slicing are explicit shims.
All117,453,580 parameters/62 tensors are retained; physical tokens/update
are2048 across ranks, **not** production131072. Full model does not mean
full production workload or native kernel execution.

The driver has hostname binding to trn22 `ip-172-31-32-253`, 64GiB available
RAM and6.5GiB disk floors, one OMP/MKL/OpenBLAS thread per worker, STOP and
source/support checks, one synchronous G2 child, and inherited authenticated
lock FD. Its standard inner timeout is3600s. This is not a measured peak-RAM
or runtime guarantee; no laptop allocation is allowed. The unchanged inner
torchrun does not retain arbitrary inherited descriptors and timeout does
not prove descendant cleanup. Retain the documented stop/quiescence rule
before any later arm; no automatic sequence is supplied.

Declared comparisons are appropriately limited:12 raw losses,62 rank0
16-hex final parameter digests, rank consistency/ownership plan, deterministic
finite checkpoint reload and output shape. Standard G2 does not expose
intermediate gradients/optimizer states/per-step parameter or logit digests.
The driver streams the retained whole-checkpoint SHA256 without tensor load;
the result auditor additionally checks explicit child/torchrun exits and
non-timeout status, which the older G2 PASS label alone is insufficient to
prove. Cross-arm checkpoint-byte equality is a pending criterion, not an
already observed result. No numeric receipts exist in this packet yet.

Cold proposal stays separately disabled, five-step startup-only target600s;
there is no native/device job, automatic cold/full/eval, source adoption or
host reservation authorized by this review. A trn21 relocation requires a
new exact binding review. Sole Operator assignment and raw-G2 independent
numeric/fresh-source review remain separate.

No host calls, training, model import, tensors, source/ledger changes or model
sessions occurred. The preparer's static script also prints PASS, showing
why that label alone did not catch this process-tree integration defect.
