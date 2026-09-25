# Seed43 guarded fallback — independent mechanical binding review

Verdict: **PASS, scoped to the exact prepared binding**, not staging, launch,
runtime success, adoption or submission. No blocking new difference found.
The packet remains disabled pending the Supervisor's conditional next-slot
assignment and the sole Operator's fresh host checks. No host was contacted.

Reviewed Operator commit `90e942ead710adb7abdb085213476cd6dce8feb9`; canonical
import `26e45fb6842ade6a0280f24493564eef4b6d0620` contains identical packet bytes.
This is an unchanged-source mechanical review, not a new model-patch or fresh
model-session review. No provider/model session was called.

## Exact pins

| Item | SHA-256 |
|---|---|
| SEED_BINDING.disabled.json | `52e11b652a82a2c139287d48c7e85684e7e1dc2b7b92386176e8dcb7aa1fc662` |
| EXPERIMENT_PROPOSAL.json | `d71ed1281913ebccbb0896979235a06d6b82d60477a6ad0aeadb123c24ac13b4` |
| host_driver.py | `abc5ca596bdf35eda58b50a1841b55bc615ebf53dd2ee3c2e8b1a9f6f7428faf` |
| Exact adopted train.py | `6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3` |
| prepare.py | `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb` |
| submission_rehearsal.py | `7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b` |
| idle_guard.py | `adafb54a98527570a76af19ec2785e68496bf1027c51c4127664435c8c963737` |
| dispatch_driver.py | `c71474bc920f1bdba8f577d072497c4b94ca2c0318c29d39a37555da57ea041a` |
| Existing exact6ef7 source/G2 receipt | `ae1de1421162f3f7e91328e9eb522d6f06736e0bf8917a28aa64657435431282` |

Source was independently read from commit
`a4aa4351789115a2d0aa947d9ca00ff689f9bd0b:workflow/proposals/research-v2/train.py`.
The exact historical rehearsal was read from
`00f2ca7f:research/scratch/submission_rehearsal.py`; later historical versions
have different hashes and were not substituted.

## Scope and checks

- The driver's functions and entire top-level try/except AST match the
  already-reviewed architecture full driver at `83db5fe3` exactly. Only its
  docstring and literal source/host/label/path/command/environment bindings
  change. No train.py is added or edited.
- Proposal argv equals binding argv. Compared with the actual archived
  same-host seed42 training-launch.json, the sole training argv difference is
  `--seed 42` to `--seed 43`. Four ranks, LNC2, width1024, pack32, context1024,
  matrixLR0.015, Recursive ON, source budget1800 and max-steps1000000 remain.
  Both phase environments consistently specify seed43. The training compiler
  policy and six cache directories match the seed42 control. No batch/LR or
  architecture change is hidden here.
- Same common lock `/home/ubuntu/.cache/trainium_freshstart/device0.lock`,
  exact authenticated `/usr/bin/flock -n ... /usr/bin/python3 -I -B ...`
  parent, inherited exclusive FD, and synchronous phases are retained.
  Fresh ps AND neuron-ls checks run under that lock before training and again
  before attached eval; the FD must remain identical. Each immediate timeout
  child inherits that FD. Direct non-flock invocation refuses.
- Busy/malformed/error observations remain fail-closed. Training nonzero exit,
  absent/false budget evidence, missing checkpoint, second-phase busy state or
  changed FD prevent evaluation. Eval failure cannot become completion. There
  is no automatic next-job publication or old-runner rearming.
- The pinned rehearsal measures monotonic training-child spawn to process
  exit and subtracts `min(observed fifth boundary, 900)`, accepting only
  `charged_through_process_exit <= 1800`. The driver requires that true
  result before eval. Synthetic exact-code cases reject 1800.1 seconds both
  above and below the startup cap. The 2730-second outer watchdog and 60-second
  kill grace are failure containment, not extra accepted training budget.
- Evaluation is a separate process after successful training/child exit and
  a new guarded idle check, under the same lifetime lock. Exact harness argv
  calls unchanged prepare.py `eval-public`, Neuron, batch1/context1024,
  20,971,520 tokens, and the just-created `train/out/final.pt`. Evaluation has
  its own 1800-second outer timeout; this is not training-time allowance.

## Independent local evidence

`test_binding.py` was written independently for this review. It executed
73 checks, exit0, both normally and with the test interpreter's `-O` flag.
Actual driver/harness AST fragments are compiled with `optimize=0` because
the reviewed outer invocation is `python3 -I -B`, not `python3 -O`; this does
not claim that the assert-based driver can safely be launched with `-O`.

Checks cover pin identity, control argv comparison, literal bindings, unchanged
algorithm AST, actual preflight source/prepare/harness drift and STOP/disk
refusals, nine mocked phase paths, actual guard parsers including another
torchrun and ambiguous Neuron JSON, real local no-flock authentication refusal,
and exact harness budget boundary arithmetic. No trainer imports, tensors,
native execution or timing experiments were performed. The existing real
lock-lifetime repair review remains applicable because those bytes and the
FD propagation are unchanged; it was not represented as a new host test.

## Required operational limits, not new gates

This disabled packet has no staged package yet. During normal assignment and
staging the Operator must authenticate the actual `launch-command.txt` and
`provenance.json` against the reviewed argv/source, plus these exact support
hashes. The unchanged harness checks those package files against each other;
the driver does not independently hardcode the inner training argv. This is
the same trust boundary as the already-reviewed two-phase driver.

The rehearsal observes STEP_LOG every50ms; retain raw observations, wall,
capped allowance, charged-through-exit and source-clock figures separately.
Do not represent a sampled boundary as higher-precision startup evidence.
Intentional warm directories are not cache-hit proof or seed43 cold proof.
The inherited-lock repair protects the synchronous driver/outer flock and its
immediate timeout child; deeper harness subprocesses can close inherited FDs.
Abnormal parent/timeout termination still requires fresh quiescence checks
before reuse, as documented in the prior repair review.

The host is currently occupied by batch timing; this review neither stages
nor launches a job. Seed43 remains a variance diagnostic versus the proper
same-host seed42 control0.9899528600562257, not a seed-selected model win.
Fresh result, checkpoint identity, actual timing and raw20M arithmetic remain
pending independent result audit. Submission seed remains42.
