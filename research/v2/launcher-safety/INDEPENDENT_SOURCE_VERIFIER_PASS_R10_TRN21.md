# Independent Verifier: disabled r10 trn21 A0 launcher source

**Verdict: PASS for the disabled source profile only.** The reviewed subject is
`22bfbab5a77d6feb670d2d737504f8b70b08ddc1`, directly descended from
the independently source-PASSed r9 launcher
`e046de16826bfc3b954136e3c60804aea0554ecc`. The r10 diff changes only
`research/fresh_start_launch.py`, two retained launcher tests, the new r10
launcher test, and `README_R10_TRN21.md`. I reviewed the source independently
in the Implementer's clean, pushed worktree, then ran offline tests with fake
SSH/Git/preflight/interlock/guard seams. I did not contact either host, stage
source, enable a request, start a monitor or guard, train, evaluate, or change
AWS or a breaker.

## Binding and source findings

- The local and embedded remote `bounded_timeout_command` function ASTs are
  equal after excluding the local docstring. The new `trn21` branch requires
  host `trn21`, hostname `ip-172-31-34-85`, label
  `fs-20260925-v2-c5c-a0-control-trn21-r1`, and the distinct proposed
  checkout `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_r10`.
  The hostname and label match the previously ratified trn21 proposal and
  disabled native skeleton. The r10 checkout is **proposed, not observed or
  staged**. Old `stage_r2` and reserved `r4` checkouts are refused even when
  their output/step-log argv are recomputed.
- Both validators require the complete ten-key child environment: LNC=2,
  four ranks, seed 42, 1,800-second training budget, the exact optlevel-1 BF16
  compiler flags, and all six cache paths under the prior trn21 control's
  warm-cache root. The entire ordered command, not a prefix, is matched:
  absolute GNU `/usr/bin/timeout` with finite outer deadline, absolute
  `/usr/bin/env -i`, fixed home/path/venv/unbuffered assignments, exact
  output/step/run-ID assignments, absolute native `torchrun`, and the
  6x1024 dense A0 training flags through matrix LR 0.015. No extra flag,
  wrapper, child environment key, altered cache root, or caller PATH is
  accepted. The trn21 remote additionally hashes `train.py` and `prepare.py`
  before guard spawn. Actual subject bytes are respectively
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
  and `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- The trn22 branch retains r9's cache root, exact environment and entire
  argv; the retained r9 positive and hostile tests pass on r10. Trn22's
  label/checkout are still bound by its separately reviewed request rather
  than a new hard-coded r10 label. This review does not broaden trn22 launch
  authority.
- `fresh_start_guard.py`, `fresh_start_host.py`, `fresh_start_sync.py`,
  `fresh_start_monitor_interlock.py`, and both disk-monitor production files
  are blob-identical to r9. Thus the reviewed early GNU-timeout check,
  optimized-Python refusal, source fingerprint list, disk floors, process and
  Neuron-holder preflight, exclusive host lock, paired-monitor liveness and
  stop checks, no-spawn receipts, and guard containment are not weakened by
  this diff. `git diff --check r9..r10` passed.

## Independent offline checks

| Check | Result |
| --- | --- |
| Four supplied launcher suites (`nonassert`, `bounded_r8`, `exact_r9`, `trn21_r10`) | 31/31 PASS under normal Python, `python -O`, and inherited `PYTHONOPTIMIZE=1` |
| Retained root-free, paired-monitor, r4 host-cleanup, early-refusal and interlock suites | 59/59 PASS in normal Python |
| Independent local/embedded-remote early-refusal matrix | 13 JSON mutations × both validators PASS; a direct non-JSON tuple-command attempt also refused |
| Local versus embedded remote exact-command validator AST | Equal |
| r10 runbook/source fingerprints and protected training/eval bytes | Match actual SHA-256 |

The independent matrix covered a truthy non-Boolean authorization, wrong and
cross host, wrong physical hostname, changed label or checkout with recomputed
argv, changed cache with recomputed argv, extra environment key, extra train
flag, Boolean disk floor, reduced prelaunch disk gate, duplicated fingerprint,
and malformed manifest policy. Every JSON mutation failed before a local
dispatch receipt/fake SSH and before remote timeout probing, Git, lock,
preflight, no-spawn receipt, manifest, job, log or fake guard contact. The
positive trn21 route reached **only fake contacts** in the supplied test.

For transparency, unconstrained discovery of all disk-monitor tests reports
three failures in the historical r3 adversarial suite on **both** r9 and r10;
this is not a changed r10 source path. Running the retained early-refusal
suite under `-O` reaches the intended launcher optimized-Python refusal rather
than its unoptimized fixture expectations. The r10-specific and full launcher
suites above explicitly test optimized mode and pass.

## Limits and next gate

This is **not** a runnable request or host-stage verdict. No r10 checkout,
source-matched native G2 decision, enabled exact request, live host preflight,
paired monitors, boundary observer, one-attempt Supervisor assignment, scored
control, checkpoint, or BPB result follows. A future request must pin this
exact r10 commit and source hashes and pass independent request review; the
Operator must freshly check host identity, label availability, disk,
processes, Neuron holders, lock, STOP/breaker state and all protected assets.
The earlier CPU G2 numeric PASS was on the older e65 checkout and cannot by
itself authorize native r10 training.
