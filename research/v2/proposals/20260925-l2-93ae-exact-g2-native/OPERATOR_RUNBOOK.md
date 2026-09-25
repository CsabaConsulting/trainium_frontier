# Operator handoff: exact 93ae architecture

This is a prepared, disabled packet. It has no live-host observations.
Read the current Supervisor assignment and newest repository directives
before acting; this packet does not replace them.

## 1. Choose a free slot and stage immutable bytes

Use the selected host's JSON directory. The proposed staging root is
`/home/ubuntu/trainium_l2_unet8x1024_93ae_20260925_<host>_r1`.
Both host roots, all CPU labels and native labels are explicit in those files.
The local canonical snapshot has no queue/ledger collision for the 12 labels;
the Operator must check live directories/labels again before any stage.

Let active training and its fresh evaluation finish. Coordinate PAUSE_NEXT
or the existing between-job mechanism so a fallback cannot race the CPU slot.
Do not terminate the active run or start a second launcher. Check canonical
and host STOP/breakers, no train/eval/G2 process, no Neuron-device holder,
disk >=6.5 GiB plus projected outputs, and MemAvailable >=64 GiB. Acquire
the existing `/home/ubuntu/.cache/trainium_freshstart/device0.lock`.
Use the current discovered STOP set; the archived root list is not a complete
future inventory.

Stage only exact Git-object bytes listed in `sources.json` into new paths.
The roles are candidate `train.py`, `control-c5c.py`,
`control-6ef7.py`, unchanged `preflight_dryrun.py` and `prepare.py`.
The CPU plan's prepare_dir is an existing known host source location and
must have the pinned prepare hash and required native environment. Record
local/remote SHA equality, source commits and exact-file provenance. Keep
source inputs clean and immutable throughout all probes. Do not install
packages or copy the tiny-test PyArrow/prepare stubs.

This packet deliberately contains no transport or replacement launcher code.
Use the proven exact-file staging and detached CPU execution pattern from
`research/v2/operator/20260925-trn21-l3-timer-dense-g2-r1/`, with this
packet's explicit roots, pins and command lists. Preserve its fresh
preflight/common-lock behavior; do not reuse its consumed label or hardcoded
L3 root.

## 2. Run the four CPU probes serially

For each entry in `cpu-g2-plan.json`:

1. Create its new cwd and tmp directory; reject a pre-existing/consumed target.
2. Recheck inputs, idle state, STOP, disk/RAM and retained common lock.
3. Use native_venv, one OMP/MKL/OpenBLAS thread, LNC2 and the listed compiler
   flags. Bind TMPDIR to that entry's temporary_root and TRN_EXP_LABEL to
   its unique label. The source/prepare/G2 file hashes must match.
4. Execute its exact command with the listed working directory and retained
   stdout/stderr. The unchanged harness appends CPU/no-compile/three-step/tmp
   output overrides and uses synthetic data; never add --real-data.
5. Capture return code, g2.json, all four raw rank JSONs, complete torchrun
   log, exact argv/environment and checkpoint URI/hash. Keep every artifact.
6. Require terminal completion and a clean read-only postflight before the
   next probe. Do not clean failed outputs, kill unrelated processes, or
   relaunch a failed label.

Full width/depth must appear in ON model_config: n_layer8, n_embd1024,
eight heads, MLP ratio4. ON must show attention0,2,4,6,7, four scalar skips,
VE only7 and no n-gram tables. OFF/reference shapes are dense6x1024.
All ranks must finish three finite updates without parameter discrepancies;
the checkpoint must strict-load to finite deterministic logits.

Run both exact `--compare` commands after the four successful probes.
Require `identical: true`, identical ownership plans, all 12 rank-step
losses and all 62 dense parameter digests. Preserve comparator output.
The c5c reference intentionally predates the timer fix; three G2 steps
and the documented fake clock establish dense arithmetic identity, not
full-run timing equivalence. The 6ef7 reference has the same timer fix
with Recursive default-OFF.

A separate reviewer checks the raw evidence and source/argv bindings.
Do not silently copy these CPU results into a native PASS, a gain claim,
or launch authorization.

## 3. Activate one full-budget architecture job only after gates

The Supervisor chooses the exact host proposal and unique native label.
If queue support has been independently extended to 93ae, bind the actual
PASS receipts and proposal SHA in the approved-source registry and replace
only an unstarted fallback. Otherwise use the already checked manual
train-to-fresh-eval machinery after the existing queue is paused between jobs.
The current unmodified queue rejects 93ae; bypassing that rejection is not
part of this packet.

The native JSON supplies the exact torchrun command, training environment,
new working directory, checkpoint and separate evaluation command. Stage
candidate train.py into the new native cwd; verify SHA again. Record the
effective warm-cache environment, without clearing cache directories or
claiming cache hits. Preserve the common lock across training and evaluation.
Use the existing step-boundary/budget observation and durable detached driver,
not a bare foreground command with no retrieval/continuation plan.

Confirm four workers appear. Track finite losses, startup and the existing
in-file budget stop; a crash, NaN, compilation failure, missing checkpoint,
failed causality check or budget failure is a failed gate. Do not adjust the
timer, reduce chip use or retry the consumed label.

After training reaches a clean terminal state, strict-load the final checkpoint
and use the unchanged scorer's fresh-process public-20M/context1024 command.
Archive nats/bytes/BPB, actual token count, timeout/budget status, step count,
checkpoint URI/SHA, complete logs, environment and postflight/quiescence.
Record native causality/batch-independence checks; do not infer them from
the CPU-only receipt. Retain protected artifacts; no deletion is authorized.

Use the selected same-host c5c control with current control-age/drift checks.
Run the unchanged mechanical decision rules with `architecture_arm`;
this packet does not grant a step-window exemption. Neither live Recursive-ON
results nor a cross-host control substitute for the matched dense baseline.
No leaderboard submission or submission-branch merge is authorized.
