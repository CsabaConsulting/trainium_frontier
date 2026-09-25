# trn22: batch262144 full-parameter CPU gate

Prepared CPU-only packet, not staged or executed. Root must review before the
distinct Operator takes the assigned next idle boundary. No new launcher,
approval framework, source patch, queue mutation or native authority is added.

Wait for `fs-20260925-l2-unet8x1024-93ae-trn22-full-r1` training **and its fresh
public20M evaluation** to finish with terminal receipts and quiescence. Never
interrupt either. Coordinate the existing between-job handoff so no fallback
or native control races this CPU slot; Operator/Supervisor owns that handoff.

## Inputs and staging

`sources.json` binds exact Git objects: candidate4e361d15/sourcea489f7,
referencea4aa4351/source6ef7, unchanged G2 and trusted prepare. Independent
source/reduced-CPU review is86597104. Freeze/extract those committed bytes,
not whichever research file happens to be at the canonical working-tree path.
This packet does not modify or restage canonical research source.

New stage root:
`/home/ubuntu/trainium_recursive_batch262144_cpu_20260925_trn22_r1`.
Stage candidate as `train.py`, reference as `control-6ef7.py`, unchanged
`preflight_dryrun.py`, plus each exact request/driver pair at its declared
probe cwd. The existing trusted prepare directory remains
`/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1`;
verify actual imported prepare path/hash, do not copy/edit evaluator or data.
No PyArrow/stub-prepare/test_support shim or package installation is allowed.

Prove all new stage/probe/tmp/output targets absent before privately creating
them. Preserve all prior artifacts. Capture local/remote hash equality and
clean committed provenance. Live preflight must recheck hostname
`ip-172-31-32-253`, canonical and freshly discovered host STOP/breakers,
disk>=6.5GiB plus projected retained outputs, MemAvailable>=64GiB, no active
train/eval/compiler/G2 workload, no Neuron holder, and the existing common
`/home/ubuntu/.cache/trainium_freshstart/device0.lock`. Do not infer idleness
from this prepared packet or stale archived process lists.

## Reuse existing driver, three serial probes

The three `cpu_driver.py` bindings differ from the reviewed trn22 architecture
candidate-off driver **only in its ROOT path literal**. Host, hashes, STOP,
memory/disk/process/device checks, clean environment, subprocess handling,
exclusive records and terminal behavior are otherwise byte-identical.
Their complete inner/outer command arrays are in `cpu-g2-plan.json` and
each `request.json`; hashes are in `SHA256SUMS`.

Use the existing detached launch pattern: outer command is `/usr/bin/flock
-n <common-lock> /usr/bin/python3 -I -B <exact-bound-cpu_driver.py>`, launched
with the established new session/private stdout-stderr/operator-launch
receipt behavior. Do not replace this with a bare G2 process, omit the lock,
or invent a launcher. Record driver/lock PID and exact outer command; no PID
in this packet is claimed live. The driver starts the unchanged native-venv
G2 tool on CPU, with one OMP/MKL/OpenBLAS thread and the established clean
environment. `--timeout3600` and `--keep` retain the existing bounded behavior.

Order:

1. `reference-6ef7`: exact adopted source with Recursive ON, batch131072.
2. `candidate-default`: sourcea489 with identical argv, new flag omitted.
3. Run the exact existing `--compare` command in the plan; preserve stdout
   as `default-lockstep.json`, require exit0 and `identical:true` before ON.
4. `candidate-262144`: same candidate/argv, plus only
   `--total-batch-size262144` as the two separate argv tokens in the request.

Require terminal success and fresh clean postflight before each next probe;
only one probe runs at a time. Keep the existing queue at its owned boundary.
Any failure/timeout stops this packet, preserves evidence, and requires the
existing owned-process containment/quiescence procedure before other work.
Do not retry a consumed label or clean another task's outputs.

## What these checks prove

All cases allocate the full dense6x1024 model: **117,453,580 parameters**,
eight heads, MLP ratio4; four CPU ranks and three optimizer steps. Inspect
raw model_config/count, all ranks and checkpoint proof, not only a PASS word.
Reference/default must have identical ownership plan, all twelve rank-step
losses and all62 parameter-state digests. ON must show accumulation2,
batch-rampFalse, six finite forward losses per rank for three updates, all62
parameters changed, cross-rank equality and strict deterministic reload.
No ON-to-control state identity is expected after changing global batch.

The existing harness slices actual input to **2x256 per rank** and appends
CPU/no-compile/3steps/private-output, synthetic tokenizer/data and fake-clock
shims. Thus physical world tokens/update are **2048 for control/default and
4096 for ON**, not131072/262144. The latter are production configuration
accounting: `1x32x1024x4` per microbatch, accumulation1/2. This is full
parameter-size CPU evidence, **not a full hardware update**, native gradient
lowering, kernel correctness, speed, BPB, cold startup or full1800 budget proof.

Retrieve per-probe preflight/launch/process/terminal/g2 JSON, all four raw
rank reports and complete torchrun/g2/driver logs, exact source/request/tool/
prepare/driver hashes, environment, checkpoint URI/SHA, comparator output and
quiescent postflight. Retain failures and kept CPU checkpoints; no cleanup is
authorized. Record the untouched G2 tool's shims explicitly.

After independent numeric acceptance, hand the slot back to Supervisor's
current priorities. Native full training remains a separate pending assignment
and requires genuine fresh6ef7 same-host control results. This packet contains
no quality score, new-base control, ledger rewrite or submission action.
