# Exact 93ae L2 architecture: pending CPU gates and full-run packet

Status: **NOT READY — exact-size CPU G2 and OFF identity have not run.**
This packet is prepared for a distinct Operator to stage at the next Supervisor-
selected free host slot. It does not authorize staging, CPU execution, queue
changes or native launch. Both current Recursive jobs and their fallback tails
remain untouched. No extra human approval ceremony is introduced: existing
standing development authority still applies subject to the outstanding gates.

The candidate is frozen at commit
`1e21c3d797eed764f108c41546b270065ad5650c`, source SHA-256
`93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5`.
There are **no train.py edits in this packet**. The independent original
architecture review and narrow timer-port review are referenced by exact Git
commit in `sources.json` and authenticated by `packet-check.json`.

Each host directory contains a complete, uniquely labelled pending proposal,
four serial exact-source CPU probe commands, and a disabled full-run command
plus fresh evaluation specification. These are alternative host bindings:
the Supervisor selects the first compatible free slot; do not automatically run
both architecture jobs. A cross-host confirmation follows the current decision
rules or an explicit Supervisor stack assignment.

| File | Purpose |
| --- | --- |
| `sources.json` | Exact candidate, c5c, 6ef7, G2-tool and unchanged prepare Git objects/SHA pins |
| `trn21/EXPERIMENT_PROPOSAL.json`, `trn22/EXPERIMENT_PROPOSAL.json` | Host-specific architecture-only proposal and actual same-host c5c control |
| `*/cpu-g2-plan.json` | Full-parameter ON/OFF/reference probes, exact commands and comparison rules |
| `*/native-job.disabled.json` | Full-chip 1800-second training and separate public-20M/context-1024 evaluation |
| `OPERATOR_RUNBOOK.md` | Staging, CPU evidence, queue handoff and native artifact checklist |
| `packet-check.json` | Static source/argv/label/manifest verification; explicitly not a G2 result |

## Exact architecture and comparison

The arm uses depth 8, width 1024, attention in layers 0,2,4,6,7, MLP-only
layers 1,3,5 and four learned zero-initialized scalar U-Net skips. The native
and G2 ON argv are identical, including `--unet-8x1024 --attention-layers
0,2,4,6,7`, `--no-ngram-ve`, context 1024, pack 32, matrix LR 0.015,
seed 42, requested 1800 seconds and LNC2/four ranks. This is the explicitly
requested architecture package, not an isolated skip experiment.

CPU ON allocates the actual 8x1024 model and optimizer states. Only the
training forward micro-batch is sliced to synthetic 2x256 by the unchanged
G2 harness. OFF uses the same candidate bytes with all new architecture flags
omitted, recovering the dense 6x1024 model. Its three steps must match both
exact c5c and exact 6ef7 with Recursive **OFF** in the same CPU environment:
all 12 rank/step losses, all 62 final parameter digests and the ownership plan.
The four probes run serially with distinct labels and retained raw reports.
The prior accepted full-size dense reference receipts are background evidence;
this packet calls for fresh matched references to keep the new identity proof
self-contained on either host.

The existing science controls are c5c A0: trn21 label
`fs-20260925-v2-c5c-a0-control-trn21-r1`, BPB 0.9914739939934344,
3675 steps; trn22 label `fs-20260925-v2-c5c-a0-control-trn22-r10`,
BPB 0.991513958330512, 3633 steps. Their independent reviews and ledger
records are authenticated. The separately reviewed timer reset is an explicit
operational source difference, with unchanged charging and model arithmetic.
Refresh the dense control if the existing age/drift/base policy requires it.
The currently running Recursive-ON results cannot serve as architecture-only
controls; stacking Recursive or FP8 requires another declared source/package.

No new decision exemption is introduced. This arm is `architecture_arm`,
with the existing step-window and BPB rules. A source/CPU PASS is not a
native result, and no gain is predicted from the public recipe.

## CPU memory and time evidence

The previous exact-size trn21 dense Recursive G2 used this same harness,
four ranks, three steps and synthetic 2x256 micro-batches. It completed in
20.0 seconds on Torch 2.11.0+cpu with **131,432,075,264 bytes** of
MemAvailable at preflight. Its request required at least **64 GiB** available.
Evidence: canonical e9109dec
`research/v2/operator/20260925-trn21-l3-timer-dense-g2-r1/`.
That observation is historical availability, not a current measurement or
peak-RAM report. The logs do not establish L2 peak RAM or duration.

Static arithmetic for the pinned no-table architecture gives 113,256,724
parameters: two vocabulary matrices, eight MLP pairs, five attention
projection sets, one VE table, five two-tap convolutions, one VE gate,
sixteen block scalars and four U-Net scalars. The same formula gives
117,453,580 for dense 6x1024 and matches the archived dense ownership receipt.
The arithmetic estimate is 453,026,896 bytes for FP32 parameters per rank;
four copies of parameters plus gradients alone require 3,624,215,168 bytes.
This is a **lower bound, not peak RAM**: optimizer and ownership buffers,
activations, cached weights, Python/runtime memory, checkpoint reload and
the existing evaluation masks add memory. Exact L2 CPU peak and HBM peak
remain unknown.

Reuse the conservative 64-GiB MemAvailable floor on the selected idle host,
plus the immutable 6.5-GiB disk floor and projected artifact headroom.
Record fresh memory before each serial probe. Do not run the full-size
probe on the 14-GiB laptop, concurrently with training/evaluation, or with
any other G2. No RAM limit was weakened and no full-size model was allocated
while preparing this packet.

## Queue compatibility and completion

At canonical e9109dec the installed `host_queue.py` accepts only c5c,
f9b6 and 6ef7 source hashes. It rejects 93ae in both manifest validation and
source selection. Consequently these disabled JSON files are not installable
queue manifests. After independent exact-size gate acceptance, choose either:

- separately reviewed 93ae source-registry support in that existing runner; or
- the already authorized manual train-to-fresh-eval route, pausing the existing
  queue between jobs first so only one launcher can claim the host.

No runner changes, source-registry mutation, fallback replacement or host
action were performed here. Only an unstarted fallback may be replaced.
Consumed labels, active jobs and protected artifacts must remain intact.

The candidate becomes ready only after exact-source ON G2, both OFF identity
comparisons, independent numeric acceptance and a compatible launch route are
recorded. A distinct Operator then applies fresh machine/STOP/lock/source
checks, records the exact environment and launches one unique full-chip job.
Warm research caches are retained without claiming measured cache hits.
The official 900-second startup cap, 30-second in-file launch margin,
15-second checkpoint reserve and 1800-second budget are unchanged.

Static check command:

```text
python3 research/v2/proposals/20260925-l2-93ae-exact-g2-native/check_packet.py
```

It imports no model, allocates no tensors and runs no host command. It verifies
all source Git-object hashes, all 12 unique labels against the frozen canonical
queue/ledger, full-chip argv, source/host/proposal bindings, full-size ON and
dense OFF scopes, actual historical controls, disabled status and memory
arithmetic. Its PASS is deliberately `PASS_STATIC_PACKET_ONLY`.
