# c5c table rank-hash diagnostic — Implementer handoff

**DISPATCH HOLD.** This folder is an offline, diagnostic-only implementation of
the Researcher proposal at
`research/agentic/proposals/20260925-c5c-table-native50-hash.md`. It is not a
one-shot host request, training authorization, scored result, adoption, or
submission candidate. No host was staged or run by this Implementer.

## Source and profiles

`train.py` is a copy of the immutable scored c5c source at
`submissions/research_v2/train.py` (SHA-256
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`).
The unchanged organizer `prepare.py` is SHA-256
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
This diagnostic source has its own different SHA-256 recorded in
`GATE_CONTRACT.json`; never substitute it for c5c in a scored arm.

The source diff is limited to a diagnostic header, three imports, two
observer functions and one call after `synchronize(device)` on step index 49,
before that step's boundary and checkpoint. The hook is unreachable unless
`--num-steps 50`. It performs no optimizer, parameter, gradient, RNG, data,
forward, collective, schedule or evaluator mutation. It reads BF16 tensors in
bounded CPU chunks and publishes one private, fsynced, no-overwrite JSON
manifest per rank. Raw tensor contents are never printed or saved. The
last-step timing includes the probe and is invalid for throughput decisions.

| Profile | Host | Unique label | Required tables per rank |
|---|---|---|---|
| A1 | trn21 | `fs-20260925-v2-c5c-a1-m64d128-trn21-r1-hash50` | `b5,t5` |
| A2 | trn22 | `fs-20260925-v2-c5c-a2-m64d128-trn22-r1-hash50` | `b1,t1,b5,t5` |

Each table and each RMSProp `exp_avg_sq` moment must be contiguous BF16 with
shape `(524288,128)` and exactly 134,217,728 raw bytes. The observer also
checks its nonmissing same-shape BF16 gradient, finite values, nonzero
weight/moment, optimizer state step exactly 50, RMSProp group membership and
exclusion from the AdamW/Muon owner plan. Profile, rank, complete group and
owner inventory, source hashes and table metadata go in
`table-hash50-rank{0,1,2,3}.json`. Across ranks, every weight hash and every
moment hash for every profile key must match byte-for-byte. A missing rank,
key, step, hash or checkpoint is a failed gate.

The two `EXPERIMENT_PROPOSAL_*.json` files are distinct completed research
proposals from the standard template. `GATE_CONTRACT.json` pins ordered
science argv, diagnostic extensions, source hashes, intended host/checkout,
unique paths, seed, LNC, compiler flags, historical cache roots and immutable
disk floor. The complete `env -i` argv, six live cache identities, native
memory projection, final launcher hash, source-stage receipt and one-shot
authority are deliberately **null** until separately reviewed on each host.
No one should infer live host readiness from this package.

## Local checks (not independent verification)

The Implementer's local check is:

```text
/home/poojith-devan/miniconda3/envs/catq/bin/python -B research/diagnostics/20260925-c5c-table-hash50/test_hash50.py
```

It compares the scored c5c and diagnostic science AST after removing only the
new observer pieces, checks a raw BF16 digest, checks two tiny CPU RMSProp
steps in lockstep, and tests the A1/A2 manifest shape/atomic-no-overwrite
logic using small fakes. This is **narrow** evidence only. It does not run
four-rank exact-m64 CPU G2 or native Neuron, and it cannot prove resource
safety or rank equality. An independent Verifier must examine this work.

## Remaining gates before any host request

1. Independent static review plus exact A1 and A2 four-rank, full-size m64
   CPU lockstep against unmodified c5c on identical synthetic seed-42 inputs
   and clock. Compare steps 0/1 losses, all parameter/table/gradient/moment
   digests, state counters and owner plans. A workstation's reduced test is
   not a substitute.
2. Each host separately passes exact c5c m64/d128 two-step G2 RMS trace and
   an **unmodified c5c** 30-step Neuron smoke with strict model-only load and
   deterministic synthetic causality/batch-independence checks. Neither
   profile proves the other.
3. A separate Verifier reviews the host-specific clean source stage, entire
   effective environment, six authenticated warm-cache directories,
   private trusted step-log parent, live boundary observer, root-free
   monitoring and owned-guard stop interlock. Recompute a conservative
   profile-specific HBM/root growth projection; never lower the 6.5-GiB
   operational disk floor. A null projection remains HOLD.
4. Only after a frozen host-specific request passes independent review may
   the Supervisor grant one-attempt native-50 authority. The Operator must
   perform fresh hostname/idle/Neuron/lock/disk/source/cache/label preflight
   and preserve one run per host. The first two losses must match the
   same-host unmodified smoke; observe boundaries 0..49 exactly once.
5. Retrieve and hash all four rank manifests, raw log, checkpoint, boundary
   and guard/monitor/postflight receipts. Independently compare full BF16
   weight and moment hashes for **every** key; strict-load the checkpoint in
   a fresh process and perform synthetic causality/batch-independence only.
   No public/private validation eval, BPB claim, `decide.py` update,
   submission, or cleanup is part of this diagnostic.

Failure consumes the unique label and requires a diagnosed new request and
label; never retry in place. Later scored A1/A2 runs require their own
unmodified-c5c source, valid same-host c5c A0 public20M control and separate
one-shot authorization.
