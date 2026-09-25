# Exact 3e79 cross-host divergence: evidence and remaining uncertainty

Role: Analyst, local read-only investigation; note only. No tensor/model import,
dataset-file access, host call, external research, source edit or launch. The
startup-adjusted source and all disabled packets remain frozen. This supplements
29bbb1cf; it does not turn the early schedule mismatch into a complete causal
explanation, reverse a review, or adopt the architecture.

## Result and common contract

The retrieved fresh-public-20M logs record trn21 **0.9931291951444726** and trn22
**0.9863403052402583**, a cross-host spread of **0.0067888899042143**. Both report
20,971,520 tokens and 83,462,186 bytes. The trn21 value is worse than its c5c
control, so it does not confirm the first architecture result. Result acceptance
and selector use remain Supervisor/Verifier responsibilities, not this note.

The reported later trn22 LR0.012 result **0.988360** is about **+0.002020** versus
same-host 0.986340. This note treats that as a reported value with raw acceptance
pending the Supervisor, not as an accepted LR conclusion or a new control.

Both training-launch receipts authenticate source
`3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c`, prepare
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb` and harness
`7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b`.
The actual rehearsal commands agree scientifically: seed42, LR0.015,
8x1024/113,256,724 parameters, attention0,2,4,6,7, UNet, initial cache refresh,
context1024, pack32, four ranks/LNC2, batch131072/accumulation1, MAX1800. Neither
has Recursive, epoch shuffle, batch ramp, FP8 or tables enabled. Printed optimizer
ownership counts and its three batched Muon shapes also agree. Distinct hosts,
paths, cache histories, timestamps and output labels are expected differences.

## What is proved versus unknown

| Axis | Supported by local evidence | Not established |
| --- | --- | --- |
| Training selection/order | Same cache pathname and exact prepare; default train split explicitly selects shards0–4 in that order, not every locally present shard. Same rank-strided row-group algorithm, packed loader and no epoch shuffle. Both logs show epoch1 at2370 and epoch2 at2380. | Actual file inventory/bytes/row-group metadata at each run, initial/per-step token hashes, and identical rank-local batches. Same path on different machines is not the same file. |
| Tokenizer | Exact prepare verifies fixed tokenizer.pkl and token_bytes SHA; recorded clean launch environments contain no tokenizer-hash/path/URL overrides. Training reaches completion through ensure_tokenizer(build_if_missing=False). | A separate observed tokenizer-file digest receipt at launch and an independently authenticated per-run tiktoken package identity. Runtime completion supports the enforced check; it is not a newly retrieved file hash. |
| Initialization/RNG | Same seed42 and model config; torch.manual_seed runs before CPU model construction and init_weights, then model moves to Neuron. Training-adapter installation asserts CPU RNG state unchanged. No init-from argv. | Native-run initial parameter/RNG digests, exact installed CPU initialization-library build equality, or every rank's initial values. Rounded initial losses are insufficient. |
| Precision/runtime | Both log Neuron BF16, direct compiled SDPA, BF16 norm output, local conv and native MLP adapter; same compiler flags optlevel1/matmult BF16, same world/ownership. | All effective kernel reductions, runtime/library binary hashes, NEFF equality, native intermediate/gradient equality, or a cross-host deterministic numerical guarantee. |
| Compilation/cache | Each run's four ranks share its principal graph keys and666/1169 forward/backward nodes. Keys differ across hosts and source paths/cache histories differ. Positive compile measurements exist on both. | Path-only causality, identical lowered arithmetic, or the existence/magnitude of a numerical effect. Equal node counts are not graph equivalence; zero compile_us is not cache-hit proof. |
| Optimizer schedule | Rank0's early LR multiplier differs; later elapsed schedules also drift slightly. WD is tied to that same elapsed progress. | Full-precision every-step/per-owner LR, WD and momentum history; exact native update equality; how much final BPB spread any difference caused. |

### Data and tokenizer detail

Pinned prepare.py lines47–49 bind both runs to
`/home/ubuntu/.cache/neuron_competition_r1/{data,tokenizer}`. Lines54 and409–432
select the five exact training files, enumerate each file's row groups in order,
then give rank r every fourth group. Lines435–444 repeat the same ordering each
epoch. Lines456–518 best-fit pack tokenized documents deterministically by their
available order; tokenizer threads default4. Train blob `b4d83b5d`, lines3941–3959,
selects the ordinary loader and reshapes its packed rows; extra shards would not
be consumed by this default path. No actual training file was opened for this
analysis, and no shard manifest was found in the two inspected run archives.
The downloader's existing-file path (prepare.py131–136) does not authenticate
training-file content. Consequently identical code and successful loading do
not establish identical training shard bytes.

Tokenizer enforcement is stronger: prepare.py64–70 pins tokenizer.pkl
`aacd6c6a838eaec2c662a7d0cba0b5412f930a6ddee95b9ecb2c8fac53f3dafc` and token_bytes
`3de03cf6f8e16a9397fecd0cebd04f435f11994b215a5c785787fcc15adb0e15`;
lines205–225 and289–298 verify those artifacts, and368–396 does not permit a
self-built fallback when build_if_missing=False. Successful execution with the
recorded clean environment is evidence against a simple tokenizer-artifact
mismatch. It does not prove training-file equality or tokenizer-library equality.

### Initialization, rank-local schedules, and what the loss lines mean

In exact source `b4d83b5d:workflow/proposals/research-v2/train.py`, lines509–527
seed before runtime setup;3893–3907 build/init on CPU then transfer. The adapter's
4924–4933 CPU RNG assertion excludes that specific installer consuming RNG.
It does not record the RNG state or initial weights for either native run.

**The printed loss is rank0's debiased exponential moving average, not raw
per-step loss or a globally averaged loss** (4103 and4126–4134). The first two
matching four-decimal lines therefore do not establish initial/batch/update
identity. The loss values below are deliberately labeled as printed EMA.

| Printed step | trn21 LR / EMA | trn22 LR / EMA |
| --- | --- | --- |
| 0 | 1.000 /9.0101 | 1.000 /9.0101 |
| 1 | 0.767 /8.7779 | 0.760 /8.7779 |
| 2 | 0.767 /8.4646 | 0.760 /8.4665 |
| 4 | 0.766 /8.2456 | 0.759 /8.2179 |
| 50 | 1.000 /5.6346 | 1.000 /5.4158 |
| 100 | 1.000 /4.7356 | 1.000 /4.5125 |
| 500 | 1.000 /3.3726 | 1.000 /3.3589 |
| 1000 | 0.973 /3.1517 | 0.972 /3.1322 |
| 2000 | 0.626 /2.9594 | 0.625 /2.9411 |
| 3000 | 0.279 /2.8168 | 0.279 /2.7981 |

These are observed trajectory differences, not estimates of causal effect.
Rank0 step0 is779.52s on trn21 versus790.37s on trn22. The scheduler samples
elapsed before that step; excluded startup is removed only after step4
(source4012,4070–4079,4104–4111). Thus step1–4 use a different LR multiplier and
Muon decay. In the cooldown segment,
`WD = 0.2 * 0.75 * (LR_multiplier - 0.05) / 0.95`.
The rounded step1 multipliers imply rank0 WD in approximately
`[0.1131316,0.1132895]` versus `[0.1120263,0.1121842]` respectively. These are
**source-derived rounding bounds**, not logged WD measurements. The code has no
LR quantization (LRM_QUANTUM=0); matching three-decimal LR lines can conceal
different exact multipliers, and LR=1 does not imply equal WD.

Later wall-clock progress differs too: at step1000 remaining_time is1311s versus
1309s, at3000 it is326s versus324s. Exact every-step progress cannot be reconstructed
from rounded ten-step logs or post-step boundary times. The code computes elapsed
separately in each rank; it does not broadcast the LR/WD progress. OwnedOptimizer
copies each rank's source-group scalars to its local owned groups, updates them,
then gathers the parameters (4653–4660,4682–4703). Only rank0's LR is printed.
Per-owner schedule differences are therefore **unmeasured**, not shown absent by
equal rank-consistency checks. The new startup diagnostic is not a guarantee of
native trajectory identity after the fifth boundary.

Total completed updates differ by only two (3631 versus3629); both report
training_seconds1784.7. Actual charged-through-exit receipts are1789.216414s and
1789.208484s, with uncapped fifth boundaries805.080502s and814.680998s. These are
accounting observations, not grounds for a step-window verdict or causal claim.

### Compiler evidence does not prove arithmetic equivalence

Trn21 raw graph_stats contains principal prefixes `bf93fa5b…` / `57a8738f…`;
trn22's authenticated cache investigation contains `53a92b04…` / `97682f6d…`.
Within each run all four ranks agree on the major pair. Counts are666 forward,
1169 backward and per-rank total rows5,6,5,5 in rank order0,1,2,3. Positive major
compile durations are196.992/464.356s on trn21 rank3 and202.390s rank2 /
477.604s rank0 on trn22. These are individual compiler metrics, not an additive
wall-time model across concurrent ranks.

The observed command/profile is the same; trn22's compiler prefix identifies
2.26.6360.0+6f180f47. This analysis does not authenticate a matching installed
compiler/runtime/library manifest for both particular runs. The retained trn22
binary MLIR hashes to its backward graph key and embeds its absolute source path;
there is no matching authenticated trn21 MLIR/NEFF comparison here. Different
keys can reflect metadata/path churn, but neither that explanation alone nor a
numerical difference is established. Both cache histories are nominal reuse,
not two fresh-cold replications; this primarily demonstrates a startup
comparability limitation. No cache hit is inferred from zero-valued metrics.

## One optional bounded read-only provenance check, not an assignment

If the Supervisor wants to discriminate the remaining data-artifact hypothesis,
the sole Operator could perform **one paired training-shard manifest check** at
an idle boundary on both hosts, under the original guard/lock rules. Whitelist
only `/home/ubuntu/.cache/neuron_competition_r1/data/shard_00000.parquet` through
`shard_00004.parquet`. Record basename, resolved path/no-symlink status, size,
mtime_ns, and complete SHA256; use streamed hashing with at most64KiB buffer,
one process, total read cap4GiB and60s timeout per host. Stat before/after each
file and mark any mutation, oversize, timeout or incomplete hash **UNKNOWN**.
Do not follow other paths, parse rows, import Torch/prepare/PyArrow, output
contents, touch validation, download/repair files, or launch training.

This is proposed only, not performed or scripted here. Full hash disagreement
would establish differing current training artifacts, a substantive alternative
to the schedule-only explanation. Matching complete hashes would exclude that
current artifact mismatch, not prove historical immutability, identical token
batches, library behavior, native arithmetic, or the cause of the BPB spread.
If the bounded check is incomplete, retain that uncertainty; no automatic larger
scan or additional host checks are proposed.

## Evidence pointers / decision

- `research/v2/operator/20260926-trn21-arch3e79-replicate/` and
  `research/v2/operator/20260926-trn22-arch3e79-full/`: completed-training launch,
  log, train/rehearsal and completed-eval/fresh-public20m records. Training log
  SHAs respectively `98a201bd36f17f03cf791dc39040873c9dc50d9538cabb01c696c015f11a8ff7`
  and `19a0b5a906c04cd799d7393fbe51e6f6dcd23a937be30226b12727af1360016e`.
- Trn21 `completed-training/graph-stats-raw.json` contains exact per-rank CSVs.
  Trn22 `research/v2/operator/20260926-architecture-cache-readonly-2148/`
  (0e783999), especially derived-graph-keys.json and authenticated-comparison.json,
  authenticates the companion graph evidence and states its limitations.
- Exact source Git blob `b4d83b5db66971f3c348837f04f53cba65aa6d33`, and pinned
  prepare source cited above. No live data or environment inferred from local
  workstation packages.

2026-09-26 analysis decision: retain the demonstrated early LR/WD confounder as
one candidate explanation, and retain data-content, initialization/package,
rank-local timing and native arithmetic unknowns explicitly. Do not claim any
one cause explains the0.006789 spread. No new science or throughput patch is
proposed. Undo: disregard this advisory note; all source, queues, gates, raw
results and frozen packets are unchanged.
