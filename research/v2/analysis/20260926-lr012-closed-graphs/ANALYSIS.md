# Closed training graph accounting — LR012 versus archived exact36 parent

The principal backward and forward compilations dominate recorded compile cost.
There is **no cache-hit evidence in this CSV schema**, even where compile_us=0.
The current training run's observed fifth boundary799.195815s is not a useful
<=180s warm result. This analysis neither verifies the pending warm-path helper
nor approves a launch.

| Rank | Current graph rows / distinct IDs | Current reported compile seconds | Parent reported compile seconds | Current FX / lower seconds |
|---|---:|---:|---:|---:|
| 0 | 5 / 5 | 15.675 | 0 | 0.014583 / 1.720161 |
| 1 | 6 / 6 | 679.583 | 674.330 | 0.015549 / 1.944740 |
| 2 | 5 / 5 | 0 | 15.790 | 0.014299 / 1.591797 |
| 3 | 5 / 5 | 12.248 | 12.708 | 0.014123 / 1.817842 |

Parent has the same5/6/5/5 row/distinct-ID counts. Both have21 total CSV rows and
10 unique full IDs across ranks. Each rank has one666-node forward and one1169-node
backward graph. Other graphs are194-node `inference`-labelled entries, three on
ranks0/2/3 and four onrank1. These are **training metrics**, not the later public
evaluation graphs. The names alone do not prove which optimizer/function each
inference-labelled graph represents; no extra component attribution is invented.

Current rank1 forward records197.130s and backward468.190s, versus parent's
195.929s and463.994s. Together665.320s dominate rank1's679.583s and about94% of
the summed rank-reported compile durations707.506s. Remaining positive durations
are the194-node training graphs, each3.242–7.253s. Summing ranks is bookkeeping,
not elapsed startup time: concurrent work, coordination and unreported waits can
overlap. FX and lowering columns also do not reconcile the whole startup. Do not
subtract these sums from fifth-boundary wall time and label the residual a cause.

## Repeats and cache inference

There are no duplicate full graph IDs within any rank's closed CSV. Across-rank
ID multiplicities are1,1,1,1,1,2,3,3,4,4 in both runs. This is graph identity
overlap across ranks, not measured repeated graph execution. The CSV contains
no execution count, cache-hit/miss, binary-reuse or provenance field. It does not
establish how often a graph ran, that no unrecorded recompilation happened, or
why one rank reports0 while another reports a positive compiler duration.
Do not infer rank2 got a native cache hit from its zero total.

Every corresponding same-rank/same-name full graph ID differs between runs;
the prefix differs, while the suffix and node count match. These opaque key
components are reported without assigning undocumented semantic meaning.
Different IDs are evidence against a claim of identical recorded graph keys,
not proof of a specific compiler-key cause or proof of a native miss. Reused
directory names, matching compiler flags, low/zero compile_us and equal node
counts do not establish cached executable reuse. No host/cache search occurred.

## Source, startup and eval scope

Both source hashes are exact36a862, but current uses matrixLR.012 and parent
LR.015, plus different executed source sandbox paths, launch times and metric
roots. It is not a controlled source-path-only or cache-only comparison.
Current fifth799.1958145109238s versus parent793.9313555649715s differs by
5.2644589459523s; charged training-through-exit is1789.4675970691023s versus
1788.9553176789777s. Those are unchanged harness observations, distinct from
source schedule estimates and CSV compiler-duration fields. No true cold-cache
qualification or warm cache hit follows from either number.

At the current raw observation03:33:12.856758UTC, attached fresh20M evaluation
was still live, terminal_seen=false and final checkpoint/postflight absent.
These four closed **training** CSVs therefore establish no eval compile cost,
final BPB, free-host state or total train+eval turnaround. Later evidence may
complete that run, but it is outside this pinned analysis.

CT implication: preserving the prospective absolute source path is worth testing
as an operational hypothesis, while the observed ~665s principal compilation
cost remains the large recorded expense. Changing small inference-labelled graph
costs alone cannot plausibly explain the desired ~620s startup reduction. The
pending anchored-control run must measure its actual fifth and archived keys;
this analysis supplies no guarantee the proposed path will recover that time.

## Reproduction and authentication

Run `python3 -B research/v2/analysis/20260926-lr012-closed-graphs/analyze.py`.
It only reads immutable local Git objects and prints JSON; no host, SDK, tensor,
source modification, launch or helper verification. Normal/-O outputs agree.

Current `be05918f` and Operator`08e1ecbd` raw bytes are identical, SHA256
`c7cdbdbea99fdb734450db1da3cfb56eaddaea3059e44a5f3b853fd57055227a`.
Parent is the already archived
`research/v2/operator/20260926-trn21-startup-clock-replication/completed/results-bundle.json`
at the same local pin; SHA256
`78aec6e6724b466a7f4ebf9c5bd83846a7c5457b1ecb753eddb50a52b94ab50f`.
All eight CSV byte counts/hashes were recomputed from captured text preserving
CRLF, then the exact nine-column schema was checked. Launch and rehearsal facts
used here were independently size/hash checked too. Full rows, individual file
digests, schema and per-graph comparisons are in results.json.

Two exploratory read-only inspection commands failed before this analysis:
one assumed the parent used base64 and got `KeyError: 'base64'` (actual archive
schema is bytes/sha256/text); another summary one-liner had mismatched brackets
and a SyntaxError. Neither touched data. The final script explicitly supports
the actual current captured-text and parent archived-text schemas, and does
not infer fields absent from them.
