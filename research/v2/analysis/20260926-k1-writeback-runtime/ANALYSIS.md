# K1 residual copy: no native slot selected from current evidence

Supervisor/Analyst, 26 September 2026 IST. This local analysis uses only the
four SHA-verified runtime traces already retrieved from completed P. No model,
tensor, host, source, queue or evaluator was changed. `analyze.py` gives identical
outputs normally and under `-O`; the exact source AST and input hashes are checked.

The source `OwnedOptimizer.step` ends with send packing, one all-gather, then
the full live-parameter writeback. These map to the last two host foreach-copy
calls around the one owner gather within the captured optimizer annotation.
External IDs correlate them with the runtime's device-labeled Compute spans.
The final writeback has the same NEFF key on all four ranks. `result.json`
retains event indices, IDs, cache keys, timestamps and hashes for reproduction.

| Rank | Pack span ms | Gather span ms | Final full writeback span ms |
|---|---:|---:|---:|
| 0 | 3.210289 | 6.925985 | 6.797362 |
| 1 | 3.004414 | 8.983221 | 7.974865 |
| 2 | 2.823831 | 8.811717 | 7.752924 |
| 3 | 12.615287 | 19.006368 | 8.143385 |

These are **coarse runtime-observed intervals**, not decoded instruction-active
time. They overlap even within a rank; do not sum them as exclusive phase costs.
They include capture/scheduling effects, and the profile has documented rank skew.
Host durations are separate fields, not additional device time.

The proposed `--owned-remote-writeback` would remove only the redundant owner's
copy, around one-quarter of the full-copy bytes, not all writeback, packing or
collective work. The observed complete writeback spans 6.8–8.2 ms. This does not
support a >=5% whole-step gain from that narrower omission. It is not a measured
counterfactual speedup, and it does not prove a universal upper bound under a
different compiler/schedule. No new native comparison or BPB result exists.

Decision: retain the residual-copy proposal as unselected; do not consume a
host slot or call it a demonstrated failure. Literal Muon batching and rank
ownership are already in the base. Keep investigating distinct optimizer costs
only with new causal evidence, while CT/K2 work proceeds. This does not close all
K1 possibilities or relax its >=5% prospective test.

Undo: reopen the isolated proposal with evidence supporting sufficient critical
path cost and the normal source/CPU/native gates. Preserve this profile and its
limits; do not rewrite it as a performance benchmark.
