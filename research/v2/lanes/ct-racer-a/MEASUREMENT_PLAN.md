# CT cold-start measurement handoff, not launch authorization

Controlling human gate is canonical
`workflow/COMPILE_TIME_GATE_2026-09-26.md`. Reuse the already-proposed default-OFF
initial-cache candidates and completed independent source/CPU gates; Operator
binds the existing manual route. No new launcher, host command execution, queue
entry or full-run approval is created by this note.

## Single-factor source/flag contrasts

- Base6: exact6ef7 at`a4aa4351789115a2d0aa947d9ca00ff689f9bd0b`, documented
 6x1024/pack32/context1024/global131072/GA1/RecursiveON/denseLR0.015/seed42.
 Candidatebcbd at`b635c8b0cf2a10e1bcdd983e86192672ed6ad56f`, only
 `--initial-cache-refresh` ON. Target true-cold startup<=600seconds.
- Arch8: exact93ae at`1e21c3d797eed764f108c41546b270065ad5650c`, original
 `--unet-8x1024 --attention-layers 0,2,4,6,7`, RecursiveOFF, all other documented
 profile flags unchanged. Candidate3e79 at`b4d83b5db66971f3c348837f04f53cba65aa6d33`,
 only `--initial-cache-refresh` ON. Target<=870seconds; full architecture run
 allowed **only if measured true-cold startup<=900seconds**, otherwise park.

For each authorized cold probe, use a fresh unused unique label, exactly
`--num-steps 5 --max-train-seconds 1800 --no-eval-public`, real data/prepare,
four ranks/LNC2 and identical compiler flags to the original profile. Keep seed42.
These are startup diagnostics, not short quality runs. The existing candidate
proposal's full-run argv is **not** permission to skip this five-step gate.
Do not alter the official five startup boundaries, schedule or timer formula.
Operator chooses host order according to current CT/P/K1/K2 priorities; do not
run every source alternative automatically or duplicate an adequate existing
true-cold control. Original93ae's warm-directory archive is not a true-cold control.

## Required evidence

1. Existing host/role/idle/device/STOP/disk/shared-lock checks and exact source,
   support, prepare, argv, SDK/version and **resolved** compiler argument hashes.
   Only Operator may launch; distinct verifier reviews candidate and receipts.
2. Genuine fresh cache provenance for every SDK-consumed layer: actual fresh
   empty bound roots and reader resolution for NEFF/local lock/HLO/NKI/Inductor/
   compiler/debug caches. Do not clear shared caches or delete protected data.
   A random new directory not actually read by the SDK is not cold proof.
3. Exact launch UTC/monotonic record and original all-rank boundary records0–4.
   Report uncapped launch→latest rank's fifth boundary, source-process→fifth
   boundary if independently available, startup scope and clock-domain method.
   A controller polling bracket stays a bracket, never a fabricated point.
   The source's capped `startup_allowance_est` is separate and cannot substitute
   for measured startup. Preserve actual total/charged-through-exit and reserve.
4. Per-rank graph_stats and first/steady source-to-graph mapping. Report all four
   model forward/backward node counts/IDs, real reported compile times, optimizer
   groups and any unknown timings. Zero metric values are not cache hits. Retain
   compile PID/input/config mapping if the existing observer can collect it
   safely; no new native replay solely to manufacture missing attribution.
5. First-five losses/steps/shape/finite flags, normal exit, checkpoint and budget
   evidence; no post-hoc score. Preserve failures. Short probe stop is intentional
   but does not authorize moving any work outside the original accounting.
6. Observed steady tokens/s from available postcompile steps, with sample count,
   min/median/max and explicit small-N caveat. Five total steps may leave only
   two or three steady observations; that is not a robust whole-step speed gate.
   Obtain robust timing later only if selected under normal source/CPU/native
   gates. A profiler/compile-overhead sample is not steady work.

## Decision interpretation

Count reduction and startup improvement are separate: priming could remove a
graph but not materially reduce startup; unchanged graph count falsifies its
proposed mechanism even if one noisy elapsed time improves. A favorable native
first-five result is not a quality win, submission eligibility or proof of a
single total graph. Native elapsed-based LR/WD may differ when compile duration
changes; no first-five or first50-update identity is promised on hardware.

For base patches above870seconds cold startup, require a measured compile offset
before any full run under the newest instruction. A steady-step slowdown greater
than2% is not automatically acceptable: quantify whether saved **charged** startup
above the allowance offsets it at the1800 budget, preserving checkpoint/exit
reserve. If both runs were already under the allowance, lower raw compile wall
time helps throughput of experiments but does not by itself buy more charged
training time. Numeric BPB, correctness and replication rules remain untouched.

No results are inserted into the ledger, no source is adopted, no full job is
approved here, and no submission branch or Downloads artifact is changed.
