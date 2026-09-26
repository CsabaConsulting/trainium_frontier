# Symmetric K2 exactness failure — no retry

One assignment6fd5f333 consumed exact710ea677 packet/source9ce2e6a0.
Actual dispatch00:08:48.310143UTC; under-lock native child launch00:08:48.428422.
Original inode596823/device66305, fresh ps/neuron-ls and private empty six
caches passed. Idle gap after P r2 was189.695133s. Full36 on trn22 unchanged.

Child ended00:09:18.565322UTC after30.136715991s, exit1; driver terminal
`DIAGNOSTIC_CHILD_FAILED`00:09:18.565542. All four actual worker receipts
3101427/28/29/30 report `RuntimeError: EXACTNESS_FALSIFIED: unique/fp32`.
Each contains13/20 cases, with the last case failing its
`compiled_enclosing_autograd_backward` byte comparison. Actual gradient is
all zeros (nonzero0), versus expected32709 nonzeros, max_abs8.0; direct and
eager FP32 paths on that case are exact. Earlier BF16 cases passed. This is
a numerical correctness failure, not a compilation error or complete native
qualification. No timing/speedup/adoption claim, threshold change or retry.

Raw31 files/556014bytes include full diagnostic log, all four rank checker,
wrapper-exit and environment receipts, four graph CSVs, source/launch/cache/
guard identities and terminal. Base64 byte lengths and SHA256 all verified.

Explicit postflight covers actual dispatch3101403/flock3101404/driver3101405,
torchrun3101422 from its raw error and all four recorded workers. The short
run ended before the first remote process snapshot: timeout and short-lived
compiler PIDs were not observed and are not invented. Full current unfiltered
process identities plus observed PID/PPID/PGID/SID and compiler/name scans,
device idle and original unchanged/free lock are recorded separately. No
unrelated process kill or breaker reset was performed. Required future launch
checks remain fresh; the independent bcbd fallback awaits its state scope.
