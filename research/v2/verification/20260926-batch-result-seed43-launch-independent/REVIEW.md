# Batch timing result and seed43 launch authentication

Verdict: **batch result is valid and BELOW_THRESHOLD; seed43 actual launch
is authenticated, not a completed-run or quality verdict**. No new host,
source, threshold, ledger or queue action is authorized/performed here.

Reviewed raw commits: batch `6c00343d4938349c55c04e9b0e4a55a429983d1d`,
seed launch `746b53df4745e5521868034128ffc3d296cba9b8`. Own stdlib-only
`test_receipts.py` passes71 checks normally and under `-O`. It independently
parses the two raw speed windows and recomputes Decimal threshold arithmetic,
verifies hashes/argv/process identities/timing, replays the archived analyzer,
and confirms its missing-boundary/budget/nonzero-exit refusals. Full input
SHA256 manifest and arithmetic are in `RESULT.json`.

## Batch262144: valid narrow miss, not a win or quality loss

All14 retrieved files named in the remote retrieval manifest match both
SHA256 and byte count. Exact source is a489f7 from4e361d15, unchanged prepare
5807c46e and rehearsal7a55. Retrieved package launch-command/provenance,
actual REHEARSAL_COMMAND and reviewed training argv agree. The native log
confirms world4/LNC2/BF16, pack32/context1024 and accumulation2. All105 raw
STEP_LOG entries and observer boundaries are exactly0..104; all selected
logged losses are finite. Target/actual step counts are105.

The prospective proposal in a4b4838f is byte-identical to the proposal used
for analysis. Its matched printed window is10,20,...,100 and both runs have
LR multiplier1 there. The control log matches its predeclared SHA256
`074fa6cc4db14ebda63b21f07450c292d14ceebc4a6ddf98f4dfdc0e144937b1`.

| Precommitted printed-median statistic | Tokens/s |
|---|---:|
| Same-host seed42 control |268831|
| Batch candidate |282266.5|
| Required control ×1.05 |282272.55|
| Shortfall |6.05|

Relative gain is **4.99774951549...%**, strictly below5%. Preserve
`SPEED_SCREEN_BELOW_THRESHOLD`; rounding to5.00% is not a PASS. The miss is
only0.0022504845 percentage points, so this is a borderline screen outcome,
not evidence of a robust scientific difference. No threshold relaxation,
automatic full run, repeat or quality adoption is inferred.

Independent observer arithmetic:

- Uncapped observed fifth boundary:794.928466952988 seconds; cap900 does
  not bind. Source estimate792.7 is a separate clock/rounding quantity.
- Training-child process span:893.1827079360373 seconds. Subtracting the
  observed allowance gives98.25424098304939 seconds charged through exit,
  strictly within1800. Source training93.1/source charged93.5 are not used
  as the external clock. The existing50ms observation precision is retained;
  the budget margin is large and no violation is evidenced.
- Boundaries104−4 span93.12771701195743 seconds for100 ×262144 tokens,
  giving281488.70004656207 tokens/s. This is a sanity diagnostic only, not
  the same estimator as the historical printed median, and is not a second
  acceptance/veto threshold. The archived analyzer exactly reproduces it.
- Training/harness/outer-flock exits are0. Checkpoint exists at harness exit,
 469834849 bytes, SHA256
  `11db02c3b83cf2c8502c088b54046f9d52567c1f99b002bcf011e7d6d673e5f6`.
  This is the harness's file hash, not a fresh independent checkpoint load or
  a claim that checkpoint bytes were downloaded in this audit.

Outer exit was20:21:47.545540UTC. At20:22:34.690729UTC the retrieved ps/device
evidence was quiescent and a lock contender succeeded. Empty fresh-cache
map/intended warm roots do not prove cache hits or cold startup. No evaluation
ran; `quality_evidence:false`, `no_eval:true`, and no automatic full remain.
The rank0 STEP_LOG observer and finite sampled window limitations disclosed
before the run are unchanged; do not claim a full-budget BPB or throughput win.

## Seed43: exact subsequent launch, completion pending

The seed run uses reviewed90e942ea binding52e11b65, proposal d71ed128 and
driverabc5ca59, with exact adopted source6ef7/a4aa4351, unchanged prepare5807
and rehearsal7a55. Staged hashes match the reviewed packet. The package
launch-command hash55840979 is independently reproduced from the exact
reviewed argv plus newline. Actual outer command/environment and every worker
argv/cwd match that binding: only seed43, no batch262144 flag or source swap.

Fresh in-lock ps AND neuron-ls passed20:23:58.983140–20:23:59.044191UTC;
training phase began20:23:59.044411UTC, after batch exit and quiescence.
The same persistent lock device66305/inode596823 appears in both launches
and final batch retrieval. Seed's live20:24:14.478240UTC contender returned1,
proving the lock remained contended at that observation. Authenticated outer
flock PID3015274 owns driver3015275; torchrun3015306 owns workers3015321–3015324,
with ranks0..3 and the same four Neuron device-holder PIDs. The old queue PID
and command matches are absent. Earlier staging20:21:16 correctly records
the active batch and explicitly claims no launch; it is not an idle receipt.

All four actually retrieved hashed seed launch files match remote size/hash.
The earlier preflight object embedded in staging-receipt reconstructs exactly
to8035 bytes/SHA b78e66b5 with documented JSON indentation/newline. The two
empty outer/dispatch logs are remote hash/size reports only and were not
retrieved in746b53df; they are not represented as locally authenticated files.
The live-process receipt itself is pinned in this audit's input manifest.

This proves actual guarded launch/live workers at the recorded instant, not
current liveness, successful steps, final checkpoint, budget completion,
fresh20M result or future eval guard success. Those await the normal result
archive. Seed43 is a variance diagnostic; submission seed42 remains unchanged.
