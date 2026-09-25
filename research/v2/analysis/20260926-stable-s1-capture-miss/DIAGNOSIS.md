# Stable-source S1: evidence-capture miss, cause unresolved

Read-only Analyst diagnosis, 2026-09-25 23:05–23:10 UTC. No training/tool source
change, host call, retry, staging, cache scan or independent acceptance. Raw
archive **190ccf4a40d48bf318e2ed231d0316918b6182e9**:
`research/v2/operator/20260926-trn22-stable-source-pair/completed/`.
Local hashing authenticates all26 raw files/90,280 bytes in results-bundle.json,
SHA `6424cfcfbb311a49a6e39080a88afa6d05b027ce951a09a506b2547e49742b0c`.
FACTS.json records the extracted facts; this is not a Verifier verdict.

## What actually failed

S1's unchanged 3e79 training child exited0 after five boundaries, fifth
**845.5936225099722s**, process wall849.8144521720242s and charged-through-exit
4.220829662051983s. Checkpoint SHA is
`a4e11a3d781dd85667e69abbc5ceba9a23cbf57eeb5762a38dff5c3984ba0a07`.
The pair then refused at23:01:52.975813UTC with
`MATCHED_PRINCIPAL_COMPILER_BYTES_MISSING`; **S2 never launched**. Preserve that
failed pair. Successful training is not successful compiler evidence or reuse.

Exact packet `research/v2/proposals/20260926-architecture-stable-source-pair/`:

- host_driver.py:84 waits synchronously for harness exit; lines88–95 check
  post-source identity, post-idle state and five-step mechanical validity.
- Lines96–98 call capture_inputs, save S1-compiler-evidence.json, then refuse
  if exact_input_bytes_complete is false. No S2 fallback or automatic retry.
- identity_checks.py:116–129 reads all four closed graph CSVs and collects
  the first64hex graph-key component for model_default forward/backward.
- Lines130–141 inspect only `debug/neuronx-cc/*/module.mlir`, <=128 immediate
  children. Missing files are skipped at139; hash nonmatches are skipped at141.
  Neither path names, nonmatching hashes nor directory inventory are retained.
- Lines150–154 report only matched input files plus missing keys. Thus an
  empty match set cannot distinguish no files from files with different bytes.

S1-compiler-evidence.json contains **zero matched raw inputs** and two missing
keys, present identically in every rank's principal CSV rows:

| Graph | Key prefix | Nodes | Positive compile time |
|---|---|---:|---:|
| forward | aa53f72d8a4c45dbc0ba1784f3536f8803525417f6de4d3f8fb82332aa2ef58e |666|201.411s, rank1|
| backward | d81a59fd782a752806c228735a21928133bf2d21c5d59344e2e0e94fb3db8c25 |1169|470.446s, rank1|

Other ranks report zero for these entries; CSV row counts are5/6/5/5. Existing
archived SDK metrics source (`20260925-trn22-l2-93ae-cpu-r1/native/metrics-source-readonly.json`,
SHA c1702b4b) returns zero when its cache-entry lookup finds nothing and exports
CSV at process exit. Zero is not a hit. Positive times and live compiler argv
establish real compilation here; S1 was intentionally cold.

## Layout versus disappearance: evidence does not decide

`live-compiler-observations.json` (SHA
8611f7354a35f42e1a6a44a5438996b07c8c2af7cf9dd5e6dfa24c72d07054e0)
records actual compiler/walrus processes at22:52:26 and22:57:31, and another at
23:01:43. Their absolute logfile/NEFF output paths use the **same** expected
S1 `train/caches/debug/neuronx-cc/` prefix. Three recorded directory names are
96b199d519a331288b0a67881f092b12,
ea30e57eba554980a991fca1b737d5ac, and
860bfc7a33ecceebfbabbde99c9f2e27. Directory names are not assumed graph keys;
which graph each represents is not authenticated by these argv snapshots.

The helper reached its return without a missing-directory, symlink, limit or
file-size exception. This shows its debug root could be inspected, not that
the expected module.mlir files survived. A completely unrelated debug prefix
is less consistent with the live argv, but relative compiler cwd was not
captured. No live input bytes/stat/hash, post-exit directory inventory or
nonmatching hashes exist in this archive. Consequently **post-exit cleanup,
different file layout, transformed/replaced bytes and a graph-key/hash mapping
difference remain unresolved alternatives**, not ranked causal findings.

Earlier full-run live capture authenticated one backward module's SHA equal to
its graph-key prefix (`20260926-trn22-arch3e79-full/live-compiler-201309/exact-byte-bundle.json`).
It was captured while compiling, not after exit. That supports the possibility
of exact matching, not this packet's unproved assumption that all principal
inputs remain available at the post-exit location. The test fixtures retained
their files and cannot settle native file lifetime. No compiler-input
normalization, decoder output or hash substitution is justified.

## Smallest next observation — proposal only, no host request made

Prefer **one bounded read-only capture of S1's existing named files**, during
a separately assigned idle window; do not spend another cold run yet. Its
explicit prefix is:

`/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-stablepath-trn22-r1-pair/s1/train/caches/debug/neuronx-cc/`

For only the three named children above, record no-follow existence and
pre/post inode/size/timestamps for module.mlir, module.neff and log-neuron-cc.txt;
retain each raw MLIR <=4MiB, stream-hash each NEFF <=16MiB and capture bounded
existing log head/tail <=64KiB each. Record every absence/oversize/hash nonmatch
explicitly. If needed list <=32 immediate names per named child, never recurse
or search other cache roots. Stop at30s or60MiB total reads; absent/oversized
files remain UNKNOWN, not permission to widen. At most16MiB artifact bytes;
NEFF contents need not be copied. Reject symlinks/races, never rewrite files.

This can establish whether existing bytes now map to a principal key, and
pair a compiler input with the sibling output named by actual compiler argv.
If files are now absent it establishes only absence **now**, not when the
helper ran. Compiler-output NEFF identity is not persistent-cache identity.
Only if existing logs identify an exact persistent entry should that named
entry be examined under the same remaining read bound. Do not guess NEFF
filenames from graph keys: the prior sampled cache names use a different form.

**Actual reuse requires an observed read/mmap or an explicit SDK reuse record
for a pre-existing authenticated NEFF**, tied to the principal key and owning
worker, plus write/compile observations. Mere existence, unchanged mtime/atime,
equal hashes or zero timings cannot provide that. Existing logs may not expose
this mapping. If they do not, report it absent and stop; no generic hit claim.

A later new-label single warm five-step diagnostic is defensible only after
the capture/mapping route is known and separately reviewed: same immutable
source pathname and exact bytes/science, S1's retained private caches, distinct
outputs, original lock/guards/budget, no auto-full/eval. Capture exact principal
inputs while their actual compiler PIDs are alive, before they can disappear,
and associate them after closed CSV export. Observe named-cache read/write
events through an already-available bounded read-only facility or explicit
SDK records; availability/overhead is unproven here. Without that facility,
do not run a blind repeat just to obtain another zero timer. This would be a
new operational diagnostic, **not resuming failed S2 or proving sole-path
causality**. Printed first-five loss/LR parity has rounding and wallclock-LR
confounds. Current full75b9 and K2/clock qualifications retain priority.
