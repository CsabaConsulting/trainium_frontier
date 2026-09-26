# One owned clean trn22 source checkout — preparation only

Supervisor assignment, 2026-09-26 06:32 UTC. Sole Operator
`/root/trn22_queue_operator` may create exactly one new detached Git worktree
for the selected K2 native-evidence proposal. This is ordinary source
preparation, not native stage/launch, cleanup, AWS action or a gate waiver.

- Source repository: `/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1`.
- New exact path: `/home/ubuntu/trainium_k2_f0f2_trn22_evidence_20260926_r1`.
- Commit: `65155c668efaa215a2136f67617b44ed32d4c9f6`.
- Original host lock: `/home/ubuntu/.cache/trainium_freshstart/device0.lock`,
  device66305/inode832589. Hold it across fresh preflight, creation and postflight.
- Proposal/binding: `research/v2/proposals/20260926-k2-f0f2-native-evidence/`.

Read-only receipt `bad64373f2a2660708ad11cc0008d91324cbb8d2d39cbbb58752d60b5e8a594f`
establishes target absent, original HEAD unchanged, exactly six old untracked
logs preserved, no tracked source edits, and3798 regular blobs/86,770,800 logical
bytes/95,301,632 rounded file bytes. Free17,296,412,672B at06:32 leaves about20MiB
after regular allocation above16GiB, excluding admin overhead. Independent
local inspection of the same commit finds23 distinct tracked parent directories
and no .gitattributes or .gitmodules. This is a sizing estimate, not a capacity
pass or reservation; actual post-creation resources still control.

Immediately before creation require exact host/uid/original lock, ps AND
neuron-ls/fuser idle, STOP/breaker clear, target absent, original HEAD/status
unchanged, and disk>=16GiB plus the rounded file allocation and8MiB allowance;
RAM>=24GiB. Verify no nested .gitattributes, configured external checkout
filters or sparse-checkout behavior would alter the fixed tree. Disable Git
checkout hooks and fsmonitor for this single command. Do not run hooks,
filters, SDK imports, data access or a training process.

Create only a normal detached worktree at the exact commit; no branch, fetch,
pull, reset, ignore-file change, history rewrite or overwrite. A worktree's
normal administrative registration in the existing Git repository is within
scope; none of the original six logs or tracked files may be changed. Use a
bounded command, record stdout/stderr/returncode and all exact argv options.

Afterwards record new HEAD, porcelain status (must be clean), actual filesystem
footprint/free disk/RAM, original unchanged HEAD/status, registry and device
idle/original-lock postflight. Require actual free>=16GiB and RAM>=24GiB;
otherwise mark preparation failed and leave all created files untouched.
No removal, retry, floor reduction or native launch is authorized. On any
error preserve the exact error and partial state for review. Existing K2
packet stays disabled until cache repair, runtime facts and independent review
plus a separate one-use release pass.
