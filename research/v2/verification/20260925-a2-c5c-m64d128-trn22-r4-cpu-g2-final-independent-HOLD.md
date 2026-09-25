# Independent Verifier — trn22 A2 exact-size CPU G2 r4 final-pending package

**Verdict: HOLD for final PASS, assignment, and any host attempt.** This is an
offline static review of Implementer commit
`164cfcd85fd1ae6b5751cf1e07b905927d7c0973`. The package is disabled,
with `fresh_host_values: null`, all request authority flags false, and no
fixed-path final review or Supervisor assignment. I did not contact a host,
create an authority file or marker, execute CPU G2, train, evaluate, alter
model/scorer bytes, delete anything, or touch AWS. This receipt is **not** an
independent final PASS and may not be used for dispatch.

## Exact reviewed bytes

| File under `research/v2/authorizations/trn22-c5c-a2-m64d128-cpu-g2-final-pending/` | SHA-256 |
|---|---|
| `CPU_G2_REQUEST.FINAL_PENDING_REVIEW.json` | `7e97d8fbe91cd7eabed65c4ef3e5b85a7f1c9f3dbb143147530e9dc6c38b9b47` |
| `INVOCATION_BINDING.FINAL_PENDING_REVIEW.json` | `4d719da8ca95f6d70cd6d22f5f35d3cf4bb23ed2b22047560109b5905df59738` |
| `RUNBOOK.md` | `c87c3f49f61cb8d6d07bdc28402a44eab469f3465af4895886d8fabc85c52601` |
| `cpu_g2_invocation_final.py` | `931a0d69473fe7cdbf9fbbf537120931579bfc61df0b530d4506541e9c4fa65a` |
| `remote_cpu_g2_once.py` | `dc3ff189a60d4511289c7ca0ba625e7f464a02d0be4652a44a9968c64c242abc` |
| `scratch_guard.py` | `d0f4a5de111f9d64bc66e784128f2b2b54adc1ac1f693bc2e7f74f1e80f4e0b4` |
| `test_cpu_g2_final.py` | `6f90f636729113e327ee1a733b788fc89fc8013892b41e87ec1e0a5d644fd08c` |

The separate independent negative test file is
`research/v2/verification/test_20260925_a2_c5c_r4_independent.py`, SHA-256
`34d3815142533fab39ad90d6fa0a0af541c8a223a4105ca3b45c6f7ccf3cb874`.
`git diff 164cfcd8^ 164cfcd8 --check` passed. The r2 disabled-G2 HOLD and
r3 disabled static PASS were read in full; the r3 PASS does not establish a
host tool path or authorize this r4 attempt.

## Blocking feasibility finding: pinned checkout does not prove tool location

The r4 wrapper, remote adapter, scratch guard and command arrays all hardcode
these two executable tools **inside** the intended clean e65 checkout:

- `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2/workflow/tools/preflight_dryrun.py`
- `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r2/workflow/tools/contained.py`

Yet `git ls-tree -r --name-only
e65a3097a5b00f6178188a935bb2acb9a5042c6c workflow/tools` emits **no
paths**. That commit's `train.py` and `prepare.py` do hash to the requested
`c5c410af...` and `5807c46e...`; it is the *tool* location that lacks
source-tree support. The prior accepted trn22 A0 CPU-G2 launch explicitly
used `/dev/shm/trainium-v2-g2-20260925-trn22-r1/workflow/tools/preflight_dryrun.py`,
not the e65 path; its archived `launch.json` hashes to
`483152e6b785e1c26be266cf0775391f579f24dc3efc17c2ce7b4e1b8641bb22`.
The 07:33 trn22 snapshot (`OPERATOR_RAW_TRN22_READONLY_20260925T073323Z.json`,
SHA-256 `c8a44adfea5b7be7f12603dc3f2ef97c32b1f1495d199e8ac42cc281591f94dd`)
reports a clean e65 checkout but **does not inventory either tool path**.
Untracked/excluded host files are not ruled out by the Git tree alone, so
actual absence is **unknown**, not asserted as a measured host fact.

This is not a silent safety bypass: the remote fresh audit will reject a
missing tool hash. But `operator_once()` creates/fsyncs its irreversible
one-attempt marker *before* that first SSH audit. Assigning the package with
the present evidence can therefore consume the only label on a predictable
source/tool-path refusal, with no CPU result and no same-label retry. The
supplied suite did not test the pinned commit's tool tree or the historical
tool placement. The independent offline test reproduces both observations
and confirms that a missing `g2_tool` or `contained_tool` fails the read-only
audit; it does **not** substitute for a fresh host inventory.

## Other static checks and limited passing evidence

- The binding's request, runner, remote, guard, canonical commands, both
  strict-host-key SSH argv arrays and stdin payload hashes reconstruct from
  the exact package. Remote `authenticate_payload()` separately pins the
  canonical request, commands and read-only probe. The supplied request,
  command, probe, capacity and host-identity mutations fail closed.
- The exact science profile remains the reviewed ordered 27-token c5c A2
  m64/d128, layers `1,-1`, BF16 RMSProp argv. It is synthetic CPU-only,
  seed 42, LNC2/four ranks, two steps; there is no native or eval command.
- The scratch guard requires an exact new private root and `tmp/`, checks
  positive free-byte/inode and projection floors, rejects TMPDIR fallback,
  and the independent symlink-TMPDIR witness fails before any G2 tool call.
  All these are offline checks; the fresh 32-GiB disk and 96-GiB available-RAM
  thresholds have not been measured for this attempt.
- The local wrapper requires actual fixed-path independent review and
  Supervisor assignment bytes, checks local STOP/breakers, and creates one
  exclusive/fsynced marker before SSH. The remote adapter checks STOP,
  process, Neuron holder, host lock, label, source and capacity in the fresh
  audit and again in the run session. A pre-exec bridge rechecks boot/mount,
  source/tool hashes and STOP before contained launch. A separate audit SSH
  is attempted unconditionally after any first host-contact attempt; an
  injected first-SSH timeout still caused that postflight attempt in the
  independent offline test. This is not a live preflight or a continuity
  proof against same-UID/privileged races.
- The planned remote result retains the aggregate, four raw rank records,
  eight ordered RMS trace records per rank, step log, checkpoint and run-log
  hashes. The unchanged G2 tool does **not** record table IDs on those RMS
  checks. Accordingly this package's numeric gate stays
  `HOLD_INDEPENDENT_PER_TABLE_PROOF`; a later *different* numeric Verifier
  must authenticate the raw files and source-anchored b1/t1/b5/t5 mapping
  before any CPU numeric PASS. A CPU PASS would still not be native A2 or BPB.

Supplied tests: 9/9 PASS under `python -B` and 9/9 under `python -OB`.
Independent offline tests: 6/6 PASS under `python -B` and 6/6 under
`python -OB`. These are bounded static test outcomes, not G2 or quality
results.

## Narrow correction and next authority

Before writing a final PASS or Supervisor assignment, obtain a **separate
read-only** trn22 receipt proving both exact e65 tool paths exist as regular,
no-follow files with the pinned SHA-256, owner/device/inode and clean linked
checkout identity, plus current STOP/occupancy/disk/RAM feasibility. This
must occur **before**, not after, consuming the one-attempt marker. If either
path is absent, a distinct Implementer must bind a separately reviewed ext4
tool/source stage or an already verified clean checkout containing both
tools, then regenerate all command/payload/SSH hashes and use a new label;
do not mutate the accepted e65 checkout or silently fall back to `/dev/shm`.
The repaired exact-byte package needs a new independent final review. Only
then may the Supervisor issue a CPU-only, one-attempt assignment; an Operator
must still perform the package's live preflight, contained CPU G2 and
unconditional distinct-session postflight. A separate independent numeric
review remains mandatory before any native-science request.
