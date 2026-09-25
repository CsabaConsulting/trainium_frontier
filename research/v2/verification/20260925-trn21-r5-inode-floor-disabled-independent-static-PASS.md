# Independent trn21 final-r5 inode-floor static review — PASS, disabled only

**Verdict: PASS_DISABLED_STATIC_ONLY.** The exact five-file candidate repairs
the terminal free-inode defect documented in the final-r4 HOLD. This review
does not enable or assign a transfer. I made no candidate-source edit, host or
SSH contact, AWS action, source transfer, training run, evaluation, cleanup,
breaker reset, or submission.

## Exact subject and predecessor

The reviewed origin branch
`codex/20260925-trn21-r5-transfer-inode-floor-disabled` resolves to
`0bacf0d009b4894264a2f8097b7e5cad8286bf5a`, whose direct parent is
canonical r4 HOLD commit `30ff004e7ded31fe881982d3b0533c7f86f94407`.
The diff adds only these five files; `git diff --check` passes.

| New file under `research/v2/authorizations/trn21-c5c-a0-r4-transfer-only/` | SHA-256 |
| --- | --- |
| `FINAL_R5_DISABLED_INODE_FLOOR_RUNBOOK.md` | `9b8c6239fc40b2d19b2ad09c98fdd48b4013156e23ebe66eb021955d1c8397a5` |
| `TRANSFER_REQUEST.FINAL_R5_HOLD.json` | `6cdd7e966238e20eed16ab0a5abcf8ad89ec324318de539b230b7513ac01d65e` |
| `operator_transfer_final_r5_once.py` | `75b1ad77476d3963208674b5f285efc5e78cca91d895c071c3c8eec9c1f0bdb7` |
| `stage_r4_transport_final_r5.py` | `edcd84c6269b5e78dd1440ea6a565913e33d078350dab574e5a89e4265080db1` |
| `test_operator_transfer_final_r5_once.py` | `ce787cfc5cc933cc66dfc592e4dda84c7b423bc959aa3489633bcbb432aba716` |

The archived r4 HOLD at origin `7c4480818cfddb3fff413a94d42effbd1b140d9f`
and canonical `30ff004e` has the same Git blob
`514f58f2d9cde8b0a606b90983cc3a4ed9bf8cf5` and SHA-256
`7a7029e30a7a9193bd3e5fd8d241ec28f277ff1bd1966c864d41019a98c196d6`.
Its independent negative test remains SHA-256
`b9798cd4d805fa6a1103f61fbaf06fb1fe55e99bdb78c9e1a0735c1d5a8f63da`.
The r4 remote payload is the same Git blob at r4 candidate `bca55b2c`,
canonical HOLD `30ff004e`, and r5 `0bacf0d0`; its SHA-256 is
`ad30d93b464175c6df1550654076bc868a8bd8cd652fedd08bd5bfe91f055e2b`.

## Gate findings

- `stage_r4_transport_final_r5.py` requires plain-integer
  `identity_before.root_free_inodes` and `identity_after.root_free_inodes`
  at least `MIN_INODES=10000` before and after inherited full-probe validation.
  This covers the fresh preflight and unconditional full read-only postflight.
  The r5 write-argv builder and session comparators revalidate the preflight
  and writer's nested full prewrite probe. The postflight comparator revalidates
  both full probes. Exactly 10000 passes; 9999, absent, boolean, float and
  string values fail.
- The archived r4 counterexample still passes r4 validation with terminal
  9999 inodes in both full probes; r5 rejects both. A mutation introduced
  during inherited validation, or between the first validation and the later
  write-argv/postflight-comparison gate, also rejects. These are local
  synthetic receipt checks; they do not claim atomic protection against a
  changing host filesystem after a live sample.
- The request exactly matches the wrapper's expected fields and pins the
  repaired transport, unchanged r4 remote, r4 HOLD/test evidence, earlier
  predecessor chain, source-stage plan, and read-only trn21 lock/ACL receipts.
  The wrapper rehashes those actual files and verifies r3/r4 predecessors are
  disabled. Both r5 authorization flags are `false`; `run_once()` raises
  `DISPATCH HOLD` before opening future authority files, its marker, journal,
  or SSH.
- The unchanged r4 remote and its local tests retain the private `.cache`
  owner/mode/inode/link pins, opened-inode ACL absence, same-namespace mount
  proof, `O_RDONLY|O_NONBLOCK|O_NOFOLLOW` lock open, compatible exclusive
  `flock`, final STOP/process/Neuron/target gate, late science-label scan
  before `O_EXCL`, lock through receipt, distinct readback, and retained
  partial-target refusal. No r4, r3, or older package file changed in r5.

## Offline checks

- Supplied package suite: **58/58** under `python3 -B -m unittest -v` and
  **58/58** under `python3 -O -B -m unittest -v`.
- Separate independent inode and disabled-gate negatives: **5/5** normally
  and **5/5** with `-O`. Their file is
  `research/v2/verification/test_trn21_r5_inode_floor_independent.py`,
  SHA-256 `22997ceb8b89bb93340a227da7f790c40079c587aba6b5854fc1818f3ba72515`.
- Archived r4 independent local checks: **8/8**. One test deliberately
  reproduces the old inode gap; the other seven cover the lock, ACL, mount,
  activity, label, and partial-target behavior carried forward by the
  unchanged remote. No test contacted trn21 or invoked a real transfer.

The r4 private-parent inference remains bounded to the observed mount
namespace and cooperative launcher locking; same-UID/privileged actors,
already-open descriptors, another mount namespace, and an uncooperative
writer racing the final scan remain outside that proof. Fresh live host
checks, a separately named enabled request, independent final review, and a
Supervisor one-attempt transfer-only assignment remain required. This PASS
does not authorize transfer, checkout, CPU G2, native training, or evaluation.
