# Independent Verifier — trn21 c5c A0 stage-r2 transfer result

**Verdict: PASS for the archived three-file persistent-ext4 transfer result only.** The
source capsule and two CPU helper files were each written once to their exact new
`/home/ubuntu` target and re-read with matching bytes and SHA-256 in a later,
distinct SSH process. All three passed a second, further distinct read-only
verification after the writes. This is a retrospective evidence verdict, not a
fresh live host preflight or authority to extract, run CPU G2, train, evaluate,
delete, reset a breaker, change AWS resources, adopt, or submit. Those gates
remain **HOLD** and require their own exact requests and role checks.

Reviewed the canonical archive at `d8c16692e668e801a76134857ce743c25992c291`,
including the Operator terminal receipt SHA-256
`9ab90252e5e21812f5aad2af7fd0bfd1e71205e2f8f740f6f41faf6606fe6a2e`,
all 17 referenced request/refresh/result files, and the prior static-reviewed
package. The narrow Supervisor assignment has SHA-256
`0127358fed6e97787b8a7d60208ede3cadb0e870791baece2f12830300c09fd5`;
it authorizes transfer only, not checkout or later gates. The contract, writer,
builder, runbook and prior static-review hashes match the assignment and the
terminal receipt. The Operator's remote-evidence branch is pushed at
`1e32a19bacc85052dbc009c2fc2e13dc5db4aad0`.

| Artifact | Bytes | SHA-256 | Host inode | Write / first verify / final verify PIDs |
|---|---:|---|---:|---|
| e65 source capsule | 104,005,787 | `32c7181668884488bf920a165982ffd2167096376aebaa6eb4cca2793b1c4264` | 296045 | 2843371 / 2843458 / 2844383 |
| CPU preflight helper | 23,550 | `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8` | 294946 | 2843757 / 2843833 / 2844458 |
| CPU containment helper | 9,185 | `0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d` | 296647 | 2844117 / 2844193 / 2844510 |

## Independent raw-record checks

- Rehashed each of the 17 referenced raw JSON files against its terminal-receipt
  pointer, and verified the final request/result pointers. The six write/first
  verification argv arrays and three final argv arrays equal the exact reviewed
  builder output, including `ssh -T`, `BatchMode`, `IdentitiesOnly`, strict
  host-key checking, the fixed trn21 endpoint and literal pinned writer bytes.
  Their recorded argv SHA-256 values also recompute exactly. No different
  target, source, writer or operation entered these commands.
- For every write and both verification passes, the recorded SSH transport
  returned zero, did not time out, and had empty stderr. Its stdout parses to
  **exactly** the separately archived writer receipt. The nine receipt times
  are within their respective transport intervals; requests/refreshes preceded
  the corresponding executions. The archived Git commits for the requests
  also precede the recorded executions. This is stronger than accepting an
  Operator-computed PASS field alone.
- Independently recomputed `compare_cross_session` from the raw write and
  first-verify receipts, then again from the raw write and final-verify
  receipts. All six comparisons match the archived result. The nine
  `(PID, process-start-ticks)` pairs are distinct; each later verification
  occurred after its write, and each final verification after its first.
- Every receipt binds hostname `ip-172-31-34-85`, UID 1000, one boot ID,
  mount namespace `mnt:[4026531832]`, the same ext4 `/dev/root` mount ID 27,
  and `/home/ubuntu` device 66305/inode 295711. Each target remained a
  single-link, regular, UID-1000, mode-`0600` file with the same device/inode,
  exact byte count and full readback SHA-256 across its sessions. The old
  consumed `/dev/shm` host target was not used.
- Before each write, all three archived prewrite snapshots show the current target
  absent, previously written targets present with expected hashes, root free
  above the required 10,737,418,240 bytes, safe ancestry, intact two protected
  checkpoints and six warm-cache identities, no recorded STOP/breaker, no
  science label, no matched training/evaluation/G2/guard process, and no
  Neuron or shared-lock holder. The local breaker-state hashes correspond to
  committed `research/v2/state.json` versions `af2b1ad1` and `28407c3c`.
  Current local copies of the capsule and two helpers independently rehash to
  the pinned bytes and sizes; the capsule remains mode `0600` locally.
- The final read-only request checked all three files again. Its postflight
  SSH command returned zero, with stdout equal to the archived full snapshot
  at `2026-09-25T01:48:36.797405+00:00`: all three target hashes/inodes still
  match, root free is 58,571,935,744 bytes, both protected checkpoints and
  six warm-cache identities still match, and checkout/science-output/G2 paths
  are absent. The terminal receipt records no checkout, CPU G2, native train,
  evaluation, AWS action, deletion, overwrite, retry, or science-label use.
  A bounded high-confidence marker scan of these Operator JSONs found no
  private-key, AWS access-key, OpenAI project-key or Bedrock-bearer marker.

## Scope and next gate

These are archived Operator receipts, not my own current SSH observation; the
prewrite checks are archived snapshots rather than separately retained full
preflight command transcripts. I performed **no** SSH, host write, extraction,
training, evaluation, cleanup, AWS or submission action. The earlier tmpfs
disappearance still has unknown cause. The verified ext4 copy is eligible for
a **new, separately reviewed checkout-only request** after a fresh complete
host preflight and safe capsule-member/extraction review. That next Operator
stage must preserve existing files and return its own independently reviewed
source/Git receipt. CPU G2 and native training remain HOLD even if extraction
later passes.
