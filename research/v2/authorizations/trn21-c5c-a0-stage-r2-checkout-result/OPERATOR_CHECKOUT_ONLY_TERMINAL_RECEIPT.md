# trn21 c5c A0 stage-r2 checkout — Operator terminal evidence

**Operator result: `STAGE_EVIDENCE_READY_FOR_INDEPENDENT_REVIEW_ONLY`.**
One exact checkout attempt occurred; it is not yet an independently accepted
checkout. CPU G2, native training, evaluation, adoption and submission remain
HOLD. I did not retry, clean, delete, overwrite, reset a breaker or change AWS.

## Authority and local gates

- Supervisor checkout-only assignment: commit
  `4c501e834e7455ab5fb880a9a449ab1ff4aa8353`, SHA-256
  `4c1d1774bd2ac9070334e1bdad5d5fe4fd9f9f71b6c72b41272c0d211243e4db`.
  The assignment reserves science label
  `fs-20260925-v2-c5c-a0-control-trn21-r1` without launching it.
- The actual on-disk independent final-request PASS receipt was a regular,
  owner-owned, single-link mode-0644 file with SHA-256
  `acfc9036cf0864a2abf019e961e7680de976aba99b5ffcd656d2d4acf39f04eb`.
  The enabled request, invocation binding, wrapper, remote adapter and
  extraction code matched their assigned hashes. The exact reviewed
  audit/stage argv and stage stdin hashes matched the assignment. The fixed
  marker was absent beneath an owner-private mode-0700 parent before use.
- Local `research/agentic/STOP` and `research/v2/STOP` were absent; the
  controller's global-correctness and trn21 breakers were untripped. The
  controller still had scored-training `launch_authorized=false` and trn21
  `launch_allowed=false`; the separate assignment authorizes checkout only.
- A separate **read-only** host check before invoking the one-shot wrapper
  returned SSH rc 0/stderr empty at `2026-09-25T03:15:58.547348Z`:
  expected hostname/UID/boot/mount, checkout absent, two checkpoints and six
  warm caches matched, no matched process/device/lock/STOP, and
  `58,571,833,344` root-free bytes. It was not substituted for the wrapper's
  own fresh preflight.

## One-attempt timeline and raw evidence

The reviewed `checkout_invocation_final.operator_once` consumed the **fixed**
exclusive/fsynced marker at
`/home/poojith-devan/.local/state/trainium-trn21-c5c-a0-stage-r2-checkout-attempt-once.jsonl`
before host contact. The marker remains present, mode 0600, owner-owned,
single-link and must never be removed or reused. Its byte-for-byte archived
copy is `OPERATOR_ONE_ATTEMPT_JOURNAL_20260925T031632Z.jsonl`:
`52,364` bytes, seven JSONL events, SHA-256
`b0b5d712ff0d83f513c9cdc017ad1e5358a42ab1e1f31072888323f8dc4493b3`.
It contains the exact reviewed argv and full SSH return codes/stdout/stderr;
the three stdout payloads are also extracted byte-for-byte into files below.
The raw journal was screened for private-key PEM, AWS/OpenAI/Anthropic/GitHub/
Slack token shapes, credential-variable names and password assignments; all
listed marker counts were zero. No private-key contents were read or archived.

| Phase | Journal event UTC | rc / stderr | Raw stdout file | SHA-256 |
| --- | --- | --- | --- | --- |
| Fresh full read-only preflight | `03:16:39.237759` | `0 / empty` | `FRESH_PREFLIGHT_20260925T031638Z.json` | `915d59627ea1538b7b6ac1e3d715065d1511ff0e18dc27ff29eb8b30f34e7cd8` |
| One exact checkout SSH | `03:17:32.168214` | `0 / empty` | `CHECKOUT_STAGE_RECEIPT_20260925T031732Z.json` | `097256565cfdf31f2bd5f17531e404f8911c22d35eb0d631c1b0a4d99c54ed37` |
| Unconditional separate full read-only postflight | `03:17:38.833530` | `0 / empty` | `UNCONDITIONAL_POSTFLIGHT_20260925T031738Z.json` | `4dea161fedcee76d335c053144bc7a7b36f3f20299ced2412151c3170c6d2fba` |

The wrapper's own fresh preflight at `03:16:38.939031Z` passed the exact
host, boot, mount namespace/ext4, parent, three persistent source/tool
identities, two protected checkpoints, six caches, absent checkout/science/G2
paths, STOP/labels, process, Neuron and lock checks. Root free was
`58,571,821,056` bytes, above the fixed 10-GiB floor.

The stage receipt reports a 6,338-member safe archive scan (6,283 files,
55 directories, four GNU long-name headers), exact frozen source HEAD
`e65a3097a5b00f6178188a935bb2acb9a5042c6c`, one shallow Git commit,
clean status, no remotes/untracked files, `git fsck` rc 0, and pinned
`train.py`/`prepare.py` hashes
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586` /
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
It reports checkout path
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2`
as owner-private mode 0700 on device 66305/inode 3168319, with science output
absent. Its CPU G2, native and evaluation authority fields are all false.

The separate postflight at `03:17:38.525471Z` again passed the full reviewed
snapshot and saw the checkout present, private and on ext4. Root free was
`58,368,110,592` bytes, still above 10 GiB. No matched process, Neuron/lock
holder, STOP or science/stage label was reported; the two checkpoints, six
caches and three persistent transferred files remained pinned. The wrapper
returned `STAGE_EVIDENCE_READY_FOR_INDEPENDENT_REVIEW_ONLY` with
`stage_attempted=true`. A different Verifier must assess these raw records
before any CPU G2 or further host action.
