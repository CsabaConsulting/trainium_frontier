# c5c exact-size A1/A2 CPU G2: minimal held handoff

Status: **IMPLEMENTER DRAFT / NO HOST CONTACT OR LAUNCH AUTHORITY**. This is
not a replacement for a proposal, independent Verifier receipt, Supervisor
assignment, or Operator preflight. S0 and the 116188 cache contrasts remain
ahead of A1/A2 in the active queue. No training, evaluation, AWS action,
cleanup, or source edit was performed for this handoff.

## Frozen inputs

- Candidate: `workflow/proposals/research-v2/train.py`, SHA-256
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.
- Unchanged organizer module: `prepare.py`, SHA-256
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- CPU tool: `workflow/tools/preflight_dryrun.py`, SHA-256
  `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`.
- Queue snapshot: `research/v2/queue.json`, SHA-256
  `d66da6cb128e2abb8ad7340222d1c013c28005c9e444a8b13429e232a7fa462e`.
  Derive each arm by replacing the **single** `--no-ngram-ve` token in
  `base_command` *in place* with that arm's `argv_delta.with`; do not append,
  reorder, change size, or use the old 35c9 source.

The reviewed A1 proposal and gate contract are in
`research/submission-validation/20260925-v2-c5c-a1-m64d128-trn21-r1/`;
the A2 equivalents are in
`research/submission-validation/20260925-v2-c5c-a2-m64d128-trn22-r1/`.
Those science proposals do not themselves authorize a CPU or Neuron attempt.

## Exact arm argv (as ordered by the queue)

**A1 / trn21:**

```text
--n-embd 1024 --ngram-ve --ngram-ve-layers=-1 --ngram-ve-table-mult 64 --ngram-ve-dim 128 --ngram-ve-trigram --ngram-ve-opt rmsprop --ngram-ve-flat-lr --ngram-ve-bf16 --ngram-ve-clip-exclude --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015
```

**A2 / trn22:**

```text
--n-embd 1024 --ngram-ve --ngram-ve-layers 1,-1 --ngram-ve-table-mult 64 --ngram-ve-dim 128 --ngram-ve-trigram --ngram-ve-opt rmsprop --ngram-ve-flat-lr --ngram-ve-bf16 --ngram-ve-clip-exclude --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015
```

## Narrower CPU-only path to review

For each arm, a *new* unique G2 label and a separate one-use Supervisor
assignment must be created after the S0 contrast work. The Operator stages
**only** the exact c5c `train.py` and exact CPU tool into a new, private run
directory, verifies both on-host hashes, and points `--prepare-dir` to an
already trusted, unchanged host checkout of `prepare.py`; no data or
`prepare.py` copy is needed. Use the existing tool with
`--steps 2 --nproc 4 --micro-rows 2 --micro-tokens 256 --timeout 3600
--trace-rmsprop --keep --out <new-run-root>/g2.json -- <exact argv>`.
Omit `--real-data`: the tool replaces training inputs with synthetic tokens.
Set an existing writable private `TMPDIR` on the intended filesystem and
verify its resolved path before launch; the current tool's unqualified
`tempfile.mkdtemp()` can otherwise fall back silently. Keep the complete
`g2-*` work directory, four raw `rank*.json` files, raw log, checkpoint and
aggregate receipt. Never reuse a consumed label or delete partial evidence.

Fresh preflight must show one job at most per host, no Neuron holder or
concurrent train/eval/G2 process, clean exact source identity, >=32 GiB free
root disk with the separate >=6.5 GiB operational floor intact, >=64 GiB
`MemAvailable`, enough inodes and reviewed scratch projection. Older
snapshots do not satisfy this gate. If the CPU tool is absent on the host,
explicitly stage and hash it; the held A2 r4 packet failed because it
assumed two untracked tools already existed in an old e65 checkout.

The independent numeric review needs every rank to report two finite
losses, one common ownership plan, cross-rank-identical parameters, a
strict-load checkpoint, and changed table digests. A1 expects `b5,t5`,
two RMSProp calls and four per-rank table checks; A2 expects
`b1,t1,b5,t5`, two calls and eight checks. Every check should have
67,108,864 elements, touched rows in 1..524,288, unchanged untouched
rows, finite gradient median and `max_abs_err_over_lr <= 0.03`.
The current tool does **not** name tables in its `rmsprop_trace.checks`.
Thus aggregate `verdict=PASS` alone is not an attributed numeric PASS:
an independent reviewer must map raw group parameter order to named tables
from the pinned source and raw rank files, or first request a separately
reviewed additive tool patch that emits authenticated table names.

The earlier A2 r3 package remains HOLD: its command contains
`UNBOUND_PROJECTION` and binds stale e65 checkout identities. It should not
be enabled by swapping placeholders. This handoff deliberately avoids a
second launcher/state machine; it proposes a one-use Operator CPU command
only. Native 30-step smoke, then a full 1800-second arm, still require
their own proposals, reviews, unique labels and same-host scored A0 control.
