# trn21 consumed-staging cleanup package

Status: **NON-EXECUTABLE / PENDING INDEPENDENT VERIFICATION**. This is cleanup-only. It
authorizes no host access, training, evaluation, r4 staging, cache cleanup, checkpoint pruning,
AWS action, submission, or deletion. There is deliberately no executable authorization blob.

Repair r2 descends from independent FAIL `e1152cb26c78dbdc68b5c57f5b0749760dcade82`
(receipt SHA-256 `c874891cbc754943eb4ca2d1f5f37b9f7a994200cd87c563a81133edd5867051`).
It replaces path-based unlinking with exact-inode same-filesystem quarantine, hashes bundles again
after quarantine and immediately before unlink, binds complete checkout membership/content, and
creates/fsyncs an append-only O_EXCL intent journal before the first destructive rename.

## Frozen scope

The only deletable names are the ordered five targets embedded in `cleanup_contract.py` and
`cleanup-request.pending.json`: the consumed r1 checkout and bundle, consumed r3 checkout and
bundle, and the consumed r2 **bundle only**. The r2 checkout and its eight exact runtime files,
the G0 checkpoint, every cache, every log/result/evidence file, all unknown paths and every other
worktree are preserved. The r1/r2 bundle hashes are archive-backed full values, not prefixes.

The package binds failure evidence `e1550203`, `5b392db4...`, `5e45a564...`, and analysis
`83e53354...`. The frozen required free space after cleanup is `7,286,616,064` bytes.

## Required independent sequence

1. Verifier reviews this commit and confirms exact hashes, target set, no science changes and all
   tests. It creates a later finalized request with the exact reviewed 40-hex package HEAD. It must
   not alter any other field. The pending file is not renamed or treated as authority.
2. Operator, under separate read-only host authority, supplies that mode-0600 finalized request to
   `python cleanup_contract.py inventory --request <finalized-request>`. The tool performs no
   deletion and writes the O_EXCL inventory to the fixed immediate child of `/home/ubuntu`.
3. Verifier retrieves and authenticates that inventory: host identity, free lock/device, no
   training/compiler process, exact origin/HEAD/clean/fsck checkout state, same filesystem,
   bundle bytes/hashes/metadata, preserved r2 files, and preserved G0 checkpoint. The conservative
   projection must meet the threshold.
4. Only a new human/Supervisor cleanup decision plus independent verification may produce a
   mode-0600 finalized authorization containing the exact request SHA, inventory SHA, reviewed
   HEAD, exact five targets and a distinct verifier commit. No such file exists in this package.
5. A Cleanup Operator may then use `python cleanup_contract.py execute --request ... --inventory
   ... --authorization ...`. It redoes every check while holding the host lock, compares the fresh
   inventory exactly, deletes by fixed parent dirfd with no shell/glob/environment expansion,
   fsyncs after every quarantine/deletion outcome, proves exact absence and preservation, requires
   at least the frozen free-byte threshold, and writes/readback-seals one O_EXCL durable receipt.
   Every caught interruption after intent yields a durable `PARTIAL_FAILURE` journal/receipt; it
   can never be reported as PASS. Unknown/replaced entries are never recursively accepted/unlinked.
6. A different Verifier authenticates the receipt before any r4 package or staging is considered.

Any missing target, hash/size/metadata drift, symlink/hardlink/nonregular entry, checkout drift,
host activity, held lock/device, filesystem mismatch, protected-artifact drift, existing output,
or insufficient free space stops the operation. There is no fallback and no broad recursive path.

## Local implementation checks (safe; no host access)

```bash
python -m py_compile research/submission-validation/20260925-v2-a1-consumed-staging-cleanup-trn21/cleanup_contract.py
python -m unittest -v research/submission-validation/20260925-v2-a1-consumed-staging-cleanup-trn21/test_cleanup_contract.py
```
