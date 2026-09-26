# Fresh independent mechanics review

Verdict: ACCEPT_DISABLED_MECHANICAL_COMPOSITION_ONLY. Launch remains REFUSED pending separate release and current machine gates. This is not a source/scientific result approval, native warm result, or adoption.

Reviewed isolated input: `/tmp/trainium-anchored-control-review-gbfk9v`.
Authenticated FILE_SHA256.json: `0eaf757f0843e6d7c0ecdfb2dbefd1d666566f31f484bef143850cbc2df2a913`; all 30 listed file sizes and hashes independently matched.
Current binding: `e71a92a55043871b7dd5660c191fad4135d20384e6fa1be3b6edc91398d83e10`.
Driver: `706b7945a7316fe610c3abef1efedaf8a95d0203eba1516fb93ac1d4a6aa9644`.
Raw observation: `fc48525fee3d026bd2fdda02daeb5ca8d1ef78369d321158711b978b30d67ccf`.

No implementer checkout, notes, branch, or history was inspected. No host/network/device/training calls were made. Only reviewer-owned local fixtures were created and removed. No input/source/history files were changed. Supplied test receipts were evidence, not the basis of the independent test implementation.

## Independently executed evidence

`reference_check.py` is the reviewer-written CPU/stdlib reference and negative-check program. `raw-normal.log` and `raw-optimized.log` are its retained output. Both commands exited zero, six methods passed each:

```
/usr/bin/python3 -B /home/poojith-devan/anchored-fresh-review-s7TzUGXv/reference_check.py
/usr/bin/python3 -O -B /home/poojith-devan/anchored-fresh-review-s7TzUGXv/reference_check.py
```

- Parsed original remote observation directly using Python integers. Current anchor matches every path, field, type, and value. Original refused binding fails. Each of four timestamps exceeds 2^53; +1, float conversion, and integer round-trip through float all fail the exact reference predicate. No rounded value is accepted.
- Independently parsed parent and current launch-command files. Same token count; only zero-based token 3 differs and equals the declared old absolute train path. Literal `--out-dir out` remains. Parent binding environments match declared parent copies; training changes only label/metrics path, evaluation only label. Fresh train/eval sandboxes and checkpoint route are inside the new root; executed source is outside it.
- Real local filesystem tests of the unchanged identity helper accept prepare0600 with readonly=False and reject readonly=True, wrong hash, hardlink, final symlink, and ancestor symlink. A byte-identical replacement changes inode and fails complete-identity equality.
- Extracted and executed the actual driver's import precheck without its main or launch operations. Local `json.py` shadow is refused; a competing `prepare/__init__.py` is refused as wrong prepare origin. These fixture files are never imported/executed.
- Independently compared parent/current ASTs for inherited preflight and phase: identical. Extracted lock validator rejects inode596824 against original66305:596823. Extracted accounting accepts a181s fifth as WARM_TARGET_MISSED with valid charge, retains UNPROVEN cache hit, and rejects charge1819s.

## Composition assessment

The current binding fixes the previously rounded nanosecond anchor; it does not silently redefine the raw reference. The driver pins that binding, checks opened-file identity with the unchanged helper, compares all captured anchor fields, and establishes a full baseline before training. It rechecks after training, before evaluation, after evaluation and at finalization. Historical directory and old packet parent fields remain fully compared; shared higher ancestors compare device/inode/mode/uid/gid, with nlink/timestamps observational. All file identity fields remain compared. The earlier specification ambiguity is resolved.

The anchored prepare0600 contract is explicitly accepted here as a present-time, no-writer, pre/post observation contract. It is not W1's readonly-two-file-directory predicate or historical/OS-enforced immutability. No chmod, replacement, repair, or writes to historical files appear in the reviewed composition. Read access can update atime; atime is explicitly outside the invariant. Transient hostile writes cannot be excluded by this scheme.

The static import contract requires the colocated prepare.py path and rejects named source import roots plus sitecustomize/usercustomize in the script directory. This is an expected standard resolution check, not runtime worker-origin proof, a complete transitive-import security boundary, or evidence about current site packages. The train source bytes themselves are not included in this isolated packet; this review does not independently regenerate source_import_roots or reopen scientific source approval. Runtime hashes must establish that the old actual train/prepare files still equal their frozen source pins.

The ordinary harness creates fresh local train/prepare copies, executes the package launch vector from fresh train cwd, and binds output/step-log there. Evaluation remains a fresh ordinary harness/evaluator process using the fresh eval package copy and the new checkpoint, public20,971,520/context1024/batch1. No evaluator modification is proposed by this review.

The unchanged guard authenticates the exact nonblocking outer flock command and its inherited already-held descriptor. Driver checks original device/inode and descriptor equality before synchronous phases and at final guard. Guard requires successful ps and neuron-ls before training and evaluation. Outer lock remains held by its parent across driver/train/eval; this review did not execute a real lock lifecycle on a host. Explicit descendant tracking/absence and original-lock release remain Operator evidence, as the binding states; a final guard or lock-free observation alone is insufficient.

## Required release/result evidence outside this acceptance

Do not convert this review into an automatic launch. Separate sole-Operator release must bind these exact reviewed hashes and enforce the absent new root, staged driver/support pins, clean tracked tree, no unresolved breaker/STOP, retired launcher absence, hostname, free resources/disk, priority, prior retrieval, and current original-lock/device/descendant quiescence. The driver alone does not implement every external protocol gate and the disabled metadata booleans are not a launch interlock.

Current live anchor equality/no-symlink/single-hardlink/import checks must pass under the original lock. They have not been observed by this reviewer. Actual worker argv/source bytes/SDK/environment and PID/starttime/PPID/PGID/SID records must be retrieved by Operator. Complete training/eval logs, charged-through-exit, finite losses, checkpoint continuity, official fresh20M result and all-observed-process postflight remain required for any valid-control claim. Source/native gates referenced in metadata are inherited claims, not independently re-proven here.

Meeting fifth<=180s would establish only the prospective operational target. It would not prove a cache hit, path-only causality, historical immutability, full-cold qualification, scientific gain, comparable mixed-regime selector pooling, or adoption. A warm-target miss alone must not stop an otherwise valid full control/evaluation. No new scientific policy, launch authority, cleanup authority, or automatic successor is created.
