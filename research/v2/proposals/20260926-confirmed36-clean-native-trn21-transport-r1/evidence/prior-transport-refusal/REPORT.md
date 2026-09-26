# Independent frozen transport review

Verdict: **REFUSE** for the exact disabled transport mechanics. Host/native status: **HOLD, NOT EXECUTED, NOT AUTHORIZED**.

Reviewed 2026-09-26 as a fresh independent Verifier, requested as GPT-6 Astra; the interface does not independently attest the runtime model identifier. Inputs were only the frozen input-only bundle `/home/poojith-devan/clean36-trn21-transport-fresh-scdhUJ`, its lane specification, original packet and gate receipts. No implementer branch/worktree, author test source, or unrelated repository state was inspected. Author results and earlier independent reports were evidence, not substitutes for executed tests.

## First reproduced failing gate

**P1: checkpoint exclusion checks an earlier file version, then emits a subsequently opened version before authenticating it.** `transport/transport.py` lines 201–213 hash the pathname, reject the prohibited hashes, emit the file header, reopen the pathname and immediately emit chunks. Only after those chunks does the code compare their bytes/hash to the earlier pin.

The reviewer test uses a 32-byte ordinary artifact and a different 32-byte synthetic prohibited checkpoint. Only in the fixture, CHECKPOINT_SHAS is replaced by the exact synthetic checkpoint hash. An emit callback atomically replaces the pathname immediately after the real file header is emitted, before the actual production streaming open. The real retrieve function emits all 32 prohibited bytes, then raises the first real error `RETRIEVAL_FILE_CHANGED:artifact.log`. There is no complete record. Feeding the emitted records to the real receiver retains those exact checkpoint bytes under `files/artifact.log`, then raises `WIRE_INCOMPLETE` and retains its failure receipt. Normal and optimized Python reproduce the same result.

This is a demonstrated byte-transfer exclusion failure, not a false successful-copy claim, observed host race, or claim that the real retained checkpoints changed. The test uses only tiny bytes and changes no real checkpoints. It directly conflicts with the specification's promise that the exact checkpoint hashes are rejected before contents are emitted. The late stable-byte check correctly detects change but cannot retract transmitted bytes. The receiver similarly cannot rely on an earlier advertised allowed hash for a no-copy guarantee.

Required repair direction: bind the bytes approved by checkpoint exclusion to the actual bytes that may be emitted, closing replacement and mutation windows, and independently retest that prohibited fixture bytes never reach the wire or retained receiver partials. A receiver completeness failure alone does not close this finding. No repair was made in this review.

## Evidence that passed

Every entry in the frozen SHA256SUMS authenticated. The expected input manifest SHA256 is `6e19cf83732b502f1bac32bd7b7d47baa6a624f6c6055fc0c06af1f5d2a32309`; exact TRANSPORT.diff SHA256 is `d1cc9a24f71ef16c85b3be6227bee0b5fa07be6b8ae01cc763be18782e79b5cb`. The diff is additive and names only SPEC.md, TRANSPORT_BINDING.disabled.json and transport.py. The transport source is `353874440b29337fbc438c359ec5110830e3cd8891037add95fcb8de2731d530`.

The real packet_modules authenticates all nine fixed packet files against original manifest `905eb5fb73281c4058c52f6d16f4824dd1146929fa80b64a7b9d8d7d41d61f8f`. A changed INPUTS payload rejects. The original 17 input records and both checkpoint hashes are preserved. The disabled binding retains false authority flags and exact packet/driver/helper/foundation/guardian pins. Transport source imports only authenticated stdlib driver/guardian/idle definitions; no dispatch or scientific helper invocation appears. Historical runtime observation remains a fact receipt, not fresh authorization.

Reviewer-owned `verify.py` implements an independent byte/count reference from explicit fixture byte strings. It compares exact SHA256/length maps, total artifact bytes, per-rank hidden partial/final receipt bytes and cache bytes against the production retrieval result. Hidden files, partial receipts and a 281600-byte binary file spanning chunks round-trip exactly. Cache bytes are counted and not copied. A total-cap failure still retrieves the complete available evidence and returns COMPLETE_COPY_FAILED_GATE.

Actual root and nested os.walk scans with injected scandir PermissionError propagate failure. A file symlink rejects. The receiver rejects an existing destination, traversal, corrupted bytes, trailing records and missing completion, retaining failure evidence. These tests use explicit exception checks active under -O.

The actual stage transaction runs with the exact frozen nine-file payload into a new fixture root, holding a real local original flock. Each write is instrumented with a second independent open/flock attempt, which must be excluded. The same lock inode/device and existing lock bytes survive staging; exact staged bytes authenticate; the release is written after the completion receipt; an existing root rejects. Host preflight, resource, STOP, runtime and retained-input checks are mocked in this focused staging fixture and are not claimed as newly executed host proof. Production source delegates those checks to the unchanged reviewed driver. The real guardian original-lock check is used. No native child is spawned.

The suite is bounded; it does not newly re-execute every earlier guardian/lifetime scenario or every possible malformed release. These limits do not weaken the independently reproduced refusal.

## Reproduction and artifacts

Executed `/usr/bin/python3 -B verify.py normal-final` and `/usr/bin/python3 -O -B verify.py optimized-final` in this output directory. Both exit 0 because the verifier explicitly expects and proves the failing behavior; the verdict printed in each log is REFUSE. `normal.log` and `optimized.log` contain the final suite outputs. Each final fixture directory retains `checkpoint-race.json`, the source replacement, transferred files, receiver failures and staging records. The initial `normal/` fixture is retained from an earlier reproduction with unequal sizes; the final tests strengthen it to equal sizes and independently prove local partial retention too.

`INPUT_MANIFEST.sha256` preserves the frozen input inventory. `SHA256SUMS` authenticates this report, reviewer source, input manifest, final logs and final race evidence. Fixture symlinks and large repeated fixture byte files are not recursively followed into that concise output manifest; the test source deterministically describes them.

No SSH, network, native SDK, full tensors, host staging/retrieval, training, evaluation, checkpoint access, launch, AWS operation, cleanup, or source repair occurred. All writes are reviewer-owned local outputs. Missing independently checked release, sole-Operator assignment, authenticated strict connector receipt, current controller state, fresh host/runtime/resource/input checks, and native package facts remain separate explicit pre-use blockers. Native scientific evidence is expected from the later diagnostic and is not demanded as proof of these local mechanics. This report grants no launch or submission authority.
