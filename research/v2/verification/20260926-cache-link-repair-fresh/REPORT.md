# Independent narrow cache repair review

Verdict: ACCEPT_NARROW_CACHE_ACCOUNTING_AND_R1_READONLY_RETRIEVAL.

Reviewer: fresh independent verifier task native_cache_links_fresh_verifier, GPT-6 inherited session. Inputs were exclusively the frozen source/spec/factual-receipt bundle `/home/poojith-devan/cache-link-fresh-review-47rwhD`; no implementer tests, worktree or notes were used. Candidate commit 22e3f301ca6054f0a69e5cab151f28d8dbeb05e6; baseline 20576a9dba83b3f7f83a73765fde8d90f279a41d. Frozen SHA256SUMS SHA256: `4a5a74f774a0881b83ba1a9ab8f1148333460cf4671a3f98d0bd701c104ff60e`. Every listed input authenticated. Review scope is the supplied cache-link-repair SPEC and its narrow amendment to the supplied transport/science/packet/K2 specifications.

Accepted source deltas: opaque lstat/readlink metadata for links beneath the six clean diagnostic cache domains and K2 cache/tmp trees; strict rejection elsewhere; original r1 retrieval-only transport, with both stage() and stage wire construction hard-disabled. K2 overlay changes only monitor.py and its SOURCE_PINS entry. r2 was inspected for unchanged numerical bytes/identity correctness but is PARKED per Supervisor's latest user steering; this report does not recommend a retry.

I independently wrote and executed check.py using tiny local real files and links. `python -B check.py` and `python -O -B check.py` both exit 0: eight grouped tests pass. Raw outputs are normal.log and optimized.log. These are CPU mechanical evidence only, not SDK compatibility, native success, timing or model-quality evidence.

The tests establish exact ordinary-file plus link-literal-byte counts, literal target metadata, and no content reads during the census: outside-file, broken, directory and cyclic links remain opaque. Open interception requires O_DIRECTORY and O_NOFOLLOW and stat interception requires follow_symlinks=False; read/pread are forbidden during this test. Real link replacement during readlink fails SCAN_ENTRY_CHANGED; replacement of a directory with an outside directory link fails before traversal. Noncache links, root links, FIFO entries and injected scan permission errors fail. K2 regular hardlinks and entry caps remain enforced. Clean hidden rank partial files remain counted; artifact caps still reject excess.

The r1 retrieval test uses the real retrieve/receive functions with only host authorization/observations replaced by explicit local fixtures. It preserves hidden partial bytes, records link metadata, creates no received cache tree, and copies evidence despite cap failure while reporting COMPLETE_COPY_FAILED_GATE. Changing a link after the first census rejects completeness. Both sender and receiver checkpoint exclusions reject; a real in-place mutation between parent hash authentication and leaf emission rejects immutable-chunk substitution. Original authenticated_chunks and receive function ASTs are unchanged.

Identity checks authenticate the complete r1 and r2 small packets through their actual packet_modules verifier. INPUTS, foundation, guardian, idle and dispatch bytes are identical. Reversing the single authorized LABEL assignment in the r2 probe recovers donor SHA937e2f239ad90b5aa61e0b40ccf067b19ac02febe246492735e948a35c6b86af exactly. No model, loader, numerical helper, evaluator, data or checkpoint change is introduced. All unaffected host-driver and K2 monitor functions/classes have identical ASTs; lifecycle, time/resource caps, process tracking, release checks and anti-substitution mechanics are preserved. Disabled authority flags remain false. The original r1 packet is read only and remains authenticated by its historical manifest.

The native factual receipt identifies an actual installed-SDK table link, lstat121 bytes, whose target is reported as4096 bytes; it explicitly says target contents were not read. This supports the practical cache-metadata incompatibility. It is not a native diagnostic pass.

One initial test fixture assumed filesystem enumeration order and attempted mutation before the first cache observation on this filesystem; it incorrectly expected rejection. The raw failed runs remain in fixture-order-normal.log and fixture-order-optimized.log. I corrected the independent fixture to mutate strictly after completion of the first scan, then reran both interpreters successfully. No candidate source was edited.

Remaining operational gates are the applicable sole-Operator release/connection binding, current machine and runtime/source/resource checks, and retrieval/result review. This report grants no host action, staging, native run, cleanup or guardian release. The separately approved idle-checked r1 hold recovery and any automatic-release launcher repair are outside this source review. The newest user steering parks clean r2; no native B1/B2 upload gate is added here. K2 native evidence remains unmeasured, subject to the user's current research-run scope.

Review limitations match the supplied specification: sampled census rather than an atomic snapshot or arbitrary same-UID adversary proof. No new protocol or theoretical gate is introduced.

Receipt hashes:

- check.py: a6e44712de68d921987616e6a0451dbb93fb8a9a3103fa2beed2b2ce3ff0e5fd
- normal.log: 02ebe137cbbe11edbdafcd96cfebdec9fa272086a8cc1278e1e68c4ed8745219
- optimized.log: 4466a5b4a34ca6bf9a58e38c8d3757d1559acf79f09dd8bcb3c6ec233f068f01
