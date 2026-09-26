# Fresh independent bounded review

Verdict: ACCEPT_DISABLED_HOST_BINDING_AND_EXPLICIT_TRUST_SCOPE_ONLY.
Native feasibility/release verdict: REFUSE_PENDING_NATIVE_RUNTIME_AND_HOST_GATES.
No staging, launch, reservation, adoption, controller change, or native result is authorized.

Reviewed frozen input only: /home/poojith-devan/k2-trn22-evidence-review-Mh0Zed.
Input SHA256SUMS digest: c8af340932a1a1e9be40ac45d6149491d83be69bc81f7351f03cd3ba3d36f410.
All input manifest entries verified before and after this review. No implementer
worktree, notes, author tests, SSH, native SDK, or controller was used. This is a
fresh review of frozen bytes, not authentication of a remote Git commit.

## Own executed evidence

Commands in this directory, all exit 0:

    python3 -B checks.py > raw-normal.json 2> raw-normal.stderr
    python3 -B -O checks.py > raw-optimized.json 2> raw-optimized.stderr
    python3 -B symlink_check.py > symlink-result.json 2> symlink-result.stderr

The independently written scalar reference uses exact rational sums, a binary
search of the finite positive FP32 lattice, and nearest-even selection. It does
not use the candidate integer shifting/rounding algorithm. All 1024 F0 channels
matched at each of four additions. Eight separately transcribed F2 scalar
witnesses matched, including 00010000 and 80010000 reachable subnormals and
positive-zero initialization. Full F2 row expansion, token metadata, all output
words, and native execution were NOT tested by this review. Raw output retains
all F0 final words and eight F2 input/result witnesses.

Own negative tests reject invalid BF16 values (-1, 65536, bool, float), the actual
disabled entrypoint, flag-only enablement, flag/path enablement with staging
disabled, wrong/absent original lock identity, missing invocation inputs,
fabricated challenge, and incomplete route. These invocation tests are narrow
in-process state tests; they are not fresh seven-route/worker/admission tests.
No test modifies a candidate file. The local host-driver seam changes only
in-memory binding/path fields long enough to exercise early refusal, then
restores them. It never reaches any host helper or native launch.

Own pin comparisons establish consumer, worker, coordinator, oracle, fixtures,
generation, evidence, artifacts, monitor, and idle guard are byte-identical to
the supplied donor pins. Inverting precisely the label change in common.py and
the read-only anchor hash in host_driver.py yields the donor SHA256s. Thus the
worker/consumer/oracle and repaired lifecycle are unchanged. The test independently
matches the complete environment manifest to host_driver.environment output.
The diff and argv template preserve the same launch structure and four ranks.
Opaque author-test pins remain in SOURCE_PINS; no omitted test was read or run,
and the stripped snapshot is not a complete runnable native packet.

## Archived receipts used, with limited scope

Original mechanics REFUSE is preserved. I rely on four-repair-VERDICT.md
(6c03ad05e0201490e99655db9ab8f3b889bb562b2864eb7094646d891a49b953)
only for its independently reported local containment and four-process admission
checks, plus independently confirmed unchanged relevant source here. I did not
repeat its process tests. I rely on seven-route-REPORT.md
(7faad08ee432c4c318e3bec55d1e086cac041523b7997347aca8eab73b5068b5)
only for its disclosed mocked seven-route receiver/worker checks, including
corruption and missing-output negatives. Those results are neither new native
evidence nor an unconditional execution proof. Its positive facade explicitly
replaces SDK/kernel behavior and therefore cannot demonstrate a production
source-preserving exploit, nor prove real hardware invocation.

## Trust-boundary decision

Replacing the trusted SDK in a CPU facade is a trust-boundary limitation for this
new explicitly bounded diagnostic, not an unconditional pre-run proof requirement
that this packet can satisfy. I accept that distinction for the new diagnostic
design only. The historical unconditional bypass gate remains NOT_ESTABLISHED,
unchanged. A bypass reachable through supported inputs while preserving trusted
source/runtime would still block release. This review found no such new bypass;
it does not prove all such bypasses impossible. The real worker directly assigns
the pinned consumer result and captures it after synchronization; ordinary code
uses embedding, immediate BF16 cast, backward, and weight.grad. Process, source,
and complete-byte behavioral evidence depend on real SDK semantics. Fake CPU
objects or an accepted facade cannot count toward any of the 28 native observations.

## Host identity and unresolved gates

Binding is exact hostname ip-172-31-32-253, original lock device66305/inode832589,
new owned project /home/ubuntu/trainium_k2_f0f2_trn22_evidence_20260926_r1 at
65155c668efaa215a2136f67617b44ed32d4c9f6. That project is NOT YET CREATED or
verified clean. The binding amendment expressly supersedes the old dirty project
selection; no cleanup exception is inferred. Both projects remain STOP roots.
The original e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43
train.py anchor is read-only, hashed only, never copied/executed/changed.
Stage/launch/reservation bits are false and review/release hashes remain null.

Host receipt f02e57f6bfd37e71a2587820148555c4afa102c48ed5a592c91a469cd62ca963
is historical metadata and observation only. torch_neuronx and torch_mlir actual
runtime origins remain unresolved. Current interpreter/import/source compatibility
must be separately established and bound; distribution metadata is insufficient.
The worker hello reports versions but is not independent import-origin attestation.
Fresh target absence, owned-project creation/cleanliness, anchor identity, original
lock, STOP/breakers, quiescence and device-holder checks remain required. Fresh
post-creation/post-stage and in-lock >=16GiB disk and >=24GiB MemAvailable must
pass; the historical ~111MiB disk margin is not reserved capacity. Full expansion,
all 28 observations, actual ordinary controls, native timing/storage/containment,
complete retrieved blobs, and final postflight remain unobserved. All original
deadlines/caps and prior failures remain; no limit reduction or cleanup is approved.

## Newly raised compiler accounting compatibility gate

During review, Supervisor reported a separate trn21 native compiler startup
failure ARTIFACT_NONREGULAR at a default_control_fast_table.bin path. Its raw
receipt and actual object type were not in the frozen input or observed by me;
I do not relabel that report as personally verified native evidence.

I independently inspected and executed candidate monitor.tree_bytes on my own
cache/debug symlink. It rejects with SCRATCH_SYMLINK. sample() applies that
function to cache/tmp as well as artifacts/logs. Thus if the actual compiler
creates such a symlink in K2's monitored cache, this packet concretely fails its
accounting gate. It also rejects nonregular files and multiple hardlinks. Native
compiler cache compatibility must be resolved before native release; it cannot
be dismissed as harmless or waived by this review. No monitor repair was made.
The own symlink probe is deliberately retained and points to a nonexistent
review-only path; it is not a native artifact. Its target is recorded in the raw
receipt; the SHA256 manifest inventories regular review evidence only.

Successful completion of this diagnostic would support only
PARTIAL_DIAGNOSTIC_F0_F2_EXACT_OBSERVED under the pinned runtime trust boundary.
It would not establish full K2 qualification, unconditional hardware attestation,
speed, BPB, submission readiness, or authorization for a successor.
