# Independent static Verifier receipt: trn22 r4 exact-source capsule

**Verdict: PASS for the offline capsule and non-dispatchable source-stage plan only.**
This is not an authorization to contact trn22, transfer or extract the capsule,
run G2, train, evaluate, alter AWS, clean up, or submit. The package's
`source_transfer_authorized`, `source_checkout_authorized`, `cpu_g2_authorized`,
`native_training_authorized`, `fresh_eval_authorized`, and
`submission_authorized` fields remain `false`. A separately reviewed transfer
writer/command, fresh host feasibility, and a Supervisor assignment are still
required before even a transfer attempt. I did not implement this package or
operate either host.

Review time: `2026-09-25T02:41:56Z` (UTC). Implementer package commit:
`4570f4e87d5e052a6d8e402326e146384759b8cf`, fetched from
`origin/codex/20260925-trn22-r4-source-stage`. Frozen source commit:
`446b657695f086f9b484fa601c9a3b61c2495912`, present at
`origin/codex/20260925-a0-trn22-r4-interlock`. The package adds only the five
files below to parent `09692c4`; `git diff --check` passed and it changes no
scored source, evaluator, guard, launcher, or monitor file.

| Package artifact | SHA-256 |
| --- | --- |
| `GATE_CONTRACT.json` | `79878b16f459d4c12b32cb0ebb85d205dad86d584ae09654e0e8161920deab69` |
| `MEMBERS.jsonl` | `7ec8798c3e6d0284a4514a15a0caeb17bc5eb2cb5df935fb7dd6e8f3aabee42a` |
| `RUNBOOK.md` | `670dfb14ea54d88dd19cea5828457e256cda9244d3e0946248a0988c25960d36` |
| `build_source_capsule.py` | `ece603f289874b93a6c122c4daa59333f5aeb626ca0d9cb5ff5cd7c15782f211` |
| `test_build_source_capsule.py` | `9b51887d7bc58c92f2914cbac57d10464818d12d64138d8b690b780e6396bede` |

The out-of-Git local artifact
`/home/poojith-devan/Desktop/Trainium-source-artifacts/trainium-c5c-r4-446b-one-commit-20260925.tar.gz`
is a regular, single-link, mode-0600 file of `104154875` bytes with SHA-256
`767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`.
The manifest has `1229617` bytes and `6368` entries. All these identities
match the contract. The local builder's full `--verify-local` scan passed, as
did its eight synthetic tests for pin mismatch, unsafe paths, links/devices/PAX,
secret markers, exclusive creation, and absent host-dispatch logic.

## Independent capsule and source checks

I separately scanned the complete tar stream and compared every member with
the manifest: `6368` members, `6305` regular files, `63` directories, four GNU
long-name headers, and `188917148` unpacked regular-file bytes. There were
zero unsafe/duplicate paths, unexpected member types or modes, missing parent
directories, metadata mismatches, or payload-size/SHA mismatches. No archive
member is a symlink, hardlink, device, FIFO, or PAX member. The manifest's
non-`.git` file set and executable modes exactly equal `git ls-tree -rz` at
`446b6576`: `6278` tracked files, zero missing/extra/type/mode mismatches.
An independent `git cat-file --batch` pass hashed all `6278` frozen Git blobs
and found zero size/SHA mismatches against the manifest. Thus the capsule's
working files, including `train.py` and `prepare.py`, are the exact frozen
source rather than the package branch's versions.

The capsule's `.git/HEAD` and `.git/shallow` each identify only
`446b657695f086f9b484fa601c9a3b61c2495912`. The archived configuration
has only core keys; there are no remote refs, alternates, or active hooks.
The builder verifies detached HEAD, a clean one-commit shallow checkout, no
remote/alternates, and `git fsck` in its disposable local extraction. The
independent path scan found no credential directory, secret-like basename,
checkpoint payload extension, data/output top-level, active hook, remote ref,
or alternate. High-confidence key/token marker scans of every uncompressed
tar payload and a second broader marker scan found zero hits. Marker scans
are not a general guarantee that arbitrary text or opaque binaries contain no
sensitive information; they are only the stated static checks.

**Ancestry limitation:** `446b6576` is not an ancestor of package commit
`4570f4e8` (`git merge-base --is-ancestor` returned `1`; their merge base is
`7f9aaaf503ca633a4e4701478316dd945aea0224`). This does not invalidate
the capsule: the pushed frozen-source branch, capsule HEAD/shallow, and all
`6278` blob hashes independently agree. It does mean a later Operator must
never substitute the package checkout/HEAD for the capsule. Exact capsule
SHA, frozen HEAD, and pinned file hashes must be reverified at each stage.

## Target and authority boundaries

The planned archive target is a new private ext4 path
`/home/ubuntu/trainium-c5c-r4-446b-one-commit-20260925.trn22-stage-r1.tar.gz`;
the planned checkout is a distinct new path
`/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r4`.
The plan forbids `/dev/shm` as the transfer target and same-path overwrite,
replacement, deletion, or retry after a partial write. The exact archive name
was not referenced in pre-package tracked research, but **offline review
cannot prove it or the checkout/output/label is absent on trn22**. Host
absence, parent ancestry/ext4, pinned SSH identity, process/Neuron/lock and
STOP/breaker state, checkpoint and six-cache identities, and sufficient free
space/inodes need a fresh read-only preflight immediately before any future
write. Historical `2026-09-24T23:39:25Z` host evidence is explicitly stale.
The pre-transfer floor `13177973911` bytes equals the 10-GiB post-stage floor
plus capsule bytes, unpacked bytes, and a 2-GiB reserve; arithmetic matches.

This package supplies **no** transfer writer, reviewed literal SSH command,
one-shot assignment, Operator receipt, independent transfer review, checkout
assignment, or native authorization. A later separately reviewed transfer must
use exclusive/no-follow creation, exact byte/SHA readback and fsync, then
independent persistence checks in distinct sessions; partial output must be
preserved on failure. A later checkout requires another request, review, and
assignment, plus safe extraction and an independent postflight before G2.
The old G2 result does not by itself grant source transfer or native dispatch.

No SSH, host, AWS, transfer, extraction, G2, native, or evaluation action was
taken in this review. Static PASS leaves every execution gate on HOLD.
