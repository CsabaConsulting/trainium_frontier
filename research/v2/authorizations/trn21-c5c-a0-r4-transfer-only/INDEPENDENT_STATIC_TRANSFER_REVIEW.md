# Independent trn21 r4 transfer review — static PASS, dispatch HOLD

Reviewed 2026-09-25 UTC by a Verifier separate from the Implementer. Exact
Implementer commit: `0e3802d658ae6c7f89a1689e3c4a8c697942e071`, branch
`codex/20260925-trn21-r4-transfer-disabled`. I read the repository contract,
current handoff, experiment protocol and decision log; the trn21 446b source
stage HOLD plan; and the full six-file transfer implementation diff and actual
bytes. The prior trn22 capsule review is source-artifact evidence only, not a
trn21 host or dispatch approval.

**Verdict: PASS for this disabled transfer-only package's offline static
safety gates. DISPATCH HOLD.** `TRANSFER_REQUEST.HOLD.json` is hash-pinned with
`dispatch_authorized: false`, and `run_once()` rejects it before the one-shot
marker, journal or SSH. The fixed independent enablement-review and Supervisor
assignment files are absent. This receipt is not either of those files and
does not authorize contact with trn21, transfer, extraction, checkout, G2,
native training, evaluation, cleanup, or submission. I performed none of
those actions and did not inspect private-key contents.

## Frozen bytes and host identity

| Reviewed artifact | SHA-256 |
| --- | --- |
| Disabled transfer request | `5af3241bf07ee3275685f1f8881be1a297e36e4542c695149269f2cdc2c1079b` |
| Operator wrapper | `81f69b8f15837fd34aced5463c605c2a91936a36330a33ca6fc32f9fcc6e4621` |
| SSH transport and validators | `fac4a840a43d7addb0ee3579330fbd02b5b49d759b4e7e7b90ee904e14a01647` |
| Remote probe/writer/reader | `be2e0aca21aaf5e29b330c5c0b004dd8a013e99c2aaaa29e820be53b0a15dd68` |
| Transfer runbook | `0c49002d9238c371e27b392b382b81cb57aac9ff03a1ecf6a1aa6b9a07cb4c03` |
| Supplied transfer tests | `1472accd6bc50cb8f94e94b890e7f3b45484e7e69dcecd6e54fe6b336827eb4b` |
| Independent adversarial tests accompanying this review | `fb0ddcf01ffc8acd36a7b1a9479f12bc9ad860fecd63e4eda8b78a64e2d80d11` |
| Frozen trn21 source-stage HOLD contract | `2f7dca0e1e07543e8b2ad7ea2c20241d6841b8ea3524e01c1c3f3028730b0219` |
| Earlier local capsule review, not trn21 authority | `d49e8c03567210283074c44a2f2937594ada39f43e399f97d176c373c1a32c1e` |

The sole local capsule for detached source commit
`446b657695f086f9b484fa601c9a3b61c2495912` is a regular, owner-1000,
single-link mode-0600 file of 104,154,875 bytes and SHA-256
`767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`.
The only possible remote write is the new
`/home/ubuntu/trainium-c5c-r4-446b-one-commit-20260925.trn21-stage-r1.tar.gz`
on the pinned trn21 root ext4. The existing e65 checkout is protected at
device/inode `66305/3168319`; two checkpoint size/SHA/inodes, all three
earlier stage files, the e65 HEAD/train/prepare bytes, and six warm-cache
directory inodes are gated in fresh probes and compared across preflight,
writer and postflight.

The literal endpoint is
`ubuntu@ec2-16-26-246-98.ap-southeast-4.compute.amazonaws.com`, with remote
hostname `ip-172-31-34-85` and UID 1000. The local `known_hosts` SHA-256 is
`e42405d8dd82c56be05b849dbe1d050dafe2ebf2678ab197857174b44057cce3`;
the sole expected ED25519 fingerprint is
`SHA256:G4eS9Bib/yRt+bpWWOUV0SgF/pqTfhLITafrA2ToCfc`. I checked these locally
without opening SSH or reading the PEM. The SSH argv fixes `-F /dev/null`,
strict host-key checking, no global known-hosts file, one pinned identity,
and only `probe`, `write` or `verify` operations. Remote identity validates
the ext4 `/dev/root` `259:1` mount, kernel block alias
`/dev/nvme0n1p1`, mount namespace, boot and safe ancestor inodes.

## Adversarial result

The separate preflight's stable identity digest is required by the writer.
The writer runs its own full protected-asset/quiescence probe, rejects boot,
mount or parent drift, then checks all proposed-target absences, STOP,
process, checkpoint-holder, shared-lock and Neuron state immediately before
the one descriptor-relative `O_CREAT|O_EXCL|O_NOFOLLOW` mode-0600 create.
Cheap target/STOP/process checks repeat after potentially slow holder
commands. Local comparisons require a distinct writer process, matching
preflight/writer identities and protected inventories, distinct read-only
verification with exact byte/SHA/inode agreement, and a final read-only
postflight even after failure. A partial target is preserved; the fixed
marker and separate raw journal each consume one exclusive attempt.

The Implementer's 15 transfer tests and five source-stage HOLD-plan tests
passed. My nine independent tests passed, including pre/post boot A with
writer boot or mount B; remote boot mismatch before parent open; STOP or
matched process appearing during holder checks; a busy Neuron device during
the immediate gate; protected e65 checkout, prior stage, checkpoint and
cache drift; exact target success/partial preservation and no overwrite;
private exclusive marker/journal; strict trn21 host-key rejection; disabled
no-dispatch; and failed-write postflight with both HOLD events. Combined
offline run: **29/29 PASS**. `git diff --check 0e3802d6^ 0e3802d6`
also passed. The frozen 446b Git object was not initially present in this
local object store; I fetched that exact commit read-only, then all five
source-stage tests passed. No Implementer code was changed.

Static tests do not prove that the current host is idle, that target paths
are absent, that protected files remain intact, or that a future SSH
invocation succeeds. The historical 03:17 UTC checkout postflight is not a
fresh transfer preflight. Any future enabled request must bind newly
reviewed exact bytes and require a separate independent enablement review,
fresh read-only host feasibility evidence, and a distinct Supervisor
one-attempt transfer-only assignment. Until then, **no dispatch**.
