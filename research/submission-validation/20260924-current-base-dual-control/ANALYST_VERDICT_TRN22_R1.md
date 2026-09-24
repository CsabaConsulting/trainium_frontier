# Independent Analyst verdict: trn22 current-base control r1

Date: 2026-09-24

Role: Analyst only. This report performs no source change, remote action, cleanup, retry,
hardware launch, push, or submission.

## Evidence audited

- Evidence commit: `68ecfeb168ddd095ed14f1908fee7c0284dbc97f`
- Label: `fs-20260924-current-base-control-trn22-r1`
- Candidate SHA-256: `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`
- Prepare SHA-256: `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`
- Checkpoint SHA-256: `1b1d1a3f0ec6b5580312b760f5fee4f702c3fe97c9ddd01ed811472736aadf1b`
- Artifact manifest SHA-256: `456c179ade10395dba16b898d91af7db54450989a0e35b5bcfbdc6694d971180`
- Completion receipt SHA-256: `07f9e57793fc79c87c046df3597ac8da2160dec51b2b33233701c223fc16fc77`
- Evidence archive SHA-256: `ad7787c633e8f64c096ca841bdbf821bb5586ec9c05b25e77b0471b595acbb39`
- Dispatch receipt SHA-256: `c7ef6fedb81bb80e7923e76ea8957448caec3bbe766f083e43a4866c8a32f5d6`
- Postflight SHA-256: `04f6beeb7b17a88146f95260de0f53c035aa7d8dd29ab22e642a9a8af7661429`

All locally retrieved manifest member hashes matched. The archive contained the 12 payload
members declared by the manifest apart from the archive's own manifest entry, and every archived
payload hash matched its retrieved file. The checkpoint was unchanged through both evaluations and
archive creation, finite, loaded with `weights_only=True`, and contained 62 tensors.

Training completed without crash, NaN, timeout, or budget violation. It produced 3,631 step
boundaries, versus expected 3,638, and charged 1,789.257 seconds through process exit. The exact
fresh-process evaluations were:

- 2,097,152 tokens / 8,418,678 bytes: `0.9873339212252918` BPB.
- 20,971,520 tokens / 83,462,186 bytes: `0.992968300659387` BPB.

## Scientific adjudication

The completed measurements are authentic and scientifically useful as an observation of the exact
current base on trn22. Step count is only four below the first trn21 run (-0.110%) and 23 below the
second (-0.629%), well inside the 10% comparability rule.

The trn22 full20M value is worse than the authenticated trn21 values by:

- `+0.0008960861739193` versus `0.9920722144854677`.
- `+0.0011329682134138` versus `0.9918353324459732`.
- `+0.00101452719366655` versus their mean `0.99195377346572045`.

Because source, seed, training budget, launch, and evaluation protocol are unchanged while the host
and run instance differ, these deltas measure cross-host/run variation; they are not evidence of a
causal regression. The result must not be pooled with trn21 or substituted for a trn21 control.

## Protocol adjudication

**Verdict: INVALID_CONTROL_FOR_ADVANCEMENT / FAIL_POSTFLIGHT_OPERATIONAL_DISK_FLOOR.**

Postflight recorded 5,928,337,408 free bytes, below the verified 6.5 GiB operational floor of
6,979,321,856 bytes, although it remained above the 5 GiB hard-abort floor and the host was idle and
unheld. The controlling independent-verifier receipts explicitly state that any pre/postflight
failure invalidates that host control. The Operator receipt accordingly sets
`analyst_advancement_authorized=false`.

Therefore:

- No Analyst advancement, candidate nomination, trn22 arm comparison, or next launch is authorized
  from r1.
- The r1 full20M and 2M values may be retained for descriptive variance and planning only. In
  particular, `0.9873339212252918` is not an advancement-eligible trn22 screening baseline.
- A later cleanup receipt cannot retroactively turn r1 into a protocol-valid control. A new control
  run is required.

## Exact recovery boundary

The recorded postflight floor deficit was 1,050,984,448 bytes (0.978806 GiB); the latest recorded
audit deficit was 1,051,017,216 bytes. The run consumed 1,854,160,896 bytes of free space between
preflight and postflight. Merely restoring exactly 6.5 GiB before another run is therefore not
adequate evidence that its postflight will pass. Under an unchanged footprint, post-staging
preflight needs at least 8,833,482,752 free bytes (8.226822 GiB), which would require recovering
2,905,178,112 bytes from the latest recorded state. Live inventory controls if the host has changed.

Before a retry:

1. Obtain separate authorization for Operator cleanup and retry; the r1 verifier receipt authorized
   neither.
2. Inventory and remove only exact, archived/reconstructable, authorized targets. Preserve the r1
   checkpoint and sealed evidence; checkpoint pruning and cache cleanup remain unauthorized unless
   separately granted.
3. Record a durable cleanup receipt proving the exact targets, reclaimed bytes, no protected
   artifact loss, idle processes, and no Neuron holder.
4. Use a new unique r2 label, checkout, output directory, request, and evidence namespace. Preserve
   r1 unchanged.
5. Independently verify the retry request, then pass a fresh post-staging preflight with the exact
   candidate/prepare hashes, seed 42, four ranks/LNC2, compiler flags, 1,800-second budget, clean
   source, idle device, and sufficient disk headroom.
6. Repeat both protected evaluations and require a finite immutable checkpoint, sealed locally
   hash-verified evidence, and a postflight with at least 6,979,321,856 free bytes.

Only that completed retry can establish an advancement-eligible same-host trn22 control.
