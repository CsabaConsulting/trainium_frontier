# Exact two historical checkpoints: absent

One bounded metadata-only read on trn22 at 2026-09-26 08:12:27.834113 UTC,
under `research/v2/assignments/20260926-old640-discarded-checkpoints.md`.

Both exact assigned targets are absent. Opening the named experiment-root
directory under `/home/ubuntu/workspace/autoloop_runs` returned ENOENT for each:

- `9a4caa7b--20260922-ngram-rms-fp32-only-default-shape--al-20260922-ngram-rms-fp32-only-default-shape-smoke-r1`
- `9a4caa7b--20260922-nki-local-conv-only-default-shape--al-20260922-nki-local-conv-only-default-shape-smoke-r1`

No contents were read, no large hashing was performed, and nothing was deleted,
staged, signalled or changed on the host. The active control was not polled or
disturbed. Directory traversal used no-follow directory descriptors; the raw
receipt preserves the exact existing parent identities and literal errors.
Historical scores/verdicts were not reclassified or represented as current20M.

Observed free space during the live control's compilation was 16,387,891,200 B.
Against the unchanged K2 whole-pair threshold 17,716,740,096 B, this is a current
gap of 1,328,848,896 B, not a projection or admission result at job completion.
These absent checkpoints offer no reclamation. The K2 deadline remains at risk
unless another existing and genuinely eligible already-scored discarded artifact
is found. Do not lower reserves or delete protected/unscored/shared artifacts.

Raw receipt `trn22-old640-checkpoint-metadata.json` SHA256:
`e1aa2c69cec00c1d50bcc531ff4079cc39e1b4d1d4ba0118b3ff4f888b1a433f`.
Collector: `inspect_old640_metadata.py`. No repeated host search is authorized
by this result.
