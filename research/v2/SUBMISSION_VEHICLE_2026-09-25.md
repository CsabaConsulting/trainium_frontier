# Scored submission vehicle: sub-452d2ae4

Human leaderboard report (2026-09-25 IST): submission `sub-452d2ae4`, uploaded
2026-09-24 at 13:56 IST, is **SCORED 0.9901 official**. It passed the
organizer's review and clean compilation. Treat this already-scored submission
as the review-proven vehicle for porting future measured improvements. Do not
re-upload it; submissions remain human-only.

The local upload file `/home/poojith-devan/Downloads/train.py` is 85,557 bytes,
modified 2026-09-24 13:55:19 IST, immediately before the reported upload. Its
SHA-256 is identical to the preserved repository package
`submissions/dense6x1024_direct_review_matrixlr015/train.py`:

```text
116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d
```

The exact launch command recorded by that package in `launch-command.txt` and
`provenance.json` is:

```text
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public
```

The file hash and command are exact for the preserved package and local upload
file. The mapping from that local file to `sub-452d2ae4` is strongly supported
by the 41-second upload timing and the human's scored-submission report, but
the organizer-side uploaded bytes and form command have not been independently
downloaded or rehashed. If the portal exposes those, bind them here before
claiming cryptographic portal-side identity.

Research remains on exact `c5c410af...` source for A1/A2 table experiments and
same-host c5c controls. Port only independently validated gains into a new
review-clean derivative of the 116188 vehicle, preserving the submission's
honest train/eval path; cold replay/audit any port before asking the human to
upload it. A research run or public BPB screen alone is not an official score.

Per the newer human instruction, the S0 **full** replay is cancelled. Only
the two already-running 120-second compile/startup diagnostics finish. A2
exact-size CPU G2/native gates take the first free host; A1 the other. The
three earlier uploads are currently reported **QUEUED** for human review, not
rejections, and must not be resubmitted.
