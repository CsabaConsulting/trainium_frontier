# Track-A A1 hybrid ownership repair

Preparation only. This directory contains no executable authorization or request. It does not
authorize SSH, a native gate, full training, cleanup, submission, or reuse of the already-authorized
A1 r2 request. That request pins the old `35c9…` train bytes and must fail against this candidate.

The only candidate edit is inside `make_plan` / `OwnedOptimizer` in
`submissions/dense6x1024_public0992851/train.py`. Candidate SHA256 is
`9191e85835d8cad50923f32d44ad90c047a9e2d3d4621ce196b11ba66cd66bae`; unchanged `prepare.py` is
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.

Hybrid mid-run optimizer serialization and resume are deliberately unsupported. Both operations
fail before the raw optimizer methods with the exact error `optimizer state serialization/resume
is unsupported with replicated RMSProp ownership; abort and restart from zero`. An interruption
therefore aborts the run and requires a new unique-label run from step zero. Production `final.pt`
is model-only and contains no optimizer key; the strict model load/evaluation path is unchanged.

Before any native action, an independent Verifier must review this commit and finalize a new exact
symbolic-branch request using the existing fresh-start launcher/guard protections: clean and
origin-aligned HEAD, exact source maps, no-follow regular files, O_EXCL label/remote/output paths,
6.5 GiB launch floor and hard abort below 5 GiB, private process group, bounded TERM then KILL,
quiescence before artifact hashing, immutable logs/manifests, and clean postflight. The first and
only initially eligible native action is:

```text
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --num-steps 5 --n-embd 1024 --ngram-ve --ngram-ve-layers=-1 --ngram-ve-table-mult 64 --ngram-ve-dim 128 --ngram-ve-trigram --ngram-ve-opt rmsprop --ngram-ve-flat-lr --ngram-ve-bf16 --ngram-ve-clip-exclude --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015
```

It requires the unique label `fs-20260924-v2-a1-hybrid-rmsprop-native5-trn21-r1`. A PASS requires
five completed finite-loss steps, a finite model-only checkpoint with no optimizer serialization,
an ownership receipt
showing RMSProp absent from owner entries and updated on every rank, exact rank agreement, and a
clean quiescent postflight. Failure evidence must be archived; retries require a new label.

Only after that PASS is archived and independently reviewed may a separate authorization be made
for label `fs-20260924-v2-a1-hybrid-rmsprop-full-trn21-r1`. Its command is the exact A1 command in
`EXPERIMENT_PROPOSAL.json` without `--num-steps 5`. The full run remains fresh-2M screening only:
it may nominate but cannot establish KEEP or adoption. Both request identities remain pending.

Local implementation gate:

```text
/home/poojith-devan/venv/bin/python -m unittest -v \
  research/submission-validation/20260924-v2-a1-hybrid-rmsprop-ownership-repair/test_hybrid_ownership.py
```
