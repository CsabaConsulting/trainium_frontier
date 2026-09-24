# Standing mission v2 — first-place campaign

Human-approved at 2026-09-24 15:30 IST. This supersedes the earlier final-week tuning order through
the 2026-09-29 18:00 IST research freeze. It does not authorize leaderboard submission, AWS instance
lifecycle changes, evaluator changes, external data, or bypassing the repository's proposal,
independent-verification, Operator, and evidence gates.

## Target and evidence status

- Human-reported live leaderboard facts: first `0.9508`, tenth `0.9832`, team best accepted
  `1.0246`. These have not been independently read from the signed-in leaderboard in this checkout.
- Best confirmed local candidate remains SHA-256 `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`,
  with two public-1024 20M evaluations of `0.9920722144854677` and `0.9918353324459732`.
- The primary external recipe is Recursive's Apache-2.0 repository
  `recursive-org/first-steps-toward-automated-ai-research`, pinned locally for review at commit
  `a962ec43e2e3d7c018e59a2ece623fe6e232fdfb`; its
  `nanochat_autoresearch/solutions/optimized_from_karpathy.py` has SHA-256
  `f92ecc814039ff177b3bfd23c7967918bc44e88765572923277bfa40b7fb597d` and reports a ten-seed mean
  BPB of `0.9109` in its own B200/2048-context harness. That number is a direction prior, not a
  transferable Trainium result.

## Active order

1. Finish the already-launched clean `116188cf...` Trn22 control. It is submission-safety evidence,
   not the Track-A `35c9...` research control.
2. Establish one seed-42, 1,800-second, fresh-2M control per host using exact source
   `submissions/dense6x1024_public0992851/train.py`, SHA-256
   `35c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2`, with the dense launch
   flags plus `--matrix-lr 0.015` and `--no-ngram-ve`.
3. First scientific arm: last-layer bigram+trigram value tables with explicit
   `--ngram-ve --ngram-ve-layers=-1 --ngram-ve-table-mult 64 --ngram-ve-dim 128
   --ngram-ve-trigram --ngram-ve-opt rmsprop --ngram-ve-flat-lr --ngram-ve-bf16
   --ngram-ve-clip-exclude`.
4. Second arm: the same table configuration on layers `1,-1`. If it is promising, compare the
   legal LNC1/world8 shape only against an LNC1/world8 control with otherwise identical bytes and
   flags.
5. Then screen the available constant/flag mechanisms: MLP sandwich norm; ReLU2 tau `0.5`; head
   gate plus per-head norm; embedding LR `0.45`, then `0.6`; warmdown `0.9`; x0 gate and out-pool as
   separate causal arms. Depth-8/width-768 is eligible only as a declared stack confirmation with
   already-confirmed table changes.
6. In parallel, implement and independently verify a touched-row-only synchronized n-gram RMSProp
   path. Do not claim it viable until duplicate-index, cross-rank AVG equivalence, BF16/FP32-state,
   clipping, checkpoint, causal-eval, complete-step, and fixed-budget BPB gates pass.

## Exact flag audit

The `35c9...` source exposes flags for n-gram VE placement/size/dimension/optimizer, matrix and
embedding LR, warmdown, MLP sandwich norm, shifted ReLU2, head gate/norm, x0 gate, out-pool,
depth, and width. It does **not** expose CLI flags for `WINDOW_PATTERN`, `SHORT_WINDOW_FRAC`, or
`ROPE_BASE`. Therefore the proposed window and RoPE arms are skipped under the human instruction
to skip nonexistent flags; they are not silently converted into code edits.

## Selection and scheduling

- A fresh 2,097,152-token public-1024 evaluation is a screen only. Delta `<= -0.001` nominates a
  cross-host 20M replicate. A 2M result never establishes KEEP or adoption.
- Adopt only after two 20M runs are both negative with mean delta `<= -0.001` against current,
  same-host controls. Re-run one control per host after changing the base.
- One run per host, unique labels, seed 42, full chip, 1,800 seconds, exact archived logs and
  checkpoints. The 6.5-GiB launch floor and approximately-5-GiB hard safety floor remain.
- When a host frees, schedule: nominated replicate; new-base control; next table/Track-A arm;
  Track-B run; clean-file validation; Researcher proposal; base replicate.
- Alert the human for milestone/adoption/submission-ready/correctness/disk/breaker events and host
  idle longer than 15 minutes. Submission and AWS lifecycle/spend remain human-only.

## Milestones

- Public-1024 `<= 0.980` by Sep 25 23:00 IST.
- Public-1024 `<= 0.975` by Sep 26 23:00 IST; prepare S1 for human submission.
- Public-1024 `<= 0.960` by Sep 28 12:00 IST; prepare S2 for human submission.
- Public-1024 `<= 0.945` by Sep 29 18:00 IST freeze.

These are campaign targets, not forecasts. Any unexpectedly large gain, especially below `0.97`,
requires an independent cold replay and correctness audit before it is reported as real.
