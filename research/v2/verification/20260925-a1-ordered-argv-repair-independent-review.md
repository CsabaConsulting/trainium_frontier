# Independent static review: A1 ordered-argv repair

Scope: controller commit `888d45fe` (original Implementer `a3714126`),
`research/submission-validation/20260925-v2-c5c-a1-m64d128-trn21-r1/`.
This reviews the repaired packet only, not a host, CPU G2 run, native smoke,
full run, result, or submission.

**Static verdict: PASS for the narrow ordered-argv repair. Native dispatch:
HOLD.**

- I compared the exact `research/v2/queue.json` A1 `argv_delta` to the revised
  `gate-contract.json` and proposal. The packet now literally replaces the
  single `--no-ngram-ve` in the c5c A0 control argv **in place**, after
  `--n-embd 1024`; it does not append the table flags after matrix LR. The
  ordered A1 flags are `--ngram-ve --ngram-ve-layers=-1
  --ngram-ve-table-mult 64 --ngram-ve-dim 128 --ngram-ve-trigram
  --ngram-ve-opt rmsprop --ngram-ve-flat-lr --ngram-ve-bf16
  --ngram-ve-clip-exclude`. Other control flags remain in their original
  order and `--matrix-lr 0.015` remains the final pair.
- The revised non-host `test_static.py` rejects the old append order and
  checks the proposal's parsed launch command against the contract. I ran it
  in a clean isolated checkout at `888d45fe`; it printed
  `PASS: exact c5c source, A1 queue flags, two-table dimensions and no-launch gates`.
  The queue hash is
  `e4dbafc9132a7046499df46eea869b57b213c8e338f3d2da05c2ca53b1d1ea30`;
  `submissions/research_v2/train.py` remains c5c SHA-256
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`;
  `prepare.py` remains SHA-256
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  `git diff --check 888d45fe^ 888d45fe` passed. The commit makes no model,
  evaluator, tokenizer, or decision-engine source edit.
- The former A1 `verifier.json` covered the **pre-repair packet** and cannot
  be inherited as approval of the new bytes. This receipt is the independent
  static review of the repaired order only. The added six-per-host queue is
  explicitly dependency planning, not six authorized jobs.

Before A1's **full 1800 s** trn21 run: exact c5c/m64d128 four-rank CPU G2
with per-rank RMSProp trace and approved memory/disk headroom; separate native
30-step smoke; an accepted same-host c5c A0 control with fresh-process
public-20M/context-1024 score; then a one-label one-attempt Supervisor
assignment and fresh host protocol preflight. `dispatch_authorized:false`,
empty gate receipts, and unbound same-host control in the revised packet are
correct. No A1 BPB gain is established.
