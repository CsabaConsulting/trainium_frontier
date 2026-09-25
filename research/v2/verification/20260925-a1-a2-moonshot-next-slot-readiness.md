# Independent next-slot readiness: c5c A1/trn21 and A2/trn22

Reviewed 2026-09-25 13:26 UTC in a separate clean worktree. The reviewer did
not prepare either arm, contact either host, run CPU G2, launch Neuron work,
evaluate a checkpoint, edit model/evaluator bytes, or change a decision rule.
This is a **static readiness review, not training authority or a BPB result**.

## Verdict

| Arm | Proposal/static state | Full-run state |
| --- | --- | --- |
| A1/trn21, last-layer b5/t5 | **HOLD**: the packet's ordered argv is not the queue's literal in-place replacement; repair the packet and independently check the repaired bytes. | **HOLD**: exact-size CPU G2, native smoke, valid same-host A0, and fresh host/disk gates remain. |
| A2/trn22, layers 1,5 b1/t1/b5/t5 | **PASS for the previously reviewed static proposal** (`research/agentic/verification/20260925-v2-c5c-a2-m64d128-trn22-r1-static-r2-review.md`); its ordered argv is the queue's in-place replacement. | **HOLD**: exact-size CPU G2, native smoke, valid same-host A0, and fresh host/disk gates remain. |

The human's later parallel-scheduling amendment permits A1 on trn21 and A2
on trn22 as *separate profiles against same-host A0* when their gates pass.
It does not make one host's control valid for the other. A2's layer-1
increment cannot be attributed without a same-host A1 comparison.

## Static evidence and one new A1 defect

- Both `workflow/proposals/research-v2/train.py` and
  `submissions/research_v2/train.py` hash to
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.
  `prepare.py` is `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`;
  queue is `1e9dcc4fd5f38e62829dc19460d2337662184ecfbd4d21d8a20f15993be9dd00`;
  G2 tool is `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`;
  `decide.py` is `a47bedd258697b0791f48dcac3098d9071b5aaf00a7664d9e39987b4052adc66`.
- Both proposal JSON files parse. Their one predeclared causal factor is the
  no-table A0-to-table-profile replacement; the predicted BPB deltas are
  priors, not measured improvements. The v2 ledger has only the earlier
  invalid trn22 A0 zero-step disk incident; the older 35c9 A1 died before
  step 1 at optimizer ownership. Neither is a c5c/m64 table score.
- `python3 -B research/submission-validation/20260925-v2-c5c-a1-m64d128-trn21-r1/test_static.py`
  printed PASS, but the test encodes the packet's **append** form:
  `[base without --no-ngram-ve] + A1 flags`. The queue declares
  `argv_delta.replace: --no-ngram-ve`, and the corresponding A2 repair was
  expressly required to splice its flags **at that token's original position**.
  The A1 packet (`gate-contract.json` SHA-256
  `44111219a3c25a7c58599ae8e047f94019e2bfa8e0298c2a86b54c44d5a16743`)
  instead places `--ngram-ve ... --ngram-ve-clip-exclude` after
  `--matrix-lr 0.015`. This may parse equivalently, but it is not the
  ordered queue request it claims to be. An Implementer should change the
  A1 contract, proposal command, runbook/G2 command and static test to the
  in-place form, then a different Verifier should check their exact bytes.
- The A2 contract (`GATE_CONTRACT.json` SHA-256
  `a40723151099e0fec912d670b702ef9f2774fe14853025660a9ffc116d2fc364`)
  already places its table flags immediately after `--n-embd 1024` and
  preserves the rest of the dense argv. It expects four tables, each
  524288x128 BF16, with RMSProp and the unchanged optlevel-1 compiler policy.
- Prior A2 exact-size CPU-G2 work is **not a G2 PASS**. The r3 package had
  disabled-only static PASS; the r4 final-pending package was held because
  it hardcoded tools inside an e65 checkout whose Git tree contained no
  such paths. A later read-only tool-path repair had disabled-only static
  PASS, not host proof. This does not require the old cross-process
  mount-identity comparison: the human explicitly relaxed that staging
  check, and a fresh plain-transfer path may bind exact source/tool SHA
  instead.

## Minimum remaining gates, in execution order

1. Finish both already-running A0/S0 jobs, retrieve logs/checkpoints, run
   fresh-process public-20M/context-1024 evaluation, and have a separate
   reviewer validate receipts **before using their scores**. For trn21 A1,
   run/score an exact c5c A0 on trn21: the S0 clean file has a different
   SHA and cannot serve as a same-base control under unchanged `decide.py`,
   even if its numerics agree. For trn22 A2, require a valid c5c A0 on
   trn22. Bind label, source hash,
   seed, compiler flags, steps and score; historical 35c9/116188 or zero-step
   trn22 A0 cannot substitute.
2. Repair A1's ordered-argv packet. On an **idle**, memory-qualified machine,
   run each arm's exact c5c/m64/d128, two-step, four-rank CPU G2 with
   `--trace-rmsprop`, real unchanged `prepare.py` import, a verified local
   temporary root, and clean process quiescence. Independently inspect all
   four raw rank records: A1 b5/t5 or A2 b1/t1/b5/t5; two RMSProp group
   calls per rank; respectively four or eight table checks per rank; each
   table changed; per-rank agreement; finite loss/gradient/recomputation;
   untouched rows exact; strict checkpoint load; normalized max error
   <=0.03. Aggregate `PASS` alone is insufficient. CPU G2 is not native
   correctness or BPB evidence.
3. On each host, only after its G2 PASS and renewed preflight, run a distinct
   30-step full-chip Neuron smoke on the **same source, table argv, seed and
   compiler policy**. Require all 30 ordered optimizer boundaries, finite
   loss, expected table/optimizer integration, no rank divergence, clean
   checkpoint/strict load, no device holders, and measured disk/HBM/startup.
   A smoke is not a scored quality result.
4. Before a full 1800-second arm, use a new unique label and fresh Operator
   checks: exact `train.py` transferred to a new location and host SHA-256
   verified; unchanged trusted `prepare.py`; exact effective seed 42,
   `NEURON_CC_FLAGS=--optlevel=1 --auto-cast matmult --auto-cast-type bf16`,
   LNC2/four ranks; no other Neuron job; >=6.5 GiB root free **plus a
   conservative table/cache/checkpoint projection**; no STOP/breaker; label
   and output paths unused. Do not use the old cross-process mount-identity
   check as a blocker. The human's break-glass pre-launch-review waiver was
   specific to S0 and A0, not A1/A2. After the run, fresh-process public-20M
   evaluation, retrieved log/checkpoint and ledger/verdict are mandatory.

Local G2 was **not** attempted: this controller has 14 GiB total RAM,
7.5 GiB available at review, and neither default Python nor `.agent-venv`
can import `torch`. The full-size four-rank table G2 should not be attempted
here or on a currently training host. No new score or table result follows
from this review.
