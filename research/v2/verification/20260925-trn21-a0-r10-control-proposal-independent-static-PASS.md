# Independent trn21 r10 A0 control proposal review — STATIC PASS only

**Verdict:** the Researcher proposal is complete and internally consistent as
a **plan** for one trn21 c5c A0 control. This is not Supervisor ratification,
an enabled native request, host preflight, launch permission, an accepted
control, or a BPB result. I made no host contact and changed no proposal or
training source.

## Exact subject and template

- Subject: `origin/codex/20260925-trn21-a0-r10-control-proposal-r1`
  at `96fcec98a538673772febed826a653a4478cd9b0`, adding only
  `research/submission-validation/20260925-v2-c5c-a0-control-trn21-r10/EXPERIMENT_PROPOSAL.json`.
  Its SHA-256 is
  `1f616cde93b726553684b13c4c0ac2081e51bc289d90fdaff2d7ac18920b7e9d`.
- The proposal has all 24 top-level fields and all required nested
  `candidate`, `control`, `gates`, and `success_criteria` fields from
  `research/agentic/EXPERIMENT_PROPOSAL.json`; it contains four ranked
  alternatives, a falsifier, duplicate search, prior results, aborts,
  retrieval, and explicit host/evaluator gates. It is Researcher-owned and
  remains `DRAFT_COMPLETE_PENDING_INDEPENDENT_PROPOSAL_REVIEW_AND_SUPERVISOR_RATIFICATION`:
  every listed gate is false, ratification/human approver/verdict are null,
  and launch/training/new-BPB flags are false.
- `proposal_id=20260925-v2-c5c-a0-control-trn21-r10` is deliberately
  different from the frozen science label
  `fs-20260925-v2-c5c-a0-control-trn21-r1`. The historical conditional
  ratification SHA-256
  `c483e2b00653ede82226a73cd9d9b7f3b01fb2988828c1f749f66e416641551d`
  binds the **old** proposal ID and SHA-256
  `55c6d500fbc535a45ad3afb6ef32882e30726b8ec90ac412e2ba987b5c71c1b9`;
  it is not this proposal's selection or r10 dispatch authority. The old
  e65 native request skeleton SHA-256
  `abe50c7b4ab64e1eadbfd12f997349c584adadf7813632895f55d0b89d336f2d`
  has `dispatch_authorized=false` and `command=null`.

## Source, launch and evaluation pins

- The proposed scored source is commit
  `22bfbab5a77d6feb670d2d737504f8b70b08ddc1` with `train.py` SHA-256
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
  and unchanged `prepare.py` SHA-256
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  The documentation branch's root `train.py` is instead
  `52ac9e6560a40592255e3499f4d42ac6e7073ea82da1b428e03dc984e6d4baa8`;
  it **cannot** be the scored-source execution root. The proposal correctly
  requires a separately accepted clean r10 checkout on both hosts and local
  same-HEAD verification.
- Both the local and embedded-remote *pure* exact-command validators in
  `22bfbab5:research/fresh_start_launch.py` accept the proposal's full
  ordered 43-word `launch_argv`, effective environment, trn21 r10 checkout,
  and frozen r1 science label, while rejecting a substituted r10 science
  label. The child is full-chip LNC=2, `torchrun --nproc_per_node=4`, seed 42,
  1,800 training seconds, width 1024 with n-gram VE off and matrix LR 0.015;
  the six warm-cache paths and BF16 compiler flags exactly match the frozen
  validator. The separate source-stage label is not a native training label.
  The launcher source has source-only PASS SHA-256
  `5e6fbfe6fd46608d0b220b99402185c78c809523026df7153882d3637de20c96`,
  not native dispatch approval.
- The proposal's **separate, unauthorised** evaluation lane specifies a fresh
  `prepare.py eval-public` process at batch 1, context 1024, exactly
  20,971,520 public tokens and expected 83,462,186 bytes, with a 1,200-second
  evaluator timeout. The actual `prepare.py` default is only 2,097,152
  tokens, so a later reviewed evaluation request must explicitly pass
  `--eval-tokens 20971520`; this proposal alone does not do so or evaluate.

## Duplicate and safety disposition

Targeted searches of `research/experiment-log.jsonl`, `research/v2/ledger.jsonl`,
and `research/trn2-experiment-results.md` at the subject commit found no
native training/result entry for the frozen trn21 r1 label or the new r10
proposal ID. Other files mentioning that label are old planning, source-stage,
read-only, or CPU G2 evidence. This establishes **no recorded local native
duplicate**, not live-host absence. The proposal properly requires the
Operator to prove fresh absence of the label, output, step log, marker,
request/monitor artifacts and conflicting host matches under the shared lock.

The proposal keeps source staging and its result review, an r10-specific G2
source-identity disposition or new G2, new Supervisor ratification, a distinct
SHA-bound native request/binding and independent review, one-attempt
assignment, both-host clean 22b HEAD, host/Neuron/process/STOP checks, disk
reserve, paired monitors and boundary observer as hard prerequisites. Root
free space must be at least 10,737,418,240 bytes before native launch; the
operational floor and monitor stop threshold remain separately enforced.
The old e65 G2 numeric PASS is useful prior evidence only. None of these
gates is discharged by this static proposal review. A valid future
public-20M result would be a same-host **control**, not an improvement or
adoption on its own.
