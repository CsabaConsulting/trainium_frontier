# Independent proposal review — trn22 c5c A0 r10

Reviewed 2026-09-25 11:10 UTC by a Verifier independent of the Researcher,
Implementer, Operator, and Supervisor for this proposed run. The immutable
subject is Researcher commit `1cf9a2907df7c8d8eeb7e5c3a2788a3cc2b612f9`,
which adds only
`research/submission-validation/20260925-v2-c5c-a0-control-trn22-r10/EXPERIMENT_PROPOSAL.json`;
that file's SHA-256 is
`0a593016af6c06b72c91e1757ac19d8863c9414d28d5630dcc365bafd2173aaf`.

**Verdict: PASS for proposal completeness and exact offline bindings only.**
The proposed r10 label is not Supervisor-ratified, the trn22 r10 source stage
is not accepted, and no final r10 G2, enabled native request, one-attempt
assignment, native run, checkpoint, or fresh public-20M result exists. Each
remains **HOLD**. This receipt neither changes any proposal gate or
`research/v2/state.json` nor grants source-stage, host, native, evaluation,
adoption, or submission authority. I made no host/SSH/AWS contact and did not
implement, stage, ratify, launch, evaluate, clean, or reset anything.

## Proposal and frozen command findings

- The proposal contains all top-level, candidate, gate, and success-criteria
  fields in `research/agentic/EXPERIMENT_PROPOSAL.json`, plus explicit r10
  dependencies. Its selected control is quantified as delta `0.0`, interval
  `[-0.001, 0.001]`, `p_improve=0.0`, and 1.1 estimated host-hours; the
  unmeasured A2/D1 shortlist alternatives are explicitly `null`, not claimed
  outcomes. This is an A0 control proposal, not a legacy autoloop job. The
  matched c5c public-20M control fields are correctly null: the r1 v2 ledger
  row is a disk-stop failure with zero optimizer steps and no score. The old
  trn22 35c9 public-2M result is confined to a historical screen.
- Its unique science label is
  `fs-20260925-v2-c5c-a0-control-trn22-r10`. It differs from the consumed
  r1 label and the r2-only ratified label. The r2 proposal SHA-256
  `2256f5f9a04622b4389523872592a18f3db3c3fbc5107405d4e6a00ede3650aa`
  and Supervisor ratification SHA-256
  `a5de803e0e138c56b10f6cbca39c3c1050f527e8956953ba6f8095649b8f1be1`
  bind e65 and r2, not r10. No old approval is carried across.
- Direct JSON comparisons with disabled native request SHA-256
  `3885415c561649bf4e8fe09e948e2793aec718b483566a4575ed0591d4b0977e`
  and gate contract SHA-256
  `837ed3064d807c45d1174c4fe99af04c9b7fae399e45d9a8d673488a164e883d`
  passed for label, trn22 host/hostname and checkout, frozen commit
  `22bfbab5a77d6feb670d2d737504f8b70b08ddc1`, `train.py` SHA-256
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`,
  and `prepare.py` SHA-256
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  The proposal's entire ordered `launch_argv` equals the disabled request's
  command: absolute bounded `timeout`, `env -i`, native
  `torchrun --standalone --nproc_per_node=4`, and the exact ordered dense
  Track-A flags. Its ten effective-environment keys equal the request's ten:
  LNC `2`, optlevel-1 BF16 compiler flags, seed `42`, 1,800 training seconds,
  and six exact retained trn22 warm-cache paths. The old actual G2 launch
  archive contains those same ten values; the selected Track-A argv also
  matches the old G2 receipt and r2 ratification.
- The proposal keeps the r10 source-stage **prewrite** gate distinct:
  13,177,973,911 free root bytes and 10,000 inodes before material write,
  10,737,418,240 bytes after reconstruction. Its later native prelaunch,
  operational, and paired-monitor stop thresholds match the disabled package
  at 10,737,418,240; 6,979,321,856; and 9,126,805,504 bytes respectively.
  Polling is 0.25 seconds against the one-second maximum. The proposal
  requires two mutually live monitors, their STOP/one-shot interlock, and
  the pinned external optimizer-boundary observer before guard spawn.
- One-run-per-host, clean same Git HEAD locally and on both hosts, fresh
  process/Neuron/lock/disk/cache/checkpoint/label checks, active Desktop
  STOP and breakers, bounded timeout, no dual active launcher, no cold-cache
  fallback, and no inferred cleanup, protected deletion, AWS action, or
  breaker reset are explicit HOLD/abort boundaries. Failures consume the
  attempted label and require a separate diagnosis and new proposal.
  Retrieval requires full logs, monitor/observer/interlock/guard records,
  checkpoint hash, budget, unconditional read-only postflight, quiescence,
  and an independent terminal verdict.
- The separate disabled evaluation plan SHA-256
  `45822174f7bd3842f89e2056cff821826de3e8ec7c81f56c3109e21c44da2ec2`
  matches the proposal: fresh process, unchanged `prepare.py eval-public`,
  context 1,024, 20,971,520 public tokens, 83,462,186 expected bytes,
  1,200-second evaluator timeout within a 1,800-second outer bound.
  `evaluation_authorized=false` and `checkpoint_sha256=null`. A 2M screen,
  the diagnostic 0.99195±0.001 band, or a later A2 comparison cannot by
  itself establish a quality win. Adoption remains human-acknowledged and
  subject to the unchanged `workflow/tools/decide.py` rules.

The disabled package at `8bdc727bf4f72955473bbc19c79272e46e5f8650`
has `dispatch_authorized=false`, all twelve contract gates false, and no
one-shot authorization. Its independent static-only receipt at
`ae71845806f0876700b5b9f8ffc2ff59b2da0999` (SHA-256
`c6f735e10a8a4bc88403beb268d684afd4da1595f0bf59253325a8bef7dc054a`)
tested normal and optimized-Python no-contact refusal and exact-profile
mutation seams. This proposal changes no launcher byte and does not convert
that static result into an enabled request or host preflight.

## Old CPU G2: conditional numeric compatibility, not an r10 G2 PASS

The old trn22 e65 four-rank, three-step CPU G2 Operator receipt SHA-256 is
`c8d73cc447a4fb99c9aba1869f3df47accee550ea2e04d704b9a43cd1bd8e116`;
its independent numeric review SHA-256 is
`e76654c7882c2ede9e382914aebb672b10841a083fc0eb2271e15b39965a1de4`.
All ten files in the archived G2 `SHA256SUMS` verified. Running the pinned
`workflow/tools/preflight_dryrun.py --compare` against the earlier c5c
anchor under normal Python and `-O` returned `identical=true`, equal plans
and losses, zero state-key or parameter-digest differences, and 62 compared
parameter digests. The archived G2 used four ranks, three sequential CPU
optimizer boundaries, 2×256 micro-batches, and a deterministic model-only
checkpoint/load check; it was not native training or a BPB measurement.

Direct Git blob checks at e65 and r10 give the identical c5c `train.py` and
unchanged `prepare.py` SHA-256 values above; `git diff --exit-code` on those
two paths also passed for e65→r10 and 446b→r10. The intervening production
Python edits are operational guard/host/launcher/interlock/monitor modules,
not imports of the tested science files. The exact ordered Track-A vector,
four science settings, world size, and all six cache **path values** match
both the r10 proposal and the archived old G2 launch (archive SHA-256
`483152e6b785e1c26be266cf0775391f579f24dc3efc17c2ce7b4e1b8641bb22`).
The old CPU harness and containment-tool SHA-256 values were
`6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`
and `0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d`,
matching repository copies today; they are separately staged tools, not
Git paths in the e65 or r10 scored-source trees.

This establishes **offline eligibility to consider numeric carry-forward**,
not final r10 G2 PASS. The earlier r4 source-identity disposition (SHA-256
`90f09877058b99a91c1742a9bf8d20c21dd47436244c10a35b03f42d1159d221`)
was numeric-only for 446b and cannot certify an unstaged 22b checkout. A
separate Verifier must first accept the actual trn22 r10 stage: exact enabled
stage request/final review/one-attempt assignment, fresh preflight,
hash-checked e65→r10 bundle, target checkout at the named path with clean
22b HEAD, all eleven requested source-file hashes, cross-session byte
readback, protected cache/checkpoint identities, disk/inode/STOP/marker
records, and a **completed** postflight. The disabled stage draft and its
static PASS do not prove any of these live facts. The local scored-source
checkout and trn21 must also be independently accepted clean at the same
22b HEAD before a scored run.

Only after that stage-result acceptance may a different Verifier bind the
**actual staged** `train.py`, `prepare.py`, ordered argv, four science
settings, six cache path values and current identities, and any used CPU
tool bytes to the old G2 archive and issue an r10-specific source-identity
disposition. A mismatch requires a new separately reviewed and assigned
four-rank r10 G2 with independent numeric review; no native attempt may
precede numeric PASS. Artifact-only label, output, step-log, and CPU
temporary-directory paths may differ only after their non-science role and
live parent/absence checks are established.

## Checks and remaining authority

The subject diff adds only the proposal JSON; `git diff --check` passed.
The proposal parsed under normal Python and `python -O`; template-key,
disabled-state, exact package/old-G2 argv/environment, disk-threshold,
evaluation-plan, and r2-ratification comparisons passed. The v2 ledger and
named duplicate-search files contain no scored r10 c5c A0 result. No new
CPU probe, Neuron compile, machine preflight, or evaluator was run here.

Before dispatch, the independent r10 stage-result and G2 decisions above,
this proposal's distinct Supervisor ratification, a new enabled native
request/binding chain and independent final review, fresh full local and
dual-host protocol preflight, live paired monitors and observer, and a
one-label one-attempt Supervisor assignment are all still required. A
separate independently accepted training terminal result and fresh-process
public-20M request/result are needed before this can be a valid same-host
A0 control. None of those gates is supplied by this proposal review.
