# S0 and `116188` submission-readiness audit — 2026-09-24

## Verdict

- **S0 remains `PATCH / HOLD`, not a canonical candidate and not human-submittable.** The only
  exact S0 source identity recorded in the canonical repository is the independently recomputed
  private-object SHA-256
  `ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb` for
  `workflow/candidates/S0/train.py` at private commit
  `c30422e2a9f0ed7203c85597eef1cb78c9480b62`. The source bytes, launch text, private lockstep
  record, and private commit are not present in the canonical repository. Their recorded hashes
  prove audit provenance only, not candidate correctness or readiness.
- **The current evidence-backed insurance candidate remains exact SHA-256 `116188cf...`.** The
  canonical package copy is
  `submissions/dense6x1024_direct_review_matrixlr015/train.py`, 85,557 bytes, SHA-256
  `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`. Its launch-command file
  is SHA-256 `b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd` and its provenance file
  is SHA-256 `5c27f134df94f5873c317d03b01e13458bfb97f9d2b59d7cfa8401f6b9486168`.
- **Do not upload the current control-plane root `train.py`.** At audited head
  `e07b97c86927b63726ef0efef17253b02f8b99b1`, root `train.py` is 429,472 bytes with SHA-256
  `52ac9e6560a40592255e3499f4d42ac6e7073ea82da1b428e03dc984e6d4baa8`, not `116188cf...`.
  Root `train.py` was promoted to exact `116188cf...` only on pinned canonical-submission commit
  `e12d0deedea806128c1999f4b0a6340b967a7e64`.
- **`116188cf...` has enough existing hardware evidence; S0 does not.** No additional training or
  leaderboard submission is needed to finalize an offline `116188cf...` upload package. S0 needs
  a canonical implementation plus the full independent equivalence and replay sequence below.
- **Neither path is presently submission-approved.** The independent `116188cf...` verifier
  explicitly says `submission_approved: false`, and the repository still lacks the complete
  explanation for a later reviewer rejection. A leaderboard submission must not be used as a
  reviewer probe.

This is an Analyst status verdict only. It does not approve a package, launch, or submission.

## Audited identities and authority

The canonical private-workflow audit is commit
`0ba1beee8920ea55397a14b01428df8a3ccc1d1f`; independent documentation receipt
`e07b97c86927b63726ef0efef17253b02f8b99b1` recomputed all 19 newly recorded private-source
object hashes and verified that no private candidate, patch, queue, evidence, or runtime code was
imported. The audited private tip `752ea5e00d020e1f2e601ee633448bf5ea52f522` and S0 commit
`c30422e2a9f0ed7203c85597eef1cb78c9480b62` are currently absent from the canonical Git object
store. Consequently S0's file size, bytes, launch-command contents, and claimed CPU output cannot
be independently re-read from the canonical checkout. The durable S0 identities are:

| Private object | Recorded SHA-256 | Canonical disposition |
|---|---|---|
| `workflow/candidates/S0/train.py` | `ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb` | provenance only; bytes not imported |
| `workflow/candidates/S0/launch-command.txt` | `61a3348a6dc5795ebd8f1a0da90262b19a9a33ba4cb941c09f7c7aa5828fc7db` | provenance only |
| `workflow/evidence/lockstep3-116188-vs-S0.json` | `abc898449fba338032c9a7c77d3ea1933b1309ad69bb29f0701805df91d78fa4` | private CPU claim only; not canonical evidence |
| `workflow/candidates/S0/EVIDENCE.md` | `6a8e962916059e3069d5a7d398674d09bda368f8eed2dd03a800cd394e2fef4d` | private claim only |

The exact `116188cf...` package launch text is:

```text
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public
```

## Gate matrix

| Gate | Exact `116188cf...` | S0 `ad223cdc...` | Can existing evidence close S0? |
|---|---|---|---|
| Exact canonical candidate bytes | **PASS.** Package bytes are present; promotion `e12d0dee` makes root bytes exact. | **MISSING.** Only private-object hash provenance exists; no canonical candidate bytes/package. | No. A canonical Implementer port must create and pin its own exact identity. |
| Causal provenance/build | **PASS.** Provenance pins review-clean base `c5956187...`; only `MATRIX_LR = 0.02 -> 0.015`. | **MISSING canonically.** Private `make_s0.py` and claims were deliberately not imported. | No. Reconstruct and independently review a minimal canonical diff from exact `116188cf...`. |
| CPU semantics, checkpoint/load-for-eval, causality, optimizer and CE gates | **PASS by exact-byte identity and preserved evidence**, per independent verifier `8a587d5983830444394e921240dabaf909ac0482`. That receipt notes tensor tests were not rerun in its controller because that Python lacked Torch. | **NOT PROVEN.** The private lockstep object hash establishes provenance only. | No whole-candidate transfer: S0 bytes differ. Preserved tests may be reused as the test plan, not as a PASS result. |
| Static reviewer-risk scan | **PASS for the pinned bytes** at `8a587d59`: one `load_for_eval`; sole `torch.load` uses `map_location='cpu'`, `weights_only=True`; no `_submission_`, `MethodType`, `setattr`, forward replacement, or hook-install marker. | **MISSING.** The audit calls S0 a plausible cleanup hypothesis, not a reviewed candidate. | No. Reviewer verification is explicitly required for S0. |
| Neuron execution/full-chip training | **PASS.** Exact bytes completed two valid seed-42, full-budget Trn21 runs. | **MISSING.** No canonical Neuron run. | No. `116188cf...` runs are a reference, not proof for changed S0 bytes. |
| Neuron semantic lockstep/equivalence to current candidate | Not applicable as a self-equivalence promotion; exact run bytes are pinned. | **MISSING.** The authoritative audit specifically requires independent Neuron lockstep. | No. The private CPU claim does not substitute for Neuron equivalence. |
| Cold compile/full-run replay | **PASS twice.** Both valid Trn21 runs used fresh empty SDK caches. Run 1: 3,635 steps, charged 1,790.328 s. Run 2: 3,654 steps, charged 1,790.191 s. | **MISSING; two cold comparable runs required.** | No. Existing cold runs prove only `116188cf...`; they can be the frozen comparison reference. |
| Fresh-process exact public20M | **PASS twice.** 20,971,520 tokens / 83,462,186 bytes: `0.9920722144854677` and `0.9918353324459732`; finite, weights-only checkpoints `503dbfda...` and `5c42f143...`; evaluator rehearsals used separate fresh empty SDK caches. | **MISSING; two comparable public20M results required.** | No. Do not assign the `116188cf...` scores to S0. A delayed/runtime-changed S0 cohort may also need a contemporaneous control under the protocol. |
| Additional Trn22 evidence | A third exact-byte workload produced `0.992968300659387`, 3,631 steps, but terminal protocol status is `FAIL_POSTFLIGHT_OPERATIONAL_DISK_FLOOR` at commit `68ecfeb168ddd095ed14f1908fee7c0284dbc97f`; it is descriptive submission-safety evidence only and cannot authorize advancement. | None. | No. |
| Independent final package/reviewer approval | **MISSING.** `8a587d59` says `PASS_AS_PINNED_RESEARCH_BASE` and `submission_approved: false`. | **MISSING.** Explicit HOLD condition. | No. |
| Official private score/portal acceptance | **UNKNOWN / not tested by this audit.** Public20M is development evidence, not an official private result. | **UNKNOWN / no submission.** | No. Do not spend a rate-limited submission to fill an offline evidence gap. |

The two valid `116188cf...` runs are evidence commits
`7fe770b2185f594cfd66fead05aee2051ce1793b` and
`28b89bfc6d65be9188f8d8a78accd6187e7627b7`. The independent canonical promotion verifier is
commit `8a587d5983830444394e921240dabaf909ac0482`. These establish the stated `116188cf...` gates;
they do not approve S0 by analogy.

## Minimal ordered path without wasting a leaderboard slot

### Fast path: finalize exact `116188cf...`

1. **Resolve package identity offline.** Choose either the exact 85,557-byte package file above or
   root `train.py` from pinned commit `e12d0dee`; require SHA-256 `116188cf...` and the exact launch
   command/hash above. Never substitute current control-plane root `train.py`.
2. **Recover the complete reviewer rejection text from the human/portal record without submitting
   again.** `CURRENT_STATE.md` says a later submission was rejected and its explanation is not yet
   recorded. Preserve submission ID, timestamp, uploaded-file hash/name, status, and verbatim
   reviewer explanation, then determine whether that rejection concerned these exact bytes.
3. **Run one final independent, offline upload-package verification on the exact staged file.** It
   should execute the official local submission/check contract, static/AST reviewer audit, and the
   preserved CPU/load-for-eval/causality/checkpoint tests in an environment with Torch; verify the
   launch command, single-file/custom-kernel closure, absence of evaluator/tokenizer modifications,
   and equality to the twice-run bytes. This is verification, not training.
4. **Bind a final verifier receipt and human checklist to the exact staged file and launch hashes.**
   Cite the two cold replays and fresh-process public20M records; do not rerun Trainium unless the
   staged bytes or required environment changed.
5. **Only after the offline receipt passes, ask the human to upload once.** Portal acceptance and
   official private score remain unknown until that human submission; the portal is not a test
   harness.

This is the fastest evidence-valid path. The remaining work is packaging/review, not hardware.

### If the human specifically requires S0 cleanup

1. First obtain the missing reviewer rejection reason; it may show whether S0 addresses a real
   blocker or merely adds schedule risk.
2. Create a human-approved S0 proposal. An Implementer must make a minimal canonical cleanup port
   from exact `116188cf...`; do not import the private candidate, patch, evidence, or conclusions.
   Pin the resulting canonical source and launch hashes.
3. An independent Verifier must run deterministic diff/build checks and the full CPU suite against
   exact `116188cf...`: initialization/RNG, logits, causality and batch/permutation independence,
   strict checkpoint load, state/buffer immutability, MLP forward/backward, CE lane, two-/four-rank
   owner semantics, state round-trip, and launch mismatch rejection, plus reviewer-risk analysis.
4. Pass a predeclared exact Neuron lockstep discriminator on the canonical port. Do not relax a
   valid numerical gate or reuse the private CPU claim as Neuron evidence.
5. Only after lockstep PASS, perform the two independently authorized, cold-cache, seed-42,
   full-budget comparable S0 runs required by the accepted audit; preserve finite checkpoints and
   run fresh-process exact public20M evaluation for each. The two existing `116188cf...` runs are
   the frozen reference, so do not repeat that control merely for symmetry unless the proposal
   determines that runtime/cohort drift makes the old comparison invalid.
6. Independently verify the exact final S0 upload package and reviewer surface, bind the run and
   evaluation receipts to its hash, then return to the human for one submission decision. On any
   failed S0 gate, retain/finalize `116188cf...` instead of probing the leaderboard.

## Deadline and submission-budget facts available in the repository

- `README.md`, the repository competition contract, says Phase 1 closes on **September 30, 2026**
  and permits **5 Phase-1 submissions per team per week**.
- `CURRENT_STATE.md` and `AUTOLOOP.md` say `2026-09-30 23:59 PT`, but that exact time/time zone is
  internal repository state, not independently backed by an official portal artifact in this
  checkout. Under the requested repo-only standard, the authoritative exact close time is
  **unknown** and should be verified by the human from official competition material.
- `STANDING_MISSION_V2_2026-09-24.md` sets a human-approved **research freeze of
  2026-09-29 18:00 IST**. That is an internal operational deadline, not the competition deadline,
  and it expressly does not authorize leaderboard submission.
- The repository does not contain a current signed-in portal receipt establishing remaining weekly
  submission allowance, current portal status, or an S0 upload. Those facts are **unknown** here.

## Final status

**S0: HOLD. `116188cf...`: evidence-backed insurance candidate, ready for offline final-package
verification but not yet submission-approved.** No S0 evidence currently removes a `116188cf...`
gate, and no existing `116188cf...` result removes an S0 semantic or replay gate.
