# Independent static/safety review: trn22 A0 r2 CPU G2 retry r3

Reviewed 2026-09-24 23:02 UTC. Verifier is independent of the Implementer who
prepared commit `4b4cad169052f9f336f7c07571c61a91f21eb30a`. Scope is the
new-label **CPU-only** G2 package at
`research/v2/g2-probes/fs-20260925-g2-a0-c5c-exactenv-trn22-r3/`.
No host, Neuron, evaluator, AWS, breaker, or submission action was taken here.

## Verdict

**STATIC PASS for a separately assigned CPU G2 r3 only; native A0 r2 and
fresh evaluation remain HOLD.** The package corrects the one known
infrastructure failure without altering model inputs or the Track-A recipe.
This receipt is not Operator preflight, a G2 result, a scored-run request, or
one-shot native authorization. The Supervisor must explicitly assign the CPU
probe, and the Operator must pass every live gate before using the reviewed
command once. A failed probe consumes this new label.

## Independent checks

- Read root `AGENTS.md`, then `README.md`, `research/agentic/CURRENT_STATE.md`,
  `research/experiment-protocol.md`, `research/agentic/DECISION_LOG.md`, then
  `workflow/AGENTS.md`, `workflow/CODEX_PROMPT.md`, the active controller brief,
  and the Verifier role. Reviewed the archived r1 disk incident, its human
  acknowledgement and independent re-arm reviews, the r2 source-stage/G2
  failure, and the complete r3 contract/runbook. `git show --stat 4b4cad16`
  adds only those two r3 documents; it changes no source, `prepare.py`, data,
  evaluator, tokenizer, decision engine, launcher, or breaker. `git diff
  --check 4b4cad16^ 4b4cad16` and shell syntax of the embedded command pass.
- Recomputed local source/tool SHA-256s: research-v2 `train.py`
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`,
  `prepare.py` `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`,
  G2 tool `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`,
  and containment tool `0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d`.
  The host must rehash its own copies before use. The r3 JSON parses and its
  source/prepare hashes, exact Track-A argv, and four science-effective
  environment values exactly match the reviewed A0 r2 contract. The r2
  contained launch receipt has the same four values and argv tail.
- The archived r2 Operator HOLD receipt SHA-256 is
  `a476312c1b9b24e785b72aac1ef78d165fc65b36fa193ea65530a1d188dc9aba`;
  its failed G2 receipt is
  `0d58c8d4f7d5ff08eec451a743778151e0c8716436854e740ea42394f69b41e0`.
  Rank 0 completed its first synthetic optimizer step but failed when
  `prepare.record_step_boundary(0)` opened a log whose parent was absent;
  the three Gloo peer-disconnects were secondary. Quiescence passed; no
  Neuron training/eval, science checkpoint, BPB, or scored-run label was
  produced. The **CPU G2 r2 label is consumed**; science A0 r2 is not.
- `prepare.py` lines 43–45 and 96–111 use the step-log path only to append
  integer boundaries from rank 0 after an optimizer step. It is not read as
  data or model input. The new r3 root is unique, outside the clean checkout,
  and must be absent before an atomic mode-0700 `mkdir`; its parent then
  exists while the step-log file remains absent. `preflight_dryrun.py` appends
  CPU/no-compile/three-step and temporary `--out-dir` arguments, overriding
  the science OUT_DIR default. `TMPDIR`, process label and step-log destination
  are the only relevant literal changes. No science output path is created.
- The archived prior c5c CPU control G2 receipt hashes to
  `17ce1ec84ddc0d585ecd103237ecfcd86d3820cdd9aaf1c4fe8b60ed308b90e3`
  and is PASS for the same `train.py`, `prepare.py`, Track-A argv, three steps,
  four ranks and 2×256 micro-batch. The future r3 receipt must be compared
  independently to that anchor, including per-rank losses, plan and final
  parameter digests; the anchor is **not** a native or BPB result.

## Breaker classification

The r1 disk stop was one failed **native training launch** and its named disk
breaker was subsequently re-armed under narrow human authority. The r2
failure occurred in workflow `G2 preflight` (step 3 of
`workflow/AGENTS.md` §3), **before** the separately gated scored `Launch`
(step 5). `workflow/tools/run_job.py` likewise treats an absent PASS G2 as
preflight refusal (exit 4), distinct from training failure (exit 5). The fact
that the CPU probe used `contained.py launch` for safe process isolation does
not convert it into a second scored/native launch. Therefore the two-
consecutive-host-launch-failures rule is **not triggered** by these two
attempts. This conclusion does not clear or reset any breaker; the Operator
must read current controller and host STOP state before acting.

The r2 error was a known package artifact-path mistake, not evidence of NaN,
over-budget training, causality failure, dirty/hash-drifted source, surviving
process, device holder, or inconsistent trained/eval model. It is a failed
G2 **infrastructure gate** and must not be relabeled PASS or counted as a
scientific result. On the archived evidence it does **not** trigger the
global **correctness** breaker. Any new contrary evidence,
new active breaker, or repeated failure is a HOLD requiring a fresh safety
review and the applicable human decision; the earlier disk acknowledgement
cannot reset another breaker.

## Conditions left for the Operator and later Verifier

Before this CPU probe: confirm the exact host and clean source checkout,
all source/tool/checkpoint/cache hashes and identities, no train/eval/G2
process or Neuron/lock holder, no STOP, root free ≥10,737,418,240 bytes,
sufficient `/dev/shm`, absent r3 root/label and absent science r2 paths.
Create only the isolated r3 root, and run only the hash-bound command in the
runbook under containment. Afterward prove quiescence even on failure,
retrieve/hash all rank and process logs plus step log, and recheck checkout
and protected artifacts. Require `verdict=PASS`, four PASS ranks, three
finite losses each, successful checkpoint/load check and exactly `0,1,2`
step-log lines. A separate Verifier must compare its complete result to the
prior c5c PASS receipt. **Native A0 r2 remains HOLD** until a new final
request, live native checks, external observer and one-attempt Supervisor
authorization pass independently.
