# Independent verifier receipt: pending c5c A0 trn22 retry r2

Reviewed: 2026-09-24 22:03 UTC. Reviewer role: Verifier, independent of the
Implementer who prepared the retry package. Reviewed package commit:
`77186376015380b6babd25b105e8189cbd0550d0`.

**Verdict: STATIC PACKAGE PASS; DISPATCH HOLD.** This verifies that the three
documents are a coherent *draft* for a new-label operational retry. It does
not approve a host action, mark the proposal complete, reset the breaker,
authorize training/evaluation, or establish a new BPB result.

## Independently checked

- Read root `AGENTS.md`, then `README.md`, `research/agentic/CURRENT_STATE.md`,
  `research/experiment-protocol.md`, and `research/agentic/DECISION_LOG.md` in
  that order, followed by the workflow role/authority documents and the three
  package files. `git show --stat 77186376` contains only the proposal,
  contract and runbook; no model, launcher, `prepare.py`, evaluator, tokenizer,
  data or decision-engine edit. `git diff --check HEAD^ HEAD` passed.
- Parsed both JSON files with `jq -e .`. Local SHA-256 matches the proposed
  unchanged research source `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
  and `prepare.py` `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
  The package files' SHA-256 values are proposal `2256f5f9a04622b4389523872592a18f3db3c3fbc5107405d4e6a00ede3650aa`,
  contract `fcf0194c6210beabfa3f2a5cfcd77e5cd00881afbd720bc1ea23cf3a75bbbf27`,
  and runbook `eeda359ff17e893366d288f7d900e3f6e7a0a1c2afb92c63ef1b94cfe4690ef0`.
- Compared the contract directly with the consumed `r1` request. The `r2`
  label is distinct; the full-chip LNC2/four-rank topology, source and
  `prepare.py` hashes, complete Track-A training-flag suffix, compiler flags,
  and all six named warm-cache paths are identical. The proposed clean
  environment explicitly pins seed `42` and max training seconds `1800`,
  which are the parsed defaults in the unchanged `train.py`; because the
  literal environment differs from `r1`, its old G2 receipt is **not** an
  exact-environment pass. The future request must bind its full `env -i`
  command, output/step paths and launcher hashes anew.
- Recomputed the disk math against the archived `r1` terminal receipt:
  floor `6,979,321,856` + max(complete dense footprint `1,854,160,896`,
  observed pre-step compile growth `1,500,942,336` + retained checkpoint
  `469,831,009`) + `1,073,741,824` reserve = `10,023,837,025` bytes.
  The proposed 10-GiB prelaunch gate is `10,737,418,240` bytes, another
  `713,581,215` bytes above that formula. This is a **provisional lower
  bound**, not proof of maximum run growth; continuous disk monitoring and
  the unchanged 6.5-GiB floor remain mandatory.
- Rechecked that the archived `r1` receipt reports zero optimizer boundaries,
  no checkpoint/eval/BPB, and a consumed one-shot label. The same-host
  historical `35c9` trn22 log at commit `499f753c` contains step-0/1 losses
  `9.0112`/`8.7822`; those are only a numerical anchor, not a valid c5c
  public-20M control. The new proposal does not impute an `r1` score.
- The training suffix includes `--no-eval-public`, while a fresh-process
  `prepare.py eval-public` at context 1024/20,971,520 tokens is explicitly
  separate and deferred. The package neither changes evaluation code nor
  claims a score or scientific win. No obvious credential patterns were
  found in the three package files.
- The draft contract has `dispatch_authorized=false`,
  `final_run_request_exists=false`, and no human breaker-reset or Operator
  preflight receipt. Only these three package files currently mention the
  `r2` label in the reviewed repository; absence of a remote checkout/output
  path has **not** been checked and must be proven by the Operator later.

## Why dispatch remains HOLD

1. The reviewed state still has both hosts below the 6.5-GiB floor and the
   trn22 breaker tripped. Only the human may arrange the exact root-volume
   and guest-filesystem expansion and explicitly acknowledge/reset the
   reviewed breaker. A fresh read-only post-resize root-device/`df -B1 /`
   receipt must prove at least 10 GiB free immediately before dispatch.
2. This Implementer-authored package is a draft, not the independent
   Researcher/Supervisor selection required by the role contract. A separate
   Supervisor must ratify the unchanged control-retry proposal; no one may
   infer authority from this Verifier receipt.
3. No `r2` source checkout, immutable source/launcher hashes, final request,
   exact-`env -i` four-rank G2 and numerical comparison, live warm-cache
   identity, idle/lock/device check, or running external boundary observer
   is yet evidenced. Before a run, check the current stop sentinel as well.
   A final request must be new, hash-bound and one-attempt-only; the `r1`
   request and label cannot be reused.
4. A later successful training run would still need five sequential external
   boundaries, finite model-only checkpoint, clean quiescence, budget and
   causality evidence, then a separately authorized fresh public-20M eval.
   None of those outcomes is established here. No AWS, SSH, training, cache
   cleanup, artifact deletion, breaker reset or submission action was taken
   in this review.
