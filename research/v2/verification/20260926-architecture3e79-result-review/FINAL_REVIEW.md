# Architecture 3e79 final result: independent authenticated research result

Verdict: **PASS_RESULT_AUTHENTICATION**. Unchanged selector verdict: **NOMINATE**, requesting the other-host confirmation. The separate human provisional threshold `delta <= -0.005` is met. This report does not adopt, submit, authorize a host launch, or edit the ledger.

| Item | Candidate | Matched control |
| --- | --- | --- |
| Label | fs-20260926-arch3e79-primed-trn22-r2-full | fs-20260925-v2-c5c-a0-control-trn22-r10 |
| Host / seed | trn22 / 42 | trn22 / 42 |
| Source | 3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c | c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586 |
| Public20M/context1024 BPB | 0.9863403052402583 | 0.991513958330512 |
| Steps | 3,629 | 3,633 |

Exact unrounded delta: **-0.005173653090253749**. This is the architecture on the old c5c schedule, not an adopted Recursive/6ef7 comparison. The prior 93ae result (1.017340669168314, 2,246 steps) remains historical evidence unchanged; it is not relabelled or erased. The query uses exact-source arm ID `L2-unet8x1024-3e79-primed`, retaining the existing 93ae row separately.

## Authentication and timing

`audit_final.py` independently reran **65 training/binding/accounting checks and 58 final checks**, under normal Python and `-O`. All passed. It reads raw training commit `8d9ee4b03f0042fc633a96b90637887e0d864490`, final commit `e5d28dc9a5c5f9539721f1de1a8cf34bd203b535`, and verifies final canonical import `7132a6482df7ce1c9b168265adc5711b724a2237` is byte-identical. All 8 training and 10 final retrieved artifacts match archived byte counts and SHA256 hashes.

Fresh evaluation has 20,971,520 tokens, 83,462,186 bytes and 57,061,344 total nats. Independently computing `nats / (bytes * ln(2))` reproduces the reported BPB exactly. Raw eval SHA256: `76d13debfbde90e523433d6dc8b1a380a6c759ddf3797f58628f2b65077ac70a`. The isolated evaluator uses the unchanged prepare.py, exact source, same trained checkpoint path, context1024, batch1 and 1,200-second timeout. It completed in 707.3155541670276 evaluator seconds; whole eval-process observer time is 720.5464018430794 seconds. Eval/outer exits are zero; terminal is 2026-09-25T20:58:07.553386Z.

The training observer recorded every consecutive boundary 0–3628. Uncapped fifth boundary is **814.6809981259285 s**; process wall through exit is **2603.889482168015 s**. Actual charge is `2603.889482168015 - min(814.6809981259285,900)` = **1789.2084840420866 s**, leaving **10.791515957913361 s** under 1800. The final 4.482617728994228 seconds after the last observed step are included. Source estimates (1784.7 training / 1785.1 charged / 812.5 startup allowance) are recorded separately and are not substituted for external timing.

This **full run reused warm exact-source cold-test cache roots**. Its large first interval is not proof of cold caches. The previously accepted separate genuine cold-five startup evidence remains distinct; this result audit neither claims a cold full replay nor imposes another one. The query-only row marks this full run cold-unproved, so the selector's diagnostic cold warning is not a new finding that the earlier source-specific startup test never occurred. Any final ledger enrichment with that earlier test must retain its separate evidence/scope.

The common lifetime lock stays bound to original device/inode `66305/832589`, with authenticated `ps` and `neuron-ls` idle checks before train and fresh eval. The outer wrapper exits after eval; the 21:00:48 UTC postflight shows no original descendants, compiler jobs, retired queues or Neuron holders, and the unchanged lock is free. No concurrent/new host job was launched by this auditor.

Checkpoint is retained on host, **453,048,594 bytes**, SHA256 `068749c2559f7c68f6acc76d558caff9d3806d85273fa8562483d87bbad50d25`. The final archive authenticates the exact unchanged pre-evaluation checkpoint receipt; it does **not** record a new post-evaluation hash of checkpoint bytes, and no checkpoint was downloaded. The successful fresh evaluator references that exact trained path. The raw evaluator does not emit a separate native causality result; `causality_passed:null` is not rewritten to true.

## Selector and handoff

Selector SHA256 remains `fa4a3d3dd3697093f93ac73b5c95335a7c1efa12534589ed3c3e2b64940ca0ac`; current ledger SHA256 at the query was `014d38b8112b315b973ae49da2bd3d5dc100f6c6a3c03e0f5ff58e680040f8a1`. `decide_arm(current_rows + candidate, exact_source_arm)` returned NOMINATE. The candidate exceeds the single-host improvement threshold, does not reach the big-win hold thresholds, and has no selector invalid reason. Cross-host evidence is still pending; the human provisional threshold is a separate policy decision, not a modification of decide.py.

`candidate-ledger-row.query-only.json` is supplied for Supervisor review and possible append; **this auditor did not append it**. `FINAL_RESULT.json` preserves the full candidate, unchanged selector output, all hash checks, training accounting and limitations. One early audit-script `KeyError` from using a nonexistent binding key was repaired and retained in `audit-attempt1.txt`; no experiment source, artifact or threshold was changed.

Reproduce locally (stdlib only): `python3 -B research/v2/verification/20260926-architecture3e79-result-review/audit_final.py`, or add `-O`. Reproduction intentionally refuses if the candidate has since been appended to the ledger, preventing an accidental duplicate query being presented as the original pre-append receipt.
