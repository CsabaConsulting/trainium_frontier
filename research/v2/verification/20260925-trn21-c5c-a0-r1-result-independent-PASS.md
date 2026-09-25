# Independent trn21 c5c A0 result-use review — PASS

Reviewed 2026-09-25 UTC by a Verifier who did not implement or operate this run. Scope is the archived trn21 control result under `research/v2/observations/20260925-trn21-a0-manual-r1/`, not approval of any later table arm or submission. This review used retrieved files and the separate archived read-only postflight; it did not contact or change the host.

## Identity and execution

- Unique label `fs-20260925-v2-c5c-a0-control-trn21-r1`; assignment `research/v2/assignments/20260925-trn21-a0-breakglass-r1.json` hashes to `1a16abec46b1cd7fef4692b5a493fa629862c43a997b7125e8841253f4b7af46`.
- The launch receipt's source `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586` and trusted `prepare.py` `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb` match independent local SHA-256 checks. Preflight, stage, poststage and launch transport receipts show exact source match, an idle host, no local STOP/breaker, and at least 57.8 GB root free before launch.
- `host-launch-final.json` exactly binds LNC2, four `torchrun` ranks, seed 42, 1,800-second training budget, `--optlevel=1`, and the assignment's ordered dense control argv ending in `--matrix-lr 0.015`. Torchrun exited 0. The training log reports 3,675 steps, 481.7 million tokens, finite sampled losses, 1,784.8 training seconds and 1,785.2 estimated charged seconds, with `over_budget: False`. No compiler exception, NaN, or traceback appears in the retrieved log.
- The retained checkpoint was 469,834,849 bytes at `/home/ubuntu/trainium_a0_warm_20260925_trn21_manual_r1/out/final.pt`, SHA-256 `fecbed822553c256c12d20eab4dfb34777398ab30a8e3d50a6c393c76612bdc4`. The separate evaluator preflight and launch independently recorded that same checkpoint hash and size. The binary remains on the host and was not transferred or deleted.

## Fresh evaluation and postflight

- A separate process invoked the unchanged trusted `prepare.py eval-public` with that exact `train.py` and checkpoint, Neuron device, batch size 1, context 1024, exactly 20,971,520 tokens, and a 1,200-second evaluator limit. It exited 0 with `eval_over_timeout=false`: 83,462,186 bytes, 57,358,336 nats, and public BPB `0.9914739939934344`. Independent arithmetic `57,358,336 / (83,462,186 * ln(2))` reproduces the reported BPB.
- I recalculated the SHA-256 of the nine artifacts listed in `OPERATOR_RESULT_PENDING_REVIEW.md`; all nine equal the listed values, including `host-training.log` `014511287763ad3c4fc1d385253fc3ff3da9c2c6dac5921eef47a04dbc8e211b` and fresh eval log `539f3917a63025ece66e7092aae919c361d8ad3914401f9d9c5c2d827b1944d3`.
- Distinct read-only postflight `research/v2/observations/20260925-dual-host-readonly-operator/OPERATOR_RAW_TRN21_READONLY_20260925T142739Z.json` ended 14:27:44 UTC. It found no matched train/eval process, Neuron PID, `/dev/neuron0` or host-lock holder, or STOP; root free bytes were 56,024,961,024 and free inodes 31,562,701. Its archived hash matches the Operator's table.

**Disposition: PASS for same-host c5c A0 control result use at public-20M/context-1024.** It is neither a new adopted improvement nor an official BPB or S0 submission-safety result. Each subsequent job still needs its own proposal, gates, one-use assignment, fresh host preflight, and independent result review.
