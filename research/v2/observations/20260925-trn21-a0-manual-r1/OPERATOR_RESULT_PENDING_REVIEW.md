# trn21 c5c A0 control: full run and fresh public-20M result

Operator evidence only, observed 2026-09-25 UTC. This is not an Analyst verdict,
an adopted BPB improvement, or submission clearance. Independent post-result
review is required before the control is used for a decision.

- Unique label: `fs-20260925-v2-c5c-a0-control-trn21-r1`.
- Host root: `/home/ubuntu/trainium_a0_warm_20260925_trn21_manual_r1`.
- Exact `train.py` SHA-256:
  `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`.
  Unchanged trusted `prepare.py` SHA-256:
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- Detached full-chip LNC2 four-rank training launched
  `2026-09-25T13:31:07.744399Z` with the command and environment in
  `host-launch-final.json`. Training ended `2026-09-25T14:14:03.251248Z`;
  torchrun exit code was `0`. It completed 3,675 steps and 481.7M tokens;
  reported training time was 1784.8 s, charged estimate 1785.2 s, 1800 s
  budget, `over_budget: False`, and finite logged losses.
- Checkpoint retained on host at
  `/home/ubuntu/trainium_a0_warm_20260925_trn21_manual_r1/out/final.pt`:
  469,834,849 bytes, SHA-256
  `fecbed822553c256c12d20eab4dfb34777398ab30a8e3d50a6c393c76612bdc4`.
  The binary checkpoint was not transferred or deleted.
- A distinct fresh-process trusted `prepare.py eval-public` started
  `2026-09-25T14:14:57.252307Z`, exactly 20,971,520 public tokens,
  context 1024, batch 1, Neuron device and 1200 s evaluator timeout.
  It ended `2026-09-25T14:26:47.151756Z` with exit code `0`.
  Evaluation reported `eval_seconds=697.5682576750405`,
  `eval_over_timeout=false`, 83,462,186 bytes, 57,358,336.0 nats and
  **public val_bpb = 0.9914739939934344**. Recalculation from nats / bytes /
  ln(2) matches exactly at displayed precision. This is the same-host research
  control, not an official leaderboard score.
- Separate read-only trn21 postflight ended `2026-09-25T14:27:44.243743Z`:
  no matched train/eval process, no non-diagnostic Neuron PID, no `/dev/neuron0`
  or device-lock holder; 56,024,961,024 bytes and 31,562,701 inodes free;
  no STOP observed. No host cleanup or second job occurred before postflight.

Retrieved artifact hashes (SHA-256; all local copies are of the exact host files):

| Artifact | SHA-256 |
|---|---|
| `host-launch-final.json` | `f97a553b32d53da3526e850040b5c31ff004f1bbb98d8f3698de5de0337b19b4` |
| `host-training.log` | `014511287763ad3c4fc1d385253fc3ff3da9c2c6dac5921eef47a04dbc8e211b` |
| `host-run-exit-code.txt` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `host-eval-launch.json` | `ad1b09d84bc0c0764696e4e9aa5249ed4cca8fb8370a0c1a033f8b9d634eda2b` |
| `host-fresh-public20m.log` | `539f3917a63025ece66e7092aae919c361d8ad3914401f9d9c5c2d827b1944d3` |
| `host-eval-exit-code.txt` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `eval-preflight-2026-09-25T141443.269840+0000.json` | `ab4c0ff18a659445e9fbaa9a41e8179b6975c4bb129aacee8351191fc10e0f01` |
| `eval-launch-2026-09-25T141452.042863+0000.json` | `979d888f5a790696284267e03e704e31e59face112733f45b926171646b7e397` |
| postflight `OPERATOR_RAW_TRN21_READONLY_20260925T142739Z.json` | `0791c1c9fae0a60a2b5d399d784809e168aff27403fa3b08d6b86a7f72e4ed3b` |

The last postflight file is in the sibling
`research/v2/observations/20260925-dual-host-readonly-operator/` directory.
The earlier launch report and receipts in this directory describe preflight,
staging, worker-presence proof, and the failed earlier S0 cold replay.
