# A1 trn21 full run: terminal Operator evidence

Label `fs-20260925-v2-c5c-a1-m64d128-trn21-r1-full`.
Training completed with exit 0; the separate fresh-process organizer evaluation
completed at **2026-09-25 16:09:41 UTC / 21:39:41 IST**, also exit 0.

- Public BPB: **0.9986312057378084**, context 1024, batch 1.
- Evaluation: 20,971,520 tokens, 83,462,186 bytes, 789.2307715 seconds,
  `eval_over_timeout: false`.
- Training: **3,051 steps**; `training_seconds: 1784.9`,
  `charged_seconds_est: 1785.5`, `startup_allowance_est: 900.0`,
  `over_budget: False`. The actual charged total includes startup above the
  cap; the 900-second allowance is not a claim that actual startup was 900 s.
- Same-host c5c A0: 0.9914739939934344 at 3,675 steps. Observed BPB delta:
  **+0.0071572117443740**. This is not an improvement.
- Source c5c and organizer prepare SHA pins are in `training-launch.json`.
  Exact command, sanitized environment and separate eval command are retained.

This is a full-budget table arm, covered by the human's explicit
`throughput_exempt: true`, `training_budget_seconds: 1800` rule. Its fewer
steps are reported rather than silently treating the run as step-window
CONFOUNDED. The independent result-use review and deterministic verdict are
the Supervisor/Verifier's responsibility; this Operator does not adopt a base.

The checkpoint remains at
`/home/ubuntu/trainium_a1_warm_20260925_trn21_manual_r1/out/final.pt`,
739,322,371 bytes, SHA-256
`6f941a544887365962ab491ff372ac090b4981ad8e13c8febf8fff1654f9d24b`.
It was freshly rehashed on the host after evaluation and not downloaded.
All retrieved logs/receipts match their recorded remote SHA-256 values.

Postflight at 16:10:52 UTC found no train/eval/CPU-G2 process or Neuron-device
holder; 128,351,368 KiB RAM available and 51,361,394,688 bytes disk free.
The empty conditional waiter PID 2926930 remained alive with no resolution.
A separate hash/checkpoint recheck at 16:11:25 UTC is captured in
`retrieval-eval-and-postflight.json`. No file or checkpoint was deleted.

After reporting this A1 result to the Supervisor, the distinct CPU-only
schedule gate was authorized and launched separately. It is not part of this
A1 training/evaluation record and does not change its result.
