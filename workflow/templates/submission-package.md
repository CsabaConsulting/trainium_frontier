# Submission package: <date>-<sha8> (prepared by agents; SUBMITTED ONLY BY A HUMAN)

| Item | Value |
|---|---|
| `train.py` SHA-256 | |
| Launch command (exactly as it will be submitted) | `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py …` |
| Research source and adopted changes | labels and verdict links |
| G2 on the clean bytes with this exact launch argv | receipt link, PASS |
| Lockstep `--compare` research vs clean (≥3 steps) | receipt link, `identical: true` |
| Neuron 100-step lockstep | steps 0–1 identical; later steps within 1e-4 relative |
| Cold replay #1 / #2 (fresh cache, bare command) | public-20M, steps, startup s, `over_budget` |
| Research mean it must match (±0.0008) | |
| Expected official | public-20M mean − 0.001 (±0.002) → range |
| Independent audit (Verifier checklist C) | receipt link, PASS |
| Remaining risks | e.g. startup margin, reviewer-sensitive constructs |

Human steps:
1. Read the audit.
2. Submit `train.py` and the launch command.
3. Record the submission id, time and official score in `research/v2/calibration.json`.
