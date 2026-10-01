# LaneB-F1 — upload-ready under Claude R164/R165

Payload: `train.py`
SHA256: `02c758a9bd980819333efeb48557e26e44c680c4e683f8eaf491782574d92c38`

Exact launch command:
`NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0`

Claude R164 authorizes this package when the original FULL checks pass and public20M is within 0.0005 of the historical LaneB mean; R165 explicitly retains that standard.
The exact-source FULL passed: public20M **0.9662405945335955**, 20,971,520 tokens / context1024; cold **646.3923283990007 s**; charged **1793.4267480380367 s**; 2,894 steps / approximately 573.3M logged tokens.
Delta vs historical parent mean 0.9661123698569338: **+0.0001282246766617**; within the acceptance band, not a new scientific improvement.
Checkpoint SHA256: `5d56fb99ea4fb1d826612952233c5d2d26b76eea918aa2e0e57e79d0d197b445` (587,788,862 bytes).
Training/evaluation exited zero; stable checkpoint and original Guardian release authenticated. Six training cache directories were empty at launch.

Raw FULL archive: `79491cd8c1a68ca26178b412d7cab99fefa018f6c031fc3d791e1099c71b95a9`.
Released packet: `c2ec474cee6969010be09395615271e4746a4d94d46555b77254b9755cfd4ca8`.
Append-only evidence: `FULL_HANDOFF_R163`, manifest `4bbcb0d84806ca9abf4bc05a6ef77b87a8919981e11fc8e195e594b2a696b0f4`.
Original 17-file export remains unchanged. Historical pending labels remain immutable snapshots; this README records the subsequent actual result and delegated acceptance. Independent appended-byte integrity is being checked separately.
This claims one actual F1 FULL; the historical parent second-receipt rationale is not a second F1 run. Unknown transient/cumulative I/O, backup and pristine-image telemetry remain unknown.
Only Poojith uploads. Codex has not uploaded or submitted this package.
