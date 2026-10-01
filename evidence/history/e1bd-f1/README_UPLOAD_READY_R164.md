# UPLOAD-READY — e1bd-F1, Claude Round 164

Current status recorded by Codex from delegated Claude Round 164 (15:51 IST),
authority SHA256 `bc62acc74790b2fcbb43b19db1f35a4e2efed89dab3d735ebcaf40833990113a`.
Claude's source verdict is **ACCEPT**; its stated FULL receipt conditions are met.
Poojith performs the upload. Codex has not uploaded anything.

Upload only this file:
`/home/poojith-devan/Downloads/Trainium-STATE-LIVE/audit-e1bd-f1/train.py`

SHA256: `25c5aec53ad7cc20f4dadf437d3ebe6dfa4f4be0214ab39d638a45df5ec7f936`

Exact unchanged launch command:

```text
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0
```

Actual FULL proof: `FULL_HANDOFF_R163/FULL_RECEIPT.json`; original verifier PASS.

- Fresh public20M/T1024: **0.9668977243944696**.
- Historical e1bd mean: 0.9671172862498154; delta **-0.0002195618553458**, within 0.0005.
- Cold startup: **678.7856580140069 s**, below 900 s.
- Charged through process exit: **1793.3613522530068 s**, within 1800 s.
- Training/eval/outer-flock exits: **0**; original Guardian released.
- Stable checkpoint: 587788862 bytes,
  `b809ce8861694669dfeb7a21a4c7702523a997eb91b86af6fe1c44270fa4141b`.
- Steps: 2850; reported tokens: approximately 564.4M (rounded log value).

Actual raw FULL archive SHA256:
`16f1f723605e1c0fd9daea2099edf79d829144497c17a3e6f2d3516bf8e3bb7b`.
Exact FULL packet SHA256:
`04d5cc3170b13a6ebba761a25f93d08fff0ee5759ce4ea2c849e85e0e2588a6e`.
Appended FULL manifest SHA256:
`128d85cd5f17833425d54b620888a5748868e70e451072073d9a59f638183953`.

The old source-export and FULL-handoff pending-audit labels are frozen historical
snapshots; this README records the later R164 acceptance and actual qualifying
receipt. Their source bytes, manifests and any Claude-authored files are unchanged.
This is a research public score, not an official result. R158 permits historical
e1bd receipts to cover the second receipt; they are not a second actual F1 run.
Existing UNKNOWN disk-write/backup telemetry remains UNKNOWN, not newly measured.
