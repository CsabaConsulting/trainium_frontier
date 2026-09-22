# Trn21 checkpoint cleanup — 2026-09-22

The autonomous controller stopped before launching the promoted full run because the root
filesystem had 7.8 GiB free, below its conservative 8.0 GiB launch gate. The human operator had
already authorized cleanup of the Trn2 workers so the campaign could continue.

Only four controller-created `final.pt` files were removed. Before deletion, every remote
SHA-256 matched the checkpoint hash already preserved in its committed result JSON:

| label | preserved SHA-256 |
|---|---|
| `al-20260922-control-trn21-lnc2x4-9a4caa7b-smoke-r1` | `ccaf772c8a877a2bd6359325e7b7ee6169643fc00819b470453747672d0c8d4e` |
| `al-20260922-control-trn21-lnc2x4-9a4caa7b-full-r1` | `608aa8304145d6c89751966f78387ca62dbf39a162d273b5775914de6b61f3bb` |
| `al-20260922-batch-ramp-only-default-shape-smoke-r1` | `c8c7e1fa90f453da4a31eb1b85ec5bbe174494be8df72deafa295403a39eb2fd` |
| `al-20260922-batch-ramp-only-default-shape-full-r1` | `c92363c886fcd478266423c73c953954d11a6b76caa85045a900eccd9e18dcf1` |

No directory, log, ledger, source tree, data cache, unknown worktree, or unretrieved artifact was
removed. Free root space increased to 13 GiB. These development checkpoints are not submission
artifacts; official evaluation retrains from `train.py`.
