# Trn22 checkpoint cleanup — 2026-09-22

The human operator authorized cleanup so the autonomous campaign could continue. A read-only
inventory proved the box idle, `/dev/neuron0` unheld, and the root filesystem at 7.4 GiB free.

Only four controller-created `final.pt` files were removed. Before deletion, each remote SHA-256
matched the checkpoint hash already preserved in its committed result JSON:

| label | preserved SHA-256 |
|---|---|
| `al-20260922-control-trn22-lnc2x4-9a4caa7b-smoke-r2` | `8b30ac70f1436131bb968333088a470ccabc456801911af5a35e68d7e44d9db0` |
| `al-20260922-control-trn22-lnc2x4-9a4caa7b-full-r1` | `090c1106c6cb11f146897d6ca6a7d3c406623b16f955df25e4b155e11e1d85f2` |
| `al-20260922-ngram-rms-fp32-only-default-shape-smoke-r1` | `053d5c2e5974216514a4d6c514c19838cf8c0c9ef71447c791aad7031a27140c` |
| `al-20260922-nki-local-conv-only-default-shape-smoke-r1` | `e0f03915dc9419b82e82358c5230eb73d2da3949eeef98cb367824940d821899` |

No directory, log, ledger, source tree, data cache, unknown worktree, or unretrieved artifact was
removed. Free root space increased to 13 GiB. These development checkpoints are not submission
artifacts; official evaluation retrains from `train.py`.
