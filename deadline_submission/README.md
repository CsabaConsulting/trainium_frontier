# Upload-ready files

Use the train.py from one variant directory as the uploaded standalone train.py. It imports only the existing competition/runtime dependencies and prepare.py. Do not upload helper scripts, logs or checkpoints.

The currently validated new option is rmsfp32: matched public20M BPB 0.9639958848, a 0.0003419094 improvement over the historical 0.9643377943 mean. It is one run and the gain is near measured training noise. It does not establish the roughly 0.009 improvement needed for the recorded leaderboard cutoff. Frozen is the cleaned equivalent of the existing official 0.9645 submission. Headbias and dualhead remain unpromoted pending their exact public20M results.

All four files remove unreachable fixed-switch branches, the unused non-flat optimizer class and unused optimizer-resume methods. They preserve readable source and comments. Canonical executable AST equivalence passed, along with CPU exact initialization/global RNG, deliberately nonzero MLP logits, loss, all parameter gradients, and optimizer settings/coverage. Trusted static checks passed for all four. This cleanup has no fresh 30-minute training rerun; it is checked for equivalent behavior to the scored source.

Common command (76 characters, --tbl_freeze_tokens defaults to 0):

```bash
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py
```

The compiler flags, four ranks, LNC2 training, seed and original wall-clock accounting remain in train.py. The shared official prepare.py remains unmodified. Public diagnostic scores are not portal scores.
