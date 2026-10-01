# T3 exact-source audit package — source review requested now

R186 current base/upload candidate; **NOT upload-ready in this export**. Both exact FULL receipts are valid. Own-checkpoint LNC1/unset2M diagnostics and separate final byte-integrity/Claude package verdict are pending. Source audit may begin immediately; acceptance is not a prerequisite to requesting it.

Proposed train.py SHA256: `2b4b1ac0d7cd39628ee20282122797abd5f88a5bfc30d7d5115b45289d172894`. Exact launch command:

```
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0
```

The source supplies main O3+BF16-auto-cast and import/eval O3 defaults. Leave caller NEURON_CC_FLAGS unset to reproduce these tested defaults. Both actual FULL environments omit it; full environment records are in cold-evidence.json. No scientific source, prepare/evaluator, tokenizer or command change is introduced by packaging.

Read AUDIT_BRIEF.md, PAIR_RECEIPT.json, each host's FULL_RECEIPT.json/cold-evidence.json, and OWN_CHECKPOINT_DIAGNOSTICS_PENDING.json. Raw JSONs without original trailing newlines are represented by exact-base64 envelopes preserving every byte and SHA. The source and parent are plain exact files; only train.py is the proposed payload, all other files are audit evidence.

Mean research public20M BPB: 0.96433779426305065. Not an official platform score. Both runs use seed42, not independent seeds. Current checkpoint presence and unrecorded I/O totals remain UNKNOWN. Existing optional paths/cache behavior are not hidden by a cleanup rewrite. No credentials, checkpoint tensors, external data, SDK installation, host calls or upload operation are part of this export.
