# CS22 exact-source early audit — READY_FOR_CLAUDE_AUDIT

R191 early evidence: source/CPU accepted; actual native step190 then710; cold
startup conservative upper bound633s (<900). Exact cold and FULL result pending.
**Claude's CS22 verdict is pending. This is NOT UPLOAD_READY; no upload performed.**

train.py SHA256: `fc4ab3b72da65ca42175489bcacc966b64e22f9d4e04c65ee05d737db3716006`.
Parent T3: `2b4b1ac0d7cd39628ee20282122797abd5f88a5bfc30d7d5115b45289d172894`. One tuple edit removes attention6 AND its coupled conv6,
explicitly authorized by R175 Q58 option2. No other source bytes change.

```
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0
```

Source-owned O3 defaults unchanged; caller NEURON_CC_FLAGS absent in released
environments. Read AUDIT_BRIEF.md, Q58_RESOLUTION.json and EARLY_NATIVE_EVIDENCE.json.
EXPORT_MANIFEST.json hashes every copied file except itself. Raw envelopes retain
original bytes and conditional flags. FULL score, charged total, exact cold,
checkpoint/Guardian validity, own-checkpoint diagnostics and official score stay
NULL until actual evidence. No extra unknown-telemetry gates or claims are added.
Claude audits; Poojith performs any subsequent authorized upload.
