# Final insurance-package verification

Verdict: **HOLD**. Candidate science and exact source identity pass, but the staged
package is not yet sealed accurately enough to mark human-submittable.

The packaged `train.py` is exactly 85,557 bytes with SHA-256
`116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d`.
It equals the canonical and twice-run candidate byte for byte. The exact bare launch
command passes the official local `prepare.py check`. Static review found one ordinary
`load_for_eval`, a sole `torch.load(..., map_location='cpu', weights_only=True)`, no
eval-only wrapper/hooks/monkeypatch/alternate weights/external data, and no S0 or
credential material. A fresh synthetic CPU checkpoint test reproduced exact logits
after strict loading and confirmed frozen, hook-free, state-stable evaluation.

Both cold Trn21 runs, finite checkpoints, and fresh-process public20M results are
authentic. All 13 receipts named by `EVIDENCE_MANIFEST.json` match their hashes.

The package remains on hold for metadata/evidence sealing:

1. `OFFLINE_CHECKS.md` says the launch file has no newline, but its exact 246 bytes
   contain one trailing newline.
2. `SHA256SUMS` verifies its four entries but omits `EVIDENCE_MANIFEST.json` and the
   three package/check/rejection Markdown files.
3. Runtime-job and postflight receipts are described through operator receipts but are
   not directly path/hash-bound by the manifest.
4. Final reviewer-risk documentation should explicitly disposition the import-time
   `NEURON_CC_FLAGS` and `NEURON_COMPILE_CACHE_DIR` defaults.
5. The full prior reviewer explanation remains unavailable. If it cannot be recovered,
   the human must explicitly accept that unresolved risk; neither truncated record is
   proven to concern these exact bytes.

No `train.py` change is required by this audit. After metadata-only correction, the
complete package needs a fresh independent receipt and separate human upload approval.
This receipt does not authorize or perform submission.
