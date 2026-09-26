# Final-slot fallback — NOT READY TO UPLOAD

Human-selected current final-package candidate,26 September2026.
Confirmed research base8c6: batch262144, matrix LR.021, ordinary warmdown.75,
momentum warmup300, seed42. No Recursive, momentum150, K2 or tables.
Actual public20M: trn21 .9869854734569898; trn22 .9867730678453942;
mean .986879270651192. These are research-source results, NOT a new official
score or measured cold result for this packaged source.

Port: start with accepted e1e2f686 file and change ONLY TOTAL_BATCH_SIZE131072
to262144 and MATRIX_LR.015 to.021. Keep the accepted launch-command bytes,
ordinary trained-model loader and all model/optimizer/time/eval code. This
avoids adding new launch flags or adapters. Existing source audit applies
historically to e1e2, not automatically to this changed candidate.

Passed locally: exact two-literal reversal to accepted e1e2; unchanged command
hash b75f844f; actual CLI/effective global identity to research8c6; three steps
of reduced8-layer128-wide BF16 CPU Muon/AdamW with two accumulation microsteps
produce identical loss, gradients, model state and optimizer state. Source
SHA256 cd0c4aa59754f46705355d964fad4528cf1de77324469b0829bcc0de149b757d.
This is author CPU evidence, not an independent audit or native qualification.

Pending: genuine cold full1800 run with fresh20M; second-seed confirmation of chosen final stack;
Claude audit via Poojith. A later confirmed winner replaces this fallback.
User deadline Monday evening package, Tuesday29 September18:00IST freeze.
This folder is local preparation only; no Downloads replacement or upload.
Only Poojith uploads. No quota assumptions or guaranteed official BPB.

Cold transport prepared locally: `prepare_cold_packet.py` mechanically derives
`cold-packet.disabled.json` from the existing trn21 private-cache full-run
transport. Driver reversal is exact and all three guardians are unchanged;
six unique training caches, fixed1800 and fresh20M are retained. No RELEASE,
live-manifest entry, queue insertion, SSH call or launch follows preparation.
If a new stack wins, regenerate against its separately frozen source instead
of running this fallback merely because a packet exists.
