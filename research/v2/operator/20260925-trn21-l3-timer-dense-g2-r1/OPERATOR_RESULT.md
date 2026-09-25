# Exact-source dense Recursive CPU G2

Authorized CPU-only label `fs-20260925-l3-timer-dense-trn21-g2-r1`.
Launched 2026-09-25 16:13:19 UTC after A1 fresh evaluation completed;
finished **16:13:39 UTC**, exit **0**, aggregate **PASS**, elapsed **20.0 s**.

This tests exact source
`6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`,
including the independently reviewed timer fix plus Recursive schedule ON.
It is not mislabeled as the prior f9b6 source. `request.json` and `launch.json`
record the exact dense production argv and the harness's explicit CPU-only
overrides. Native environment reported Torch **2.11.0+cpu**.

The unchanged pinned G2 harness used four CPU/gloo ranks, three optimizer
steps and synthetic 2x256 micro-batches at full model width 1024. All ranks
reported PASS, three finite losses and no cross-rank parameter discrepancy;
rank 0's checkpoint strict-load/deterministic-logit check passed. No table
RMS attribution is applicable to this dense `--no-ngram-ve` profile.
No real data, Neuron training, BPB evaluation or native correctness is claimed.

The common device lock was held for the CPU probe; its clean preflight had
131,432,075,264 available RAM bytes, 51,361,021,952 free disk bytes, no other
workload and no Neuron holder. After completion, source/tool/prepare pins
were unchanged, no matching process/device holder remained and the lock was
available. The empty host queue remained alive and unresolved.

`g2.json`, each raw rank report, the full torchrun log, launch/environment,
terminal and retrieval/postflight records are preserved. All file SHA-256
values were checked against the remote originals, including preserving their
original missing trailing newlines. The synthetic checkpoint remains on the
host with URI/hash in `retrieval-final-and-postflight.json`; not downloaded.

This is an Operator result, pending separate numerical result review. The
newly requested default-OFF versus c5c full-size identity probes are separate
unique labels and are not falsely claimed complete by this ON receipt.
