# Independent retrieved G2 result review

ACCEPT_RETRIEVED_STANDARD_CPU_G2_ONLY. Executed my own receipt/arithmetic comparison script; 134 checks passed, exit 0. Command:

```
/home/poojith-devan/venv/bin/python /home/poojith-devan/checkpoint-fresh-review-vgtBCW/review_actual_g2.py
```

Raw output: `actual-g2-raw.json`. Inputs were only the isolated supplied bundle, final process census and retained identity map. The script imports no model/Torch/helper, reads no host files and performs no host actions. This accepts evidence for the completed standard CPU G2 pair only; it does not release a new job.

I authenticated outer receipt hashes and recomputed the UTF-8 byte counts/SHA256 of the used embedded raw requests, launch records, G2 receipts, all eight individual rank receipts, logs, terminal/checkpoint identity/comparison records. The source and request pins match the previously reviewed exact sources/spec. Both arms completed 3 steps, 4 ranks and synthetic 2x256 microbatches. Each of eight rank records reports PASS, 65/65 changed parameter tensors, empty mismatch/unchanged lists, finite three-loss sequences and the identical plan hash. Summing each owner partition independently gives 113256724 elements. Both logs also report exactly 113,256,724 parameters.

My direct comparison of all 65 rank0 parameter digest entries produces no differing keys or values; these are the standard G2's 16-hex digest prefixes, not full raw checkpoint tensor verification. My subtraction of the 12 paired rank/step loss values yields twelve exact zeros. These computations agree with the separately returned successful standard comparison. The exact loss vectors and plan evidence are in the raw receipt.

All 17 retained identity records match the bundle's identity map and report stable positive-length files. The two private checkpoint identities agree across the separate identity record, observed host hash receipt and retained identity map: reference 453048594 bytes, SHA623577bb7e6f60d183a1944b28a5624cd55767c16fcf03a9248fd4573f6a6762; candidate 453044626 bytes, SHA3c7efed1e053cf7607e2099d52643cb368e8a1f14aebe2a36ba3f1444b63c9c9. I did not retrieve or rehash checkpoint file contents locally; this accepts authenticated host-retrieved identity receipts only. A later ordinary helper must reauthenticate exact files pre/post.

The outer command interval was 42.800372 seconds. The 253-row final census has unique PIDs. Independently filtering PID/PPID/PGID/SID against the five recorded dispatcher/parent IDs found zero related rows. The two arm process metadata IDs and dispatcher ID appear in the observed set, and explicit PID checks report all five absent. The supplied device and fuser results are idle, and the original lock device66305/inode596823 is available. Exact live torchrun/worker PIDs were never captured because the short jobs finished before the live snapshot. This is a documented evidence limit: the final census supports no remaining recorded session/parent descendants, but cannot prove historical worker identity or rule out an unrecorded worker that detached and changed its identity/session. I make no live-worker PID claim.

The standard G2 uses explicit BF16, synthetic tokenizer/dataloader, deterministic clock, CPU kernel and training-microbatch shims. Its checkpoint check is not the ordinary unpatched helper. That helper has not run; raw saved-tensor equality and ordinary B1/B2/B3 x T256/T257 output identity/causality remain unqualified. Native correctness, official scoring, BPB, adoption and submission are not approved. Fresh machine preflight and separate scoped release remain necessary before any later operation.

Receipt SHA256:

- results-bundle.json: `198f3307d82a71bc22664bd8638915f1b90b3f9d24fd8a953c280edfbaa3550d`
- explicit-process-postflight.json: `2aed3ac7d384477c03caa313a9a9e6e778aab7785a4c184846d4dcb59a9b407f`
- retained-input-identities.json: `9b54804666da69af0cf1771424c7024587fa9b337c5e1499288c0a11f45fb23c`
