# Independent full-parameter CPU result audit

PASS for the prescribed standard-G2 identity scope. This is not native, cold-startup, speed, or quality acceptance and grants no host-launch authority.

Frozen raw archive: `2718bc5c2de31a73ef01ffaad753eb4da0b9fe45`, exclusively `research/v2/operator/20260926-trn22-base-cache-cpu/results-complete/`. The incomplete earlier `results/` transfer was not used. All 88 host artifacts match their recorded SHA-256 and byte count; retrieval manifest SHA-256 is `09631f193308158b82fdc896df9e7527a9f08f83dc9204330424a50b93bb282c`.

Reference `6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3` and candidate `bcbd148dd51b5a014289cbecf3f0e684e35cf537e9588c878e15b265fcbfd522` were run sequentially as reference, default, explicit OFF, and ON. Source/G2/prepare pins, actual argv, reviewed CPU-plan/request/driver bytes, original lock inode 66305/832589, both fresh under-lock idle checks, successful exits, resource floors, and quiescent postflights are authenticated.

The four 117,453,580-parameter / 62-tensor models have exactly equal 12 recorded losses, all 62 final parameter digests, and optimizer ownership plan `b89d62227c844eb13706fbe81bddfb9b8b7db5fefcd57b4e7d0ea3213dca5031`. All 16 individual rank reports exactly equal their aggregate entries; all 62 tensors changed and cross-rank mismatch lists are empty. All three original unchanged-harness comparisons returned exit 0 and `identical:true`; the independent audit recomputed equality directly rather than accepting those labels.

Each checkpoint reload is deterministic at [1,256,8192]. All four retained checkpoint file identities equal SHA-256 `51bff5a31ab16d04e3336412c00972f72520e592da6f6fa598e0abbfa259cadc`, 469,834,849 bytes. The auditor did not download or load checkpoint tensors; this is authentication of retrieved file-hash receipts plus the G2 reload observations.

Important limits: CPU Torch 2.11, BF16 shim, compilation disabled, deterministic fake clock, synthetic tokenizer/data, and actual 2×256 tokens per rank over three updates. Actual global update is 2,048 tokens, not the production 131,072; nominal context1024/pack32 does not make this a production-sized token/update test. NKI CE is disabled by the harness. No RMSProp table updates ran (`calls:0`), and these receipts do not expose full intermediate gradients or per-step optimizer-state equality. No native compiler, cold-startup, performance, real-data learning, or validation claim follows.

`audit.py` is an independent stdlib-only verifier: 290 checks pass normally and under `python3 -O`, with identical results in `RESULT.json`. No hosts, tensors, candidate edits, or ledger writes were performed. The separately reviewed cold-five binding still needs Supervisor assignment and Operator fresh preflight; the full-parameter CPU prerequisite is now satisfied within the exact scope above.
