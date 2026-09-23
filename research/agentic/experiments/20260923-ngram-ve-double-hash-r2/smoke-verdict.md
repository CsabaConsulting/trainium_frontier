# Verdict: DISCARD — 20260923-ngram-ve-double-hash-r2 (smoke)

- Hypothesis: The hashed bigram/trigram value-embedding tables are the single largest measured win in train.py (-0.0152 bpb, train.py lines 621-650) and the run is data-bound (~2140 steps = 1 epoch; the ledger shows every higher-step variant crossing the epoch boundary got WORSE, so per-step quality, not throughput, is the open lever). Each table row (ngram_mod ~1M rows, prime) is shared by several distinct contexts in an 8192-vocab corpus with many millions of distinct bigrams/trigrams, so every lookup carries collision noise. Averaging two gathers from the SAME table at two independent hash positions (hash-embedding / Bloom-style probing, no new parameters) makes each context's own signal add coherently across both rows while the colliding contexts differ between rows, raising per-lookup SNR. Predicted: lower val_bpb at matched steps with a small (<5%) step-time cost from one extra gather per table.
- Single causal change: One architecture change confined to train.py: a new module-level boolean toggle USE_NGRAM_VE_DOUBLE_HASH (default True on the candidate branch; env NEURON_COMPETITION_R1_NGRAM_VE_DOUBLE_HASH='0' disables; CLI --ngram-ve-double-hash/--no-ngram-ve-double-hash following the existing --ngram-ve-trigram boolean-pair pattern; persisted and restored exactly like NGRAM_VE_TRIGRAM through ARCH_TOGGLE_NAMES, with a MISSING key restoring to False). When True, every n-gram VE table lookup (bigram and trigram, on every table-bearing layer) becomes 0.5*(table[idx0] + table[idx1]) where idx0 is today's index computed by the unchanged GPT._ngram_hash_indices code, and idx1 is a second index computed from the SAME raw token ids (prev,cur / prev2,prev,cur) with a disjoint set of bounded primes plus a XOR salt in the same int32 arithmetic, reduced by the same self.ngram_mod. No new parameters (592.9M unchanged), no table-size, layer, optimizer, LR, gate, precision, or kernel change. With the toggle False the code path is byte-identical to 9a4caa7b.
- Flags: ``
- Commit: `05422cf0503821f9031e349613eb38db1af8f5c4`
- Box: trn21 (`ip-172-31-34-85`), shape LNC=2 nproc=4, seed 42
- Label: `al-20260923-ngram-ve-double-hash-r2-smoke-r1`; control label: `al-20260922-control-trn21-lnc2x4-9a4caa7b-smoke-r1`
- Budget: num_steps=5000 max_train_seconds=300
- NEURON_CC_FLAGS: `--optlevel=1`

## Reasons
- smoke step rate 1.1436 < floor 1.1513 steps/s

## Metrics (public dev shard, evaluate_bpb, same box/day/commit)

| metric | candidate | control |
|---|---:|---:|
| public_val_bpb | 1.171049 | 1.165855 |
| num_steps | 325.0 | 334.0 |
| steps/s | 1.143560872624912 | 1.1748153359127682 |
| params (M) | 592.9 | 592.9 |

These are development measurements, not leaderboard scores.

## Analyst

DISCARD at the smoke tier: candidate throughput was 1.1436 steps/s, below the supplied floor of 1.1513 steps/s.
The double-hash candidate was 2.66% slower than control (1.1748 steps/s), completing 325 versus 334 steps despite nearly identical training time.
Validation BPB worsened from 1.165855 to 1.171049, a +0.005194 delta, exceeding the listed 0.003 tolerance; the candidate cleared the 310-step minimum.
Both runs used 592.9M parameters, seed 42, the same host and compiler settings, and passed causality checks.
The quality comparison is confounded by unequal training exposure: the candidate processed 42.6M versus 43.8M tokens, so the BPB regression cannot be attributed solely to double hashing.
Runs occurred on different dates, leaving possible runtime variability uncontrolled.
The supplied floor is approximately 98% of control throughput, not the listed 93% ratio; this metadata inconsistency does not change the deterministic DISCARD verdict.
