# Independent correction to cycle 21:53

This addendum supersedes the count caveat and C2-next language in the frozen cycle a22e010d, brief353bc927 and NEXT_ARMS021788b6. Their original bytes are retained as evidence.

The exact `check_due.py` (SHA920c3840ae077e0f2109bbbbacef419300dbec3ff77ff591cd878d6a10986f9a) function `result_ids` was executed against ledger025cdcdd, 103 rows. It returns **97 eligible unique result IDs**. Its literal predicate is `public_20m is not None OR truthy raw_archive OR status in ('failed','evaluated','complete','completed')`, followed by a truthy `manual_job_key or label` and exact-key uniqueness. This is not semantic deduplication. Six rows fail that predicate; all remaining97 have distinct keys:

- Row7: human_aborted_not_quality
- Row10: native_correctness_failed
- Rows16,19,20: startup_diagnostic_complete
- Row74: MEASURED_TIMING_GATE_FAIL

Those six lack non-null public_20m and truthy raw_archive. This explains cadence behavior; it does not reinterpret their actual outcomes or change the cadence implementation. Root alone updates its cursor.

**C2r1 is already dispatched/running per Root**, not next. Local dispatch44ed6d948a68a7ef3fd86d8cf7219a10a5a377659512ca1391d274a3c3824e3c records DETACHED_DISPATCH_STARTED at2026-09-27T16:23:43.477081Z (21:53:43IST). This local archive proves dispatch, not continued liveness. Root explicitly logged the sequencing deviation: original reviewed C2r1 after actual control plus ended-A supplemental clearance, without waiting for unavailable decoded metrics. No decoder-completed or profile-informed selection claim is made. Transfer/decode remains unfinished Root-owned work, not authority to overlap Explorer with training.

Own arithmetic recomputed A/B ratios and 512/531 phase1 steps ×131072 =67108864/69599232 tokens. These stopped phase1 totals do not project full-budget throughput. Speed stops are not quality losses; no public20M, startup or peak-memory result is invented.

Primary sources were reopened. The [compute-optimal paper](https://arxiv.org/abs/2203.15556) supports joint model/token budgeting, not a prediction of this recipe's optimum. [AWS Summary Viewer](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-explorer/overview-summary-page.html) distinguishes estimated bandwidth utilization, DMA-active time and spill metrics, excluding host overhead from on-device duration. [AWS profile analysis](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-explorer/how-to-analyze-profile-output.html) documents processed Parquet/schema inspection; it does not certify locally installed schema or selected-warm attribution. Original cautious conclusions are supported.

Verdict: PASS with this mandatory correction. Zero new arms, closed-lane revivals, evaluator/data/legality changes or launch authority. No host, source, queue, ledger or cursor changes.
