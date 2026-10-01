# Round37 exact saved-checkpoint identity evidence

Status: PASS_R37_NATIVE_SAVED_CHECKPOINT_IDENTITY. This document summarizes previously executed, authenticated native evidence. Package assembly made no host call, model load, tensor execution, data evaluation or retraining call.

Candidate source 6497 and original b585 are pinned in the JSON below. The same checkpoint was unchanged before and after evaluation and matches the original cold run. The new fresh process evaluated 20,971,520 tokens / 83,462,186 bytes at context 1024, batch 1 on Neuron. Total nats 56,499,256 produce BPB 0.9766242696436925, exactly the original BPB. Eval duration 692.061454049s, timeout 1200s, over-timeout false. The wrapper returned 0 with zero training boundaries. Existing research evaluation caches were reused; this is not a cold-cache evaluation.

Independent final verdict (verbatim JSON):

```json
{
  "verdict": "PASS_R37_NATIVE_SAVED_CHECKPOINT_IDENTITY",
  "source_sha256": "6497d163d930275dfbc0bb494fbb546456478e3751119dcc1baf918f5638f9d3",
  "original_source_sha256": "b585d0a647962009ae13068fd0f823e17fd19392b5d0dbf1087e6e5f40e48742",
  "checkpoint_sha256": "7ba52e1d9c390005295512bc28b28e605710f54bc1d6b9efd073ae7720aa93f7",
  "checkpoint_bytes": 453044690,
  "checkpoint_before_after_and_original_identity": "EXACT",
  "prepare_and_harness_retained_identity": "EXACT",
  "results_bundle_sha256": "8f9e6f172ba31f7f8e001573fede97d2ee0843970c8a3faf0f7d69392618bd31",
  "released_packet_sha256": "1e809c3e1cc07d0ce78440522cacf2a6a81dc7814e1788b17a5401958a6ebf7d",
  "reference_bundle_sha256": "1a93745b2b3320471af8633e5837571ff0a2e4b8bd6540a793ca02ddf201e448",
  "bundle_members_verified": 33,
  "packet_members_verified": 14,
  "packet_members_equal_to_runtime": 13,
  "reference_members_verified": 42,
  "eval_device": "neuron",
  "batch_size": 1,
  "context": 1024,
  "eval_tokens": 20971520,
  "eval_bytes": 83462186,
  "total_nats": 56499256,
  "original_val_bpb": 0.9766242696436925,
  "candidate_val_bpb": 0.9766242696436925,
  "absolute_bpb_delta": 0,
  "maximum_absolute_bpb_delta": 0.00001,
  "eval_seconds": 692.0614540490787,
  "eval_over_timeout": false,
  "guardian_events_verified": 55,
  "guardian_and_final_idle": "PASS",
  "checker_command": "python3 -B research/v2/verification/20260928-r37-eval-path-independent/check_native_identity.py",
  "checker_exit_code": 0,
  "checker_sha256": "70206e611ac3f1563c4afa7a8cff7eab761beaf7f965d51aac0a494ff5430f9c",
  "training_performed": false,
  "host_calls_by_verifier": 0,
  "quality_improvement_claim": false,
  "submission_authorized": false
}
```

The independent review verified all 33 R37 archive members, 14 released-packet members and 42 original-control members, plus unchanged source/checkpoint/evaluator/harness identity and original Guardian terminal/idle evidence. The current package assembly separately authenticated each retained member of the R37 archive and all eight parent result archives, and checked candidate bytes against the canonical proposal source; it did not repeat the native test or claim a new independent review.

Source proof: the independent source reviewer reconstructs the exact candidate as seven authorized edits (six docstrings plus MLP predicate/comment), with training routes identical. Its 26,040 metadata/cache/shape pairs and 52,080 symbolic forward calls establish routing/expression identity, not native floating-point identity by themselves. The later actual native result clears the one-checkpoint identity gate. Neither result is a universal bitwise or new-quality guarantee.

Primary receipts:

- R37 archive: `research/v2/operator/20260928-r37-saved-checkpoint-eval/artifacts/results-bundle.json`, SHA256 `8f9e6f172ba31f7f8e001573fede97d2ee0843970c8a3faf0f7d69392618bd31`.
- Final native verdict: `research/v2/verification/20260928-r37-eval-path-independent/NATIVE_IDENTITY_VERDICT.json`, SHA256 `9fc25ab1aec025a9f711a58a64434200345990ae9ad85ab746e7558ec2be7d86`.
- Final review: `research/v2/verification/20260928-r37-eval-path-independent/NATIVE_IDENTITY_REVIEW.md`, SHA256 `ac08da5446806555cd13c05cd51e0ddc77398692762d7e907cc946b8650893a3`.
- Original checkpoint run: `fs-20260928-conv-dw-store-filler-full-trn22-r3`; archive `research/v2/operator/20260928-conv-dw-store-full-trn22-filler-r3-preparation/artifacts/results-bundle.json`, SHA256 `1a93745b2b3320471af8633e5837571ff0a2e4b8bd6540a793ca02ddf201e448`.
- R37 eval log member SHA256: `ff45263cebf9fa4207dc00c55a4f8f067f4b37d03b2e722c2808bb1a10f1f51e`.
- Before/after checkpoint receipt members: both SHA256 `593075c7d7e4af2ca02582a31593d401769408d5a17d4056de5e17dd0b0c6552`.
- Protected evaluator SHA256: `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`; evaluator and harness were unchanged and are not packaged.

The full R37 member manifest, parent-run receipts and origin-document hashes are in cold-evidence.json. The native-checker historical command is evidence only; package assembly did not rerun it. All timestamps, launch/idle and source-git facts are archived historical receipts, not fresh host admission.
