# Architecture 93ae fresh-public-20M result audit

**PASS for result authentication; quality verdict DISCARD.** This is the completed original architecture run, not the proposed cache-refresh rerun, not the adopted Recursive schedule, and not submission evidence. No host action, model import, ledger/history mutation or adoption was performed.

## Result and matched control

Candidate `fs-20260925-l2-unet8x1024-93ae-trn22-full-r1`, arm `L2-unet8x1024-93ae`, exact source `93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5`, completed 2,246 updates and fresh public-20M/context-1024 evaluation:

| Evidence | Value |
| --- | ---: |
| Candidate BPB | 1.017340669168314 |
| Same-host c5c control BPB | 0.991513958330512 |
| Candidate minus control | +0.02582671083780197 |
| Raw total nats | 58,854,764.0 |
| Raw evaluated bytes | 83,462,186 |
| Raw evaluated tokens | 20,971,520 |
| Evaluation seconds / timeout | 669.8979084530147 / 1200 |

Independent arithmetic `total_nats / (num_bytes * ln(2))` exactly reproduces the reported BPB. The matched control is `fs-20260925-v2-c5c-a0-control-trn22-r10`, source/base `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`, 3,633 steps. Its raw evaluation hash and BPB arithmetic were independently checked. No Recursive or different-host control is substituted.

Unchanged `workflow/tools/decide.py` (SHA-256 `fa4a3d3dd3697093f93ac73b5c95335a7c1efa12534589ed3c3e2b64940ca0ac`, rules `v2-2026-09-25-general-fixed-budget`) queried the full current 11-row ledger plus one in-memory candidate. It returns `DISCARD`, reason `single-host delta >= +0.0005`, with no invalid runs. The 0.6182 step ratio, first-two-step timing spikes, missing exact measured startup and unproven cold start are diagnostics; they do not cancel the negative fixed-budget quality result. Ledger SHA-256 `6cf6b8850ca50be18d649daf9e37e6adc44e2d7d0cf32503532c1228bbeed9b2` and selector bytes remained unchanged.

## Identity, completion and timing

All nine raw artifacts in operator archive `72da6811` (canonical import `851e8835`), under `research/v2/operator/20260925-trn22-l2-93ae-cpu-r1/native/final-evaluation/`, match the retrieval hashes and sizes. Training source, unmodified prepare SHA-256 `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`, exact argv/environment, label and the separately invoked evaluator match the bound native job. Training/evaluation exits are zero; terminal is COMPLETE at `2026-09-25T18:37:24.202476+00:00`. Retrieval reports no relevant processes or neuron0 holders.

The evaluator starts after training exit, uses the exact candidate and `out/final.pt`, batch size 1, context 1024, 20,971,520 tokens, and a 1200-second timeout. The 453,048,594-byte checkpoint was retained on the host and freshly rehashed on retrieval, not downloaded. Its SHA-256 is `fb41241f812b0a357a36c18e0cbfbf81adc1df26ababf86d90fa40f51ded7c6b`. The post-training hash and post-evaluation retrieval hash agree.

The timing conclusion from independent exit receipt `21afce7cefdf446bfb6e902da5c5d3605d8fde42` stands. The same driver records training START `17:41:38.131774` and END `18:26:01.637384` UTC, enclosing the process and cleanup: 2663.505610 seconds. The first five disjoint rounded step intervals give a startup lower bound of 1544.745 seconds, proving saturation of the allowed 900-second exclusion. Thus the conservative charged-through-exit UTC enclosure is at most **1763.505610 seconds**, with at least 36.494390 seconds of headroom. This is not a newly invented independent monotonic clock reading or a precise fifth-step startup timestamp.

The separate source summary reports training 1784.6 seconds, charged estimate 1785.0, startup allowance estimate 900.0, and `over_budget: False`. Those source estimates are not substituted for external wall measurements. Startup excess was charged, not refunded. Fresh evaluation time is outside training. Existing research cache paths were reused: this is **warm research budget compliance only**, not cold compilation or official submission certification. No new native causality observation is fabricated (`causality_passed` remains null).

The human-directed compile fix plus one rerun is a new intervention; it does not relabel this run as promising or erase its DISCARD result. Adoption remains unproven.

## Independent checks and reproducibility

The attached stdlib-only `audit.py` passed **57 checks in normal and optimized Python**. `AUDIT_RESULT.json` retains the exact query-only candidate, selector output and raw artifact hashes. The audit uses explicit checks, not optimizable assertions. It allocates no tensors and imports neither candidate nor hardware tools.

```text
python3 -B research/v2/verification/20260926-architecture93ae-result-review/audit.py
python3 -B -O research/v2/verification/20260926-architecture93ae-result-review/audit.py
```

The first audit attempt expected a checkpoint path inside the hash-only checkpoint file. It stopped locally with KeyError before the selector query; the path is actually in retrieval and evaluator argv. The auditor was corrected to cross-bind those records. This was an audit-helper schema error, not an experiment failure or a change to evidence.
