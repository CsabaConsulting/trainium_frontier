# Basebcbd source and reduced-CPU qualification review

Verdict: **PASS for the bounded technical source/reduced-CPU prerequisite to later CPU qualification.** No new source defect found. This is a historical-context verifier's technical audit, **not** the user-required fresh isolated source review. Full117M standard G2, native execution, cold startup, performance and final source acceptance remain pending. No host, model-source, launch binding or ledger was changed.

## Exact source boundary

- Base commit `a4aa4351789115a2d0aa947d9ca00ff689f9bd0b`; source SHA256 `6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`.
- Candidate commit `b635c8b0cf2a10e1bcdd983e86192672ed6ad56f`; source SHA256 `bcbd148dd51b5a014289cbecf3f0e684e35cf537e9588c878e15b265fcbfd522`.
- Path in both commits: `workflow/proposals/research-v2/train.py`.
- Independent whole-file AST equality holds after removing exactly the new parser call and the new first-step conditional. Four executable lines added; no removals. The parser defaults OFF and supports explicit OFF/ON.
- ON refresh runs after `synchronize(device)`, `t0 = time.monotonic()` and zeroing, before the first forward. It therefore remains inside the timed step; no timer/startup allowance change. `cached_linears` contains existing `Linear` modules only when weight caching is enabled. The unchanged refresh uses `self.weight.to(dtype)`, without `detach` or `no_grad`.
- Normal end-of-step/evaluation refresh, model parameter definitions, optimizer, Recursive schedule, batch size, data, save/load and evaluator paths are unchanged. This flag is an execution option, not a model/checkpoint configuration change. The earlier `--compile-only` branch does not execute the new loop block and is not evidence of this flag's effectiveness.
- BF16 casts from FP32 masters remain differentiable tensors. The selected custom MLP path obtains the same cache through its existing weight helper; this audit does not claim a new native compiler proof for that path.

## Independently checked evidence

`check.py` authenticates all four historical `initial-cache-refresh/tiny-{base,default,off,on}.json` raw receipt hashes and their exact source identities; it compares contents rather than relying on their PASS labels. All four ranks have three identical losses, and all 62 parameter records (including missing-gradient status) agree before clipping, after clipping and after the update at every observed step. Optimizer-state digests agree. Rank0's final parameter digests, checkpoint-payload digests and entire checkpoint-file SHA256 agree across arms:

`1bc4752a2f82da10dd944c1a58bcdc2859ccb4bedea59222d3e77bb41efe098c`.

ON records 40 additional refreshes per rank before forward 1, all with gradients enabled and connected BF16 caches. The other arms start with 40 empty caches. Later refresh totals are 160 ON versus 120 other arms. Each forward graph contains six custom MLP backward nodes; the observer field `mlp_custom_backward_calls=18` counts graph nodes, **not an executed-backward hook counter**. Raw checkpoint reload records are deterministic and successful; the archived checkpoint itself was not newly loaded in this audit.

These receipts use six layers of width128, four CPU ranks, three steps, physical synthetic inputs2x128/rank, a synthetic prepare module, CPU BF16 shims, no compile and a fake clock. They include an intrusive gradient/cache observer. They do not establish exact-size standard G2, real tokens, native execution or time-budget compliance. Architecture receipts were not used as base approval.

## New independent tiny reference

The verifier wrote `check.py`, AST-extracted the exact unchanged `Linear` and the candidate's new conditional, and compared default/OFF/ON against a separately expressed uncached functional projection. Four BF16 cases cover ordinary/transposed weights and one/two sequential micro-backwards, each over three steps. Output, input gradient, FP32 master gradient, simple momentum state and updated weights are exactly equal. Master identity and weights-only state roundtrip are checked. This simple optimizer is a reference, not a claim to re-run production Muon.

Both normal and `-O` executions passed **784 explicit checks** with Torch2.13.0+cpu and one thread. Maximum weight size is35 elements; no full training module or model was imported. Normal1.44s/240508KiB peak RSS; optimized2.36s/242672KiB. The CPU slot was released immediately afterward. The initial overly broad FP32-cache fixture failure is retained in `initial-test-error.json`; it was a verifier scope error involving an unchanged FP32 identity-cast behavior, not a new candidate defect or an acceptance-threshold change.

Commands, from this worktree:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /usr/bin/time -f 'elapsed=%e peak_rss_KiB=%M' /home/poojith-devan/venv/bin/python research/v2/verification/20260926-base-cache-source-cpu-independent/check.py --repo /home/poojith-devan/Desktop/Trainium --tensor
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /usr/bin/time -f 'elapsed=%e peak_rss_KiB=%M' /home/poojith-devan/venv/bin/python -O research/v2/verification/20260926-base-cache-source-cpu-independent/check.py --repo /home/poojith-devan/Desktop/Trainium --tensor
```

## Next gate and authority

The mechanically repaired CPU-only packet remains separately reviewed in commit `1b8267dd3628ad1b216cdf0ce29c3efa2a0ad244`, exact implementation repair `2cf504d9b763e98e7130447d52fc93799e26db50`. This result supports continuing CPU qualification after applicable independent-source policy/assignment, not bypassing it. The existing packet is bound to trn22; there is no trn21 relocation or launch authorization here. Operator remains sole host authority under the common lifetime flock plus fresh ps and neuron-ls checks.

`prepare_fresh_packet.py` creates a new non-repository input directory containing **only** the full-context exact diff, neutral lane specification, raw four-arm receipts and manifest. It excludes this review/test/results, all implementation notes/tests, repository/session history and credentials. It does not launch or resume a model session. `fresh-packet.json` records the actual prepared path/manifest. No fresh reviewer has run or issued a verdict for this base source in this task.
