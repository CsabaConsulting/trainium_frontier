# Startup-adjusted optimizer schedule: implementation self-check packet

This is a **scientific schedule change**, default OFF, on exact old-schedule
architecture3e79. It does not establish the cause of the cross-host BPB spread.
No Recursive stack, speculative throughput helper, host operation or native
launch was implemented. Independent/fresh review, full113M CPU qualification,
native correctness/cold accounting and any Operator assignment remain separate.

Proposal `f4918452` was committed before production source edit `7419927c`.
The only production file is `workflow/proposals/research-v2/train.py` in this
isolated worktree; canonical production files, prepare/evaluator/data are untouched.
Source SHA256: `36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb`.
Base Git `b4d83b5db66971f3c348837f04f53cba65aa6d33`, source SHA
`3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c`.

## Exact change and preserved accounting

16 added /2 replaced lines: one pure helper, one Boolean CLI option, one shared
`schedule_elapsed` local and LR/WD input substitutions. The helper is called only
when `--startup-adjusted-schedule` is present. Absent or explicit
`--no-startup-adjusted-schedule`, every previous expression gets exact `elapsed`.
No environment-controlled hidden default was introduced.

Before step5, helper process age is `elapsed + (budget_started - _PROCESS_STARTED)`:
the existing step-start sample, with **no new monotonic call**. It subtracts only
the existing900s allowance after adding the existing30s source safety margin,
clamped below at zero. At step5 onward it returns exact original elapsed. Thus
startup exceeding the cap still contributes optimizer progress, including the
unchanged conservative margin. It does not make later host trajectories identical.

Only optimizer LR and `wd_progress` use this local; existing beta2/momentum/DEMON
progress consumers already derive from `wd_progress`. Step-based momentum warmup
is unchanged. Selected profile keeps batch ramp/EMA/dynamic loss scaling OFF;
these are not stacked or redirected. No new parameter, tensor, buffer or
checkpoint-format/loader change. The flag is recorded in the proposal/argv, not
as an evaluation architecture toggle.

Original bytes can be reconstructed exactly by deleting the helper, option and
schedule local and reverting the two references. Static tests verify that full
reconstruction, not merely equality of a few timer lines. Process start, step t0,
initial cache refresh, five boundaries, official1800 budget, startup reset/excess,
predictor clearing, stop conditions,15s checkpoint reserve and exit stay unchanged.
This is not permission to exclude extra work or change official accounting.

## Bounded CPU evidence and reproduction

Completed: all six final arms PASS. Normal and `-O` comparators give identical
`tiny-summary*.json`; default/OFF are bit-identical to base for three updates,
and the two below-cap ON histories are bit-identical through six updates.
Every recorded gradient/missing-gradient status, parameter/update, optimizer and
checkpoint comparison passes. The1000s fake-delay arm retains positive excess
before and after the fifth boundary. ON differs from OFF as intended. These are
Implementer self-check findings only, not independent or native acceptance.

`test_static.py` imports no Torch/candidate. It extracts exact source AST functions
and optimizer scalar assignment block. Six test groups cover byte reconstruction,
Boolean default/OFF/ON, baseline/OFF scalar grid, cap/margin boundaries, recorded
startup-duration contrast, and original reset/stop AST plus over-cap replay.
Run normally and under `python3 -O`; neither relies on Python assert for a gate.

`run_tiny.py` invokes the unchanged standard G2 driver sequentially via a narrow
observer extending the existing all-gradient/cache observer. Support bytes are
authenticated in `g2_schedule_observe.py`; source bytes are authenticated before
launch. Fixed local runtime `/home/poojith-devan/venv/bin/python`, Torch2.13 CPU,
OMP/MKL/OpenBLAS1 per rank. Only one four-rank CPU arm exists at a time.

Model8x128,4523316 parameters/65 parameter tensors; each actual forward sees
synthetic2x128 tokens. Architecture component flags preserve eight-layer skip/
partial-attention wiring at reduced width; **do not use `--unet-8x1024` locally**,
because it forcibly restores full width. The selected real MLP custom backward is
eligible at256 tokens; raw observations check8 such autograd nodes per step.
This is not an exact113M model, native kernel, full hardware microbatch or quality
run. Memory preflight requires6GiB locally available; observed available RAM was
about8GiB before starting. No memory-peak measurement is claimed.

G2-only shims: CPU/gloo/no-compile, BF16 compute dtype override, NKI CE install
disabled, synthetic tokenizer/data, import-only PyArrow stub that rejects real
Parquet reads, microbatch slicing, deterministic1ms-per-call clock. The standard
data-free prepare stub has a two-step startup exclusion; this test restores the
candidate module's imported count to the real pinned prepare's five, explicitly
recorded in every final receipt. Neither prepare file nor production source is
edited for this override. The additional
observer advances only that fake clock after the first optimizer update by an
explicit779/790/1000seconds, simulating compilation latency without waiting or
moving actual training work. First update elapsed was already sampled, as in the
real loop. These integer delays are controlled synthetic histories, not exact
replays of the real hardware timestamps.

Arms: base/default/OFF3steps with790s fake delay; ON6steps each with779/790/1000s
fake delay. `tiny-commands.json` stores exact executed argv/environment. Raw
`tiny-*.json` include every rank's losses, before/after-clip gradient digests,
missing-gradient status, all65 post-update parameter digests, optimizer state,
scalar groups, cache observations and checkpoint payload/file digests. G2 also
checks rank consistency and rank0 loader finite/repeated-output behavior.

`compare_tiny.py` requires base/default/OFF exact equality of those observations;
the two below-cap ON histories must match through6steps including checkpoint.
ON1000 must retain positive charged-excess effect on WD before and after step5.
ON and OFF intentionally differ. Test summaries are implementation self-checks,
not independent G2 approval, native causality or a BPB result.

Commands, after reserving the tiny CPU slot:

```text
python3 research/v2/lanes/startup-adjusted-schedule/test_static.py
python3 -O research/v2/lanes/startup-adjusted-schedule/test_static.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/poojith-devan/venv/bin/python -B research/v2/lanes/startup-adjusted-schedule/run_tiny.py
python3 research/v2/lanes/startup-adjusted-schedule/compare_tiny.py
python3 -O research/v2/lanes/startup-adjusted-schedule/compare_tiny.py
```

The recipe intentionally refuses to overwrite existing arm receipts. For a new
reproduction use a fresh packet-output directory/worktree with the pinned sources
and support paths available; preserve existing raw evidence. Full-size CPU needs
its own safe64GiB host packet, not a laptop shape substitution.

## Failed attempts preserved

1. Initial baseline observer failed before update0: it accessed
   `OwnedOptimizer.param_groups`, while the actual wrapper stores these at
   `self.optimizer.param_groups`. Raw `failed-attempt1-base.json` is retained.
   Only the test observer was corrected; no candidate/baseline source changed.
2. An early comparator invocation before the sequential ON arms finished failed
   closed with FileNotFoundError for `tiny-on790.json`. No tensor test or source
   was changed by that premature read. Final comparison runs only after all arms.
3. The complete first six-arm sequence used standard G2's two-step startup stub.
   All arms passed G2 and default/OFF and ON-history identity, but the explicit
   first-five WD assertion failed at step2. All six raw receipts and commands
   are preserved in `attempt2-stub2/`; they are not five-step evidence. Only the
   test-only imported startup count was corrected to the real five before rerun.

## Remaining gates

Independent/fresh source review and their own tests; exact-full-parameter CPU
gate; disabled exact-source native diagnostic proposal/guard binding; actual
first-five/cold graph cost, finite updates and through-exit budget. No native
binding, quality-run command, full-run/eval automation, submission eligibility
or accepted speedup is created by this packet. Original architecture results
remain intact and the observed BPB spread has no established causal attribution.
