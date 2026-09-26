# Clock36 + Recursive: new source, CPU implementation only

This proposal authorizes no host work. The earlier blocked flag-only finding at2dd6b317 remains unchanged. The selected task is now a **new, default-OFF source port**, to be implemented by a distinct Implementer and independently reviewed. Do not edit launch commands, submission files, evaluator or active source/ledger/state.

## Exact inputs and minimal patch

Base: `7419927c7417b5666a984ec9c2b849feadabaaa9:workflow/proposals/research-v2/train.py`, SHA256 `36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb`.

Donor: `fe0969fd311d654e143dc6113614416c8970a424`, same path, SHA256 `75b944f7caf7f16a8392e720c135d7a6436f4c1ef980a6bb0c9ac15bafc9f70f`. Its three schedule functions are AST-identical to adopted6ef7 at `a4aa4351789115a2d0aa947d9ca00ff689f9bd0b`, SHA256 `6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`. The exact function AST hashes are in the proposal; this static provenance observation is not a child correctness PASS.

Port the donor's small schedule diff, not the whole donor file. Add default-False `RECURSIVE_SCHEDULE`, its BooleanOptionalAction and saved toggle, the three schedule branches, device-beta2 scalar choice, and per-group LR selection. Preserve donor attribution. No new global defaults, architecture, optimizer math outside the intended package, model/eval routes or helper refactoring.

The necessary integration difference from the donor is that **both** calls to `lr_multiplier` take base36's existing `schedule_elapsed`, not raw `elapsed`. Existing `wd_progress` already uses that clock; it continues to feed momentum, beta2, WD and any already-enabled optional Demon behavior. For non-Muon groups use the donor's non-Muon LR route; do not silently restrict it to one named optimizer kind. `no_lr_decay` keeps precedence. ON ignores the old LR quantization path exactly as the donor does; OFF retains every old fallback.

## Timebase and intended schedule semantics

Keep `startup_adjusted_schedule_elapsed` byte-identical. With clock ON at steps0–4, use `max(0, elapsed + budget_started - process_start +30 -900)`; at step>=5, use existing `elapsed` after the unchanged reset. Clock OFF uses the original elapsed path. No new monotonic reads, timer resets, stopping rules or exemption are introduced. The official900-second capped startup allowance,30-second source margin and1800-second budget are not retuned.

Using clamped progress p from that selected elapsed/1800, retain the donor operations and their order:

- Muon LR: warmdown fraction0.95; non-Muon LR:0.65; both floor0.05. Preserve existing step warmup first.
- Muon momentum: step warmup0.85→0.95 over300 steps, then donor quadratic cooldown to0.79 when p>0.05. If large charged startup makes this branch override early step warmup, retain that existing behavior; do not invent a fix.
- Muon beta2:0.95 through the initial5% then linearly to0.97. Supply its complementary value through the existing device-tensor scalar path when Recursive is ON, preserving all fallback behavior when OFF.
- WD remains base36's existing linear progress-dependent behavior. Flat-LR groups keep their old precedence. LR peaks, seeds, batch and all other scientific flags stay unchanged.

Use donor expressions verbatim where possible; algebraically equivalent rewrites can change floating-point rounding and are not part of this port. Budget/reset/stop/logging/record_step_boundary/checkpoint code remains exact36, including recent-step predictor reset and checkpoint reserve. Save only the new schedule toggle through the existing mechanism; loader/model math and `weights_only` behavior must not change.

## Independent CPU evidence to require

Tests are planned, not performed by this Researcher. Serialize one-thread tiny tensor jobs; no full model on the laptop. Full113256724-parameter standardG2 is a separately bound CPU-host gate, not permission granted here.

| Clock flag | Recursive flag | Exact claim |
|---|---|---|
| OFF | omitted or explicitOFF | Default/explicitOFF child numerics equal exact36 clockOFF |
| ON | omitted or explicitOFF | Child numerics equal exact36 clockON |
| OFF | ON | Schedule outputs/routing equal donor75b9 at identical elapsed; model/update equivalence checked under the same deterministic inputs |
| ON | ON | Donor schedule applied to independent capped-clock reference; intentionally not identity to36 or old75b9 |

The reviewer must write/run its own reference, not merely rerun Implementer tests. Minimum focused claims:

1. **Scope and scalar oracle.** Check whole diff/AST and default flags. Independently evaluate progress0, values immediately around0.05 and0.35,1 and out-of-range clamps; steps0,1,4,5,6,299,300,301; varied warmup and flat-LR groups. Verify the three donor formulas/routing, not just helper counts.
2. **Clock boundaries.** Fake process ages779,790,869.999,870,870.001,900,1000 seconds and arbitrary process-start offsets. Compare helper/callsite against independent formulas at steps0–4 and>=5 after the real fifth-boundary reset. Verify both LRs, momentum, beta2 and WD consume the selected timebase; budget variables/stop decisions remain exact36 for identical clocks. Tiny harness must explicitly record production startup exclusion5: existing standardG2 stub2 is not proof.
3. **Actual tiny updates.** With fixed clocks, seed and small deterministic tensors/model, compare default/OFF losses, gradient presence/values, all updated parameters and optimizer state against36. Check ON actual updates against a donor-formula reference with the same selected clock, not against unchanged OFF weights. Include >=6 updates to cross the boundary; finite-only ON is insufficient. Check the device beta2 input route separately by extraction/mock or CPU capture without pretending this proves native capture.
4. **Save/load and causality.** Compare tensor/state content and logits, not serialized checkpoint-container bytes after adding a metadata key. Record the expected `RECURSIVE_SCHEDULE=False/True` metadata difference and preserve loading older checkpoints. Check trained-checkpoint round trip and existing causal/state-invariance rules with no eval adapters.
5. **Negative controls.** Deliberately supply raw elapsed to only non-Muon LR, reset optimizer progress to zero despite startup>870, use clock adjustment after step5 incorrectly, ignore flat-LR precedence, or leave changing beta2 on the Python-scalar branch. Each test must reject the corresponding wrong result. Keep original failures and raw outputs.

Fresh review input remains only complete diff, neutral lane spec and raw gate receipts; exclude Implementer narratives/conclusions/tests as task input. Archive the reviewer's own executed test, actual session/model, input manifest and verdict. Source/CPU success is not native success.

## Later gates and result interpretation

After new exact-byte source review and exact-size CPU qualification, a separately frozen/reviewed/assigned true-cold five-step test must measure child launch→fifth<=900s (target870), finite five updates, exit0, retained checkpoint and valid official through-exit accounting. KeepMAX1800 and intended model/optimizer flags. Parent36 cold845.627s and75b9 cold831.422s do not qualify their combination. A warm-cache replay is not cold proof. Full quality and attached fresh20M need separate release after accepted child cold evidence.

The newly authenticated trn22 formal6ef7 control0.9912010195941572 makes clock-only36's0.9883821509811542 delta−0.002818868613003; Supervisor reports unchanged-selectorNOMINATE. This selects the follow-up, not adoption or a predicted child score. The trn21 exact36 replica remains live at proposal time. Prior architecture+Recursive75b9 without the clock scored0.9896621160468637 and wasDISCARD versus formaltrn21control0.9899528600562257; preserve it. Earlier schedule-arm scores remain historical arms, not controls.

Any future full result reports both incremental same-host delta versus clock-only36 and formal delta versus genuine6ef7 control, keeping row roles intact. Unchanged selector, stricter two-host mean threshold and human adoption/submission rules remain. No host/launcher/ledger/state changes, command changes, claim of parent qualification transfer or immediate native authority are part of this proposal.
