# MobileLLM depth-over-width suggestion: Supervisor triage (2026-09-25)

This is literature and prior-run triage, **not** a new experiment verdict or an
instruction imported from Csaba's email or research note. No code, host, data,
evaluator, or submission was changed for this suggestion.

## Evidence

- [MobileLLM (ICML 2024)](https://arxiv.org/html/2402.14905v2) reports that
  deeper, thinner approximately 125M-parameter models improve **zero-shot
  reasoning accuracy**. Its architecture ablation used 0.25T training tokens
  on 32 A100 GPUs; selected models used 1T. Its released 125M model has
  30 layers, width 576, 9 attention heads and 3 KV heads. The 30-layer,
  width-512 entry in Table 11 is an *ablation*, not that released model.
- Csaba's research note at
  `CsabaConsulting/trainium_frontier@d3b2dba66a0c19f2d56e49d123a66cae8883d42f:research/depth-vs-width-review-2026-09-25.md`
  independently concludes that this is a poor immediate transfer to a
  fixed-30-minute BPB objective. Treat its cost extrapolations as estimates,
  not measurements of a 30-layer Trn2 run.
- Our prior-stack, same-host 900-second fixed-width control was 6x640,
  1094 steps / 1.081255 BPB. At the same width, 7 layers gave 908 steps /
  1.093900 (+0.012645), and 8 gave 839 steps / 1.094958 (+0.013703),
  at that campaign's public-eval settings. Source:
  `research/trn2-experiment-results.md` section T0.2 and the associated
  run logs. A separate full-1800-second prior-stack comparison had 8x768
  at 1.052406 versus 6x640 at 1.043489. Those runs are not the current
  dense 6x1024 base, so they do not prove every deeper dense shape loses.
- Under a table-equipped prior stack, 8x640 improved step-matched BPB by
  roughly 0.003 but nearly doubled seconds per step. That is evidence for a
  quality/throughput trade, not a fixed-time win. Source:
  `research/trn2-experiment-results.md` section "Depth=8 at FIXED width".

## Queue decision

Do **not** displace the already selected dense A0 controls and A1/A2 table
tests with a 30-layer MobileLLM transplant. It has no current dense,
same-host, full-budget BPB evidence and is likely to lose many training
updates. Keep the existing depth-with-tables item conditional; revisit a
bounded, one-factor, full-1800-second shape proposal only if a *measured*
throughput change of at least approximately 15% or an adopted table stack
materially changes the depth trade-off. Compare against the same-host
current dense control, with identical data, seed, evaluation context, and
retrieved logs. No predicted BPB gain is claimed from the MobileLLM paper.
