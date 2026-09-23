# Primary-source research map — 2026-09-23

This is a ranked research queue, not an experiment authorization and not a result. Every hardware
run still requires a unique proposal, independent verification, the machine protocol, seed 42, a
same-host control, and durable retrieved evidence.

## Current measurement

`20260923-ngram-ve-cooldown-lr-flag-only-default-shape` completed its smoke process successfully at
334 steps and `1.168643` public-development BPB. The same-host smoke control is 334 steps and
`1.165855`. The candidate therefore preserved throughput but was `+0.002788` BPB worse at the short
horizon. Its proposal deliberately uses smoke only as a correctness/throughput gate; the controller
and independent Analyst must collect and classify it before any full promotion. This is not an
improvement.

## Ranked research queue

### 1. Restore the trained local convolution to the same block's attention path

This is the highest-information, lowest-complexity architecture lead.

Repository evidence:

- Before commit `7518d91e`, `Block.forward` applied
  `x = x + local_conv(norm(x))` and then passed that updated `x` to attention.
- Commit `7518d91e` introduced a pre-convolution `q_in = k_in = v_in = x` snapshot for the MUDD
  experiment. MUDD was later rejected, but the ordering survived. The current default path captures
  Q/K/V before the convolution, then updates `x`; consequently the same block's attention does not
  see the convolution even though current source comments say it feeds Q/K/V.
- The original pre-attention GPU arm improved fixed-step BPB from `1.174188` to `1.171962`.
- Later Trn2 work made the zero-initialized convolution trainable and improved the full result from
  `1.040879` / 2037 steps to `1.038386` / 2100 steps even with the surviving pre-convolution
  snapshot. This proves the component is useful through its remaining MLP/later-layer path; it does
  not measure the missing same-block attention path.
- The frozen dense6x1024 candidate contains the same ordering, so a control-stack KEEP could later
  be ported as a separate matched dense-candidate experiment.

Primary-source support: Canon layers place a lightweight trainable causal convolution after
normalization and before attention (Canon-A), and report cumulative benefits from horizontal token
mixing. The paper also warns that real-world pretraining comparisons are noisy, so it supports a
hypothesis rather than proving a BPB gain here:
<https://arxiv.org/abs/2512.17351>.

Bounded hypothesis: move only the Q/K/V capture relative to the already-computed, already-trained
zero-init local-convolution residual so default attention consumes the updated `x`. Add no operator,
parameter, data change, optimizer change, or precision change. Step 0 should be exact because the
convolution is zero-initialized. The verifier must explicitly check default, parallel-block,
QK-shift, checkpoint, causality, and evaluation semantics. Do not implement or launch from this
research note alone.

### 2. Evaluate modern Muon improvements as isolated feasibility studies

The public Modded-NanoGPT optimization benchmark's current record reports 2690 steps versus 3250
for its tuned Muon baseline, but that result is a stack and the maintainers explicitly say several
active details are not independently proven. Its active mechanisms include SOAP-Muon, a per-output-
row update floor, radial control, PowerCool, EMA-Nesterov, and a tail-EMA readout:
<https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_3_optimization>.

The relevant queue is:

1. **RowUpdateFloor/radial-control feasibility.** It may be cheaper than covariance
   preconditioning, but needs an operation/launch-count design inside the existing compiled Muon
   path before a training proposal. The public record added RowFloor and cautious weight decay
   together, so its individual benefit is not established.
2. **SOAP-Muon feasibility.** Map the current matrix groups, required row/column covariance state,
   distributed ownership, numerical precision, and complete-step overhead. No run unless a
   microbenchmark predicts a fixed-budget benefit.
3. **Newton-Muon feasibility.** It right-preconditions gradients with an inverse activation second
   moment. The paper reports 6% fewer iterations and about 4% less wall time to the Muon target in
   its Modded-NanoGPT reproduction despite 1.8% per-step overhead; moderate refresh intervals
   16–64 worked better than refreshing every step: <https://arxiv.org/abs/2604.01472>. First work is
   an offline shape/state/memory plan and a bounded optimizer microbenchmark, not a 30-minute run.

Do not duplicate existing negatives: this repo already measured its current NorMuon-style row
second-moment beta regime, Demon beta1, and Adam-group cautious weight decay as neutral or negative.
The existing full-weight EMA path was 6.3% slower in smoke and its EMA weights were worse than raw;
Tail-EMA is closed unless a new sparse-update design removes that measured cost.

### 3. Complete the saved ROW-FP8 systems lane

The saved ROW-FP8 forward component remains the best systems moonshot: roughly 9.4% faster in the
one-rank component measurement and 19.1% under four-rank contention. It still has no backward,
complete-step, or BPB proof. Trainium2's Tensor Engine can execute E4/E5 FP8 matmuls at double the
BF16/FP16 throughput under its layout/tiling constraints:
<https://awsdocs-neuron.readthedocs-hosted.com/en/v2.30.0/nki/guides/architecture/trainium2_arch.html>.

Next work is backward completion followed by numerical, cache/checkpoint, causality, and whole-step
gates. Require at least a repeatable 3% complete-step improvement before spending a 30-minute
quality run. Generic whole-model FP8 remains closed.

### 4. Revisit training-time input corruption only after a repeated-data regime exists

Recent controlled work found random token replacement with clean targets to be the strongest
individual augmentation in a many-epoch, data-constrained setup, and found that augmentation
interactions can help or hurt: <https://arxiv.org/abs/2606.16246>.

The current control finishes near but before the documented epoch boundary, so this evidence does
not transfer directly. Do not spend hardware on it now. If a faster accepted systems stack produces
substantial repeated-data training, a legal train.py-only, late/repeat-only, low-rate corruption arm
could be proposed after the verifier confirms the supplied data and clean evaluation are unchanged.

## Parked directions

- Generic full-model FP8/autocast, generic RMSNorm quantization, and generic NKI local convolution:
  already measured or superseded by more specific boundaries.
- MTP at this small scale, MIR's second forward pass, broad Mamba/linear-attention swaps, Hyena FFT,
  and tokenizer/byte-model changes: poor fixed-budget fit, locally closed, unsupported on Neuron, or
  incompatible with the frozen tokenizer/competition contract.
- Reopening weight decay, ordinary warmdown, epoch shuffle, generic EMA, QK shift, U-Net skips,
  attention temperature, Engram, and n-gram double hashing without new source-specific evidence.
- Any external dataset, pretrained weight, private validation signal, evaluator/prepare.py change,
  or leaderboard adaptation.

## What a very low leaderboard score plausibly combines

No public source found so far reveals another Trainium Frontier team's private recipe. The defensible
inference from public speedrun history and this repo's measurements is that a strong entry likely
combines: (1) a capacity/architecture shape that earns more BPB per token, (2) an optimizer and
schedule that improve learning per update, and (3) targeted numerical/kernel work that buys useful
additional updates without damaging quality. Throughput alone has repeatedly failed here, and a
single paper technique is unlikely to explain the gap. This inference must not be recorded as a
claim about any competitor.

## Selection order

After the active n-gram cooldown experiment closes, the Researcher should first reconcile and, if
still duplicate-clear, propose the same-block local-convolution wiring repair. Optimizer and FP8
lanes should begin with zero-hardware feasibility artifacts. Only independently verified KEEP
changes may later enter a declared stack-confirmation experiment, and any port to the frozen
dense6x1024 candidate needs its own matched control and submission-contract validation.
