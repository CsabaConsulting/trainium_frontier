# Round20 consistency audit — no MLP rebuild mismatch found in scope

Read-only local audit of exact archived500f,MLP3r4 andMLP5r2. **No incorrect hidden-size reconstruction, non-strict load, or hardcoded4096 in MLP forward/custom-autograd/evaluation was found.** Production-shape meta loader checks preserve4096/3072/5120; deliberately wrong saved config/state shapes refuse. Actual archived MLP training and eval branches match exactly on the bounded CPU test.

The arms do **not** show an eval-minus-training-proxy gap increase greater than0.005 versus500f: MLP3 change−0.0028799470;MLP5−0.0095887164. Absolute gaps all exceed0.005, including base, but the persisted Round20 alarm is a **gap increase**, not an absolute-gap test. This sparse proxy is not an apples-to-apples generalization measurement, and either threshold alone would be suspicion rather than proof.

Round20 supersedes the historical Round19 branch-C selection: **MLP4.5 HOLD**, no further capacity launch is authorized by this audit. No model/evaluator fixes, saved-checkpoint re-evaluation, host actions, state or ledger changes were performed. Root retains disposition authority; these findings provide no basis for a corrective checkpoint re-evaluation under the concrete-loader-bug exception.

## Last50 optimizer updates: raw mean unavailable

`train/steps.jsonl` contains step IDs only. Source logs a debiased EMA every10steps, not raw losses; all three last50-update windows contain **five** printed EMA samples. Their arithmetic means below are means of five rounded logged EMA values—not50raw losses. Exact last50 raw-loss means remain UNKNOWN.

| Run | Hidden | Updates | Last50 zero-based window | Mean of5 printed EMAs | Train proxyBPB | EvalBPB | Eval−proxy | Gap change vs500f |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| 500f | 4096 | 2685 | 2635–2684 | 2.64046 | 0.9571814762 | 0.9778364060 | 0.0206549298 | 0.0000000000 |
| MLP3 | 3072 | 2885 | 2835–2884 | 2.68788 | 0.9743714906 | 0.9921464735 | 0.0177749828 | -0.0028799470 |
| MLP5 | 5120 | 2364 | 2314–2363 | 2.70400 | 0.9802150805 | 0.9912812940 | 0.0110662134 | -0.0095887164 |

| Run | Actual printed samples: step / EMA |
|---|---|
| 500f | 2640: 2.6396; 2650: 2.6320; 2660: 2.6737; 2670: 2.6163; 2680: 2.6407 |
| MLP3 | 2840: 2.6838; 2850: 2.7038; 2860: 2.6771; 2870: 2.7142; 2880: 2.6605 |
| MLP5 | 2320: 2.7110; 2330: 2.7435; 2340: 2.6965; 2350: 2.6785; 2360: 2.6905 |

For **each** archived evaluation: num_tokens=20,971,520;num_bytes=83,462,186. Therefore actual eval bytes/token=3.9797871589660643. The requested diagnostic conversion is:

`proxy = mean_of_5_printed_EMAs / ln(2) / (83,462,186 / 20,971,520)`

`eval_bpb = total_nats / ln(2) / 83,462,186`

All three actual eval BPBs reproduce from their recorded nats and byte counts to <1e−14. The first formula borrows the **evaluation** byte distribution to rescale a training objective; it does not measure actual training BPB. Rounding error in printed EMA is ±0.00005 nats per sample, but the larger limitation is sampling/objective/data mismatch, not that rounding bound.

## Exact archived source lines

Line numbers below refer to each run's archived `package/train.py`, not an edited candidate. All top-level function/class bodies are identical across the three archives; reversing only global `MLP_RATIO` at line37 reproduces500f byte-for-byte. Full quoted functions and hashes are in the accompanying JSON.

### Configuration and strict checkpoint reconstruction

- Line37 is respectively `MLP_RATIO = 4`, `MLP_RATIO = 3`, `MLP_RATIO = 5`.
- Line155: `mlp_ratio: int = 4` is a dataclass fallback, **not** an eval override.
- Line983 passes `mlp_ratio=MLP_RATIO` into `GPTConfig(...)` during training.
- Line1313 saves `'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'model_config': asdict(model.config)`.
- Lines1323–1329:

```python
payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
config = GPTConfig(**payload['model_config'])
if config.sequence_len < EVAL_MIN_SEQ_LEN:
    config.sequence_len = EVAL_MIN_SEQ_LEN
model = GPT(config)
model.init_weights()
model.load_state_dict(payload['model_state'], strict=True)
```

- Lines1330–1332 call `model.to(device)`, `model.eval()`, `model.requires_grad_(False)`.
- The only loader config adjustment is sequence capacity to8192 (line1315/1325–1326), not hidden size or ratio. Actual evaluation remains context1024, as each archived REHEARSAL_COMMAND shows. Rotary tables are nonpersistent buffers (lines830–832), rebuilt from config; there is no saved MLP weight truncation.
- `init_weights()` runs before strict loading, so checkpoint state replaces initialized parameters. The bounded wrong-ratio check fails with size mismatch rather than silently using a4096 model.

### MLP train/eval and custom autograd

Lines767–770:

```python
hidden = config.mlp_ratio * config.n_embd
fc_out = hidden
self.c_fc = None if coalesced in ('all', 'vfc') else Linear(config.n_embd, fc_out, bias=False)
self.c_proj = Linear(hidden, config.n_embd, bias=False)
```

Lines775–781 select the ordinary path in eval (`if not self.training ...`), then `self.c_proj(F.relu(h).square())`; training's eligible BF16 token shapes call `_FusedMLP.apply(..., up, down)`. Both derive feature widths from actual weights. Lines730–732 define the common reference forward:

```python
relu = F.relu(F.linear(x, up.to(x.dtype)))
return (F.linear(relu.square(), down.to(x.dtype)), relu)
```

Lines747–750 call that same reference through `_fused_mlp_kernel` and save `x,up,down,relu`. Lines754–760 backward uses only weight/tensor dimensions:

```python
activated_t = relu.square()
dhidden_t = down.to(x.dtype).T @ grad.T
dpre_t = dhidden_t * (relu * 2.0)
dx = (up.to(x.dtype).T @ dpre_t).T
dup = (dpre_t @ x).to(up.dtype)
ddown = (grad.T @ activated_t.T).to(down.dtype)
```

No4096 or `4*n_embd` occurs in these forward/backward/eval paths. Dormant coalesced constructors at791/793 also use `config.mlp_ratio * config.n_embd`, and coalescing is fixed `'none'` at787.

The **sole literal4096 occurrence** is line1612 in `make_plan`:

```python
muon_ms = {(1024, 1024): 12.54413 / 24, (1024, 4096): 9.76316 / 6, (4096, 1024): 8.98251 / 6, (8, 32): 0.23355 / 3}
```

Line1625 uses `muon_ms.get(tuple(p.shape), 0.01)`; this is an estimated optimizer ownership/load-balancing cost, not a dimension assignment, slice, reshape or evaluation path. Other shapes fall back to0.01. That is a performance-model limitation, not evidence of incorrect saved-model geometry.

### Weight cache, logits and loss semantics

- Linear initialization line648 sets `self._cached_weight = None`; lines650–659 cache/cast the actual weight and call `F.linear`. The loader produces fresh uncached Linear modules. Training refreshes caches before its first FB at1504–1505 and after each optimizer update at1531–1532. There is no stale serialized cast cache in `state_dict`.
- Neuron compute dtype is BF16 in both training and loader (lines141–146). CPU/meta checks are not evidence about native compiler rounding.
- GPT.forward lines959–964 returns softcapped logits when targets are absent; when training with targets it uses MTP CE. Eval calls logits only, not `model(x,y)`.
- Lines1512–1513 assign `last_loss = loss.detach()` on each microstep. Line1537 reads that final microstep;1557–1562 apply0.9 EMA/debiasing and print every10steps. It is not an average over ranks or all accumulated microsteps.
- Lines615–617 set auxiliary MTP weight to `0.5 * max(0.0, 1.0 - max(0.0, progress) / (2.0 / 3.0))`; line634 normalizes main+aux losses by `weights.sum()`. Training earlier in the run differs from main-token eval objective.
- **Tail nuance:** all five samples per run have18–57seconds remaining of1800; the schedule is beyond2/3 and auxiliary weight is0 at those observations. Do not attribute the final-window gap to an active auxiliary term. EMA/history, data distribution and byte normalization still prevent a raw train/eval equivalence claim; earlier matched-token comparisons additionally have different LR/MTP progress.

## Trusted evaluator identity and byte accounting

All three archived source manifests pin the same `prepare.py` SHA5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb and rehearsal SHA7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b. The local protected evaluator's bytes were checked against this exact recorded hash; neither file was changed.

In that evaluator: line624 `logits = model(x)`;628 computes trusted `F.cross_entropy(..., reduction="none")`;631–635 use valid targets and token byte lengths, mask zero-byte tokens from summed nats, and sum actual bytes. Line650 is `val_bpb = nats / (math.log(2) * nbytes)`. Counts are equal across these runs, so no observed evaluator token/byte denominator drift explains their BPB differences. This comparison does not prove training tokens have the same byte distribution.

## Local tests and limits

Read-only script `20260927-round20-audit.py` SHAb796771356b9042b528a04abf38b493e6d709914ee4c2550e4be57789fa3c7dc; one shared-lock process, one thread,2GiB address-space/180CPU-second limit; peak RSS411752KiB. Checks:

- 40 member hashes/sizes per archive; exact source/manifest and only-ratio differences.
- Production-config meta GPT plus actual archived load_for_eval on synthetic config/state: correct hidden geometry, strict wrong-shape refusal, eval/frozen/uncached state.
- Real production-feature MLP at256BF16tokens for each hidden size: training custom-autograd forward versus eval path max absolute difference0.
- Actual saved checkpoint **tensor payloads were not downloaded/read/re-evaluated**. Synthetic meta state tests do not replace native evaluation or establish whole-model native numerical parity. Archived successful strict eval and stable checkpoint identities are supporting evidence, not a claim that this CPU audit inspected those tensors.

## Exact inputs

- **500f**: `research/v2/operator/20260926-operator-recovery/manual-jobs/trn22-r13-gate-adamw-full-r1/results-bundle.json`
  - Archive SHA `9931d177cb25b912a18406dae29d20d79a45c2a402be06ef46417de0d279d57c`.
  - Archived `package/train.py` SHA `500f80fec9eac2a42d6f0eae646b0e91643d46c4524ee3722e332434751fd368`; ratio4,hidden4096.
  - Eval nats 56,569,380; BPB 0.9778364059642928; all40 archive member hashes/sizes checked.
- **MLP3**: `research/v2/operator/20260926-operator-recovery/manual-jobs/trn22-r13-mlp-ratio3-gatefixed-full-r4/results-bundle.json`
  - Archive SHA `c3d5ce59b4c11bb021172d9b11674b80337766df6ca64534d8f5a30113459014`.
  - Archived `package/train.py` SHA `64eb5e17f897db38ba823fd47ae25c6f0f5eb0ec4d3b4e09625e5afa6da7b60c`; ratio3,hidden3072.
  - Eval nats 57,397,240; BPB 0.9921464734785841; all40 archive member hashes/sizes checked.
- **MLP5**: `research/v2/operator/20260926-operator-recovery/manual-jobs/trn21-capacity-mlp5-gatefixed-full-r2/results-bundle.json`
  - Archive SHA `e8b7c5ea8cbe78e6b22315ecea3def4f77be3df4bfc190fed5a80365aaa92414`.
  - Archived `package/train.py` SHA `17239e61fb21f49bf22a62953bce6a9a616c1a6c1228853166cf812e3055f865`; ratio5,hidden5120.
  - Eval nats 57,347,188; BPB 0.9912812939805707; all40 archive member hashes/sizes checked.

[Round20 directive](../../../workflow/ROUND20_2026-09-27_2310.md) SHAb080f3a6dca43038bcd24af5294f507589bc95a3ecf29053fd70c858bace1497; report was prepared within its bounded CPU-only audit. Companion `20260927-round20-consistency.json` records exact samples, arithmetic, source quotes, hashes and test scope. No source/evaluator/host/state changes or inferred corrective authority.
