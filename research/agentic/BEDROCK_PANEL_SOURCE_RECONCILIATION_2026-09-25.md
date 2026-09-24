# Bedrock panel source and evidence reconciliation — 2026-09-25

## Scope and disposition

This is a read-only Analyst reconciliation of the local Bedrock records matching
`research/agentic/runs/20260924T20*.json`.  It is based on control-plane commit
`33774742dc49de7a5d6f5fd539c1530430306763`, the exact archived `35c9` source,
the two archived Track-A control launches, the raw experiment ledger, and the
preserved head/loss component records named below.

No `train.py`, proposal selection, `CURRENT_STATE.md`, decision-engine state,
host state, or experiment was changed.  This report is not an authorization to
implement or run any idea.

**Bottom line:** the panel did not establish a new run-ready family.  Head
dimension 128 and Polar Express are already in the exact launched source; the
existing fused softcap-CE path does not eliminate full logits/gradient
materialization; local-convolution ordering remains a pre-existing researched
lane.  Only the last two topics survive, respectively as a feasibility study
and an existing proposal candidate, not as validated wins.

## Bedrock record boundary

The local `runs/` directory is ignored and the JSON records are not imported by
this report.  The following hashes identify the bytes actually audited:

| Local record | SHA-256 | Evidence disposition |
|---|---|---|
| `20260924T200029Z-researcher.json` | `1aaad4d840f1f8d1470c2de619dd501dec36ec267ed227f4be60f732c84ffc48` | Partial Fable portfolio, `max_tokens`; it explicitly lacked source, README, protocol, raw ledger, results and proposal template. |
| `20260924T200111Z-verifier.json` | `d7a9c51c845e3283b466c6cc11dbb1bb4ab0d7ded8586d0c1584c8050c583276` | Astra correctly returned evidence-blocked `NEEDS WORK`; no source or proposal was present in its packet. |
| `20260924T200123Z-analyst.json` | `a0e96b51c4ab520f0c696ab207b48c255d211aea1e1a792e42157e2846f7de2e` | Astra portfolio analysis; it explicitly treated reported metrics as document claims rather than raw-artifact verification. |
| `20260924T200150Z-researcher.json` | `5826ea79f6f64a1ac98da77fced6b7b15151c54597f973ad2488964611895daa` | Fable stopped at `max_tokens` and persisted an empty answer; no finding can be recovered from it. |
| `20260924T200258Z-researcher.json` | `ecd3ae8debef8dc3120418127975351afeb66431ed5f593de7d30077725e1bc1` | Astra returned three `NONE` slots because its packet lacked source, launch, raw ledger, results and proposal template. |
| `20260924T200554Z-researcher.json` | `5e660e339ed7c41a418cea758a817107b38de7bc1d618a4b4661cd31261bb1e9` | Current Fable bytes propose head-dim, Polar-Express, and head/loss materialization ideas; the first two rested on source assumptions now disproved. |
| `20260924T200813Z-researcher.json` | `83b33716e12f5a496da36eb7029593916712ca79919b82034eed9aa4b07d9e8c` | Astra source audit.  Its source-level findings are confirmed below; it correctly left source-hash and launch-flag binding unresolved because those artifacts were absent from its packet. |
| `20260924T200957.992499Z-analyst-183536.json` | `f2581bc630c682d3460bffde98698ca1f9ab343a04fe2863e56443270aec6099` | Astra reconciliation.  Its duplicate/feasibility classifications are confirmed, subject to the independent launch and ledger evidence below. |

None of those model packets contained a complete ledger.  Their novelty and
historical-performance statements therefore cannot substitute for the raw
search in this report.

### Concurrent-runner collision

The collection incident at `20260924T200554Z` is material provenance: an
original root-source Astra audit was overwritten by the current Fable record.
The overwritten bytes are unavailable and are not inferred here.  Commit
`33774742dc49de7a5d6f5fd539c1530430306763` fixes the runner prospectively by
using microsecond timestamps plus PID in filenames and exclusive (`"x"`)
creation.  The post-fix analyst filename demonstrates the new form.  The fix
prevents recurrence; it does not reconstruct the lost answer or make the other
panels ledger-complete.

## Exact source and launch binding

The audited source is:

`runs/Trainium-v2-track-a-a1-r2-launch/submissions/dense6x1024_public0992851/train.py`

Its independently computed SHA-256 is
`35c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2`.
Both archived Track-A control `result.json` files bind their source to that same
hash:

| Control evidence | Result SHA-256 | Training-log SHA-256 |
|---|---|---|
| `runs/Trainium-v2-track-a-35c9-control-trn21-operator/.../fs-20260924-v2-35c9-lr015-control-trn21-r1/runtime/out/fs-20260924-v2-35c9-lr015-control-trn21-r1/` | `4020d2a9a82ca4e8c05bb158ae0881da2464e3ecad55c9dfef6c8847973645dc` | `a3b21eb9c24bff94bfd27bd98c3cc89f32b58f48f0b48bd4981f37778071d21e` |
| `runs/Trainium-v2-track-a-35c9-control-trn22-operator/.../fs-20260924-v2-35c9-lr015-control-trn22-r1/runtime/out/fs-20260924-v2-35c9-lr015-control-trn22-r1/` | `c25ba29b2fbb39b4c43f4f46148cb1db4a486fdf58a34823ade36244fdd21efe` | `2c118358a1e260371bba03656d631b1a2032a1c7882dc785f05d0e7c9624b076` |

The complete artifact paths, with `result.json` or `training.log` appended, are:

- `runs/Trainium-v2-track-a-35c9-control-trn21-operator/research/submission-validation/20260924-v2-track-a-35c9-controls-a1/operator-evidence/fs-20260924-v2-35c9-lr015-control-trn21-r1/runtime/out/fs-20260924-v2-35c9-lr015-control-trn21-r1/`
- `runs/Trainium-v2-track-a-35c9-control-trn22-operator/research/submission-validation/20260924-v2-track-a-35c9-controls-a1/operator-evidence/fs-20260924-v2-35c9-lr015-control-trn22-r1/runtime/out/fs-20260924-v2-35c9-lr015-control-trn22-r1/`

The two controls have the same archived launch command:

```text
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015
```

There is no `--head-dim` or `--muon-ns-steps` override.  Source lines 117 and
169 set `HEAD_DIM = 128` and `MUON_NS_STEPS = 5`; the parser carries those
defaults at lines 3518 and 3531, and line 3574 assigns the parsed head dimension.
Both logs report `n_embd: 1024`, `n_head: 8`, and `vocab_size: 8192`.  Thus the
actual launch used 8 heads of dimension 128.  This closes the principal launch
gap left by the final two panels.

## Idea-by-idea adjudication

### 1. Head dimension 64 to 128: reject as duplicate/no-op

The `200554` memo assumed the exact source might use 16 heads of dimension 64.
It does not.  `build_config()` derives `n_head = n_embd // HEAD_DIM`, and the
source default plus archived launch and logs prove 1024/128 = 8.

The raw ledger also contains a related, but non-causal-for-width-1024, alias:
`research/experiment-log.jsonl` label `t02-d6-hd64-e640-900s` explicitly used
`--head-dim 64 --n-embd 640`, reaching 999 steps and BPB 1.099107.  The matched
same-commit `(6,640,128)` control recorded in
`research/trn2-experiment-results.md` reached 1,094 steps and BPB 1.081255, so
that old 640-wide arm was slower and worse by +0.017852 BPB.  That result is not
extrapolated to a different width; the exact-source no-op is sufficient to
reject the proposed 64-to-128 change.

### 2. Polar Express coefficients: reject as duplicate/no-op

Source line 2255 already defines five iteration-specific
`POLAR_EXPRESS_COEFFS` triples, not one repeated canonical triple.  The default
iteration count is five.  The compiled Muon body consumes those triples at
lines 2315-2321; the compiled dispatch clamps against the list at lines
2864-2875; and NKI/eager paths consume the same list at lines 2892-2910.
`research/trn2-experiment-results.md` also calls the existing optimizer body
the “real Polar-Express Newton-Schulz body” in its world-size arithmetic audit.

The raw-ledger `t04-nssteps-4-900s` arm changed iteration count, not the
coefficient family, and therefore is not a missing coefficient A/B.  No new
hardware result is required to decide the panel proposal: replacing fixed
coefficients with Polar Express describes the bytes already launched.

### 3. Local-convolution ordering: survives only as the existing lane

The exact block snapshots `q_in = k_in = v_in = x` at line 1773, applies the
trained causal-convolution residual to `x` at lines 1778-1779, and sends the
pre-convolution snapshots to the serial attention path.  The source audit is
therefore correct: changing same-block attention to consume the updated `x` is
a real source change.

It is not a fresh Bedrock discovery.  The history in
`research/agentic/PRIMARY_SOURCE_RESEARCH_MAP_2026-09-23.md` attributes the
snapshot to commit `7518d91e`, records an older fixed-step local-convolution
comparison (1.174188 to 1.171962), and records a later old-stack full-budget
comparison (1.040879/2,037 steps to 1.038386/2,100 steps) in which the snapshot
still existed.  Raw aliases are `sol9a-conv`, `t01-baseline-fullbudget`, and
`t01-kernel2-fullbudget` in `research/experiment-log.jsonl`.

Those runs differ in stack, shape, or causal factor and do **not** prove the
exact dense35c9 ordering change.  The `--nki-local-conv` smoke in
`DECISION_LOG.md` tested implementation, not ordering.  The same log already
records “Dense35c9 local-conv pre-attention ordering research” as `RESEARCHED
NEXT`, explicitly requiring a new proposal and normal gates.  It remains the
highest-priority existing learning-efficiency lane after A1/A2, with no new
performance claim from this panel.

### 4. LM-head/CE materialization: feasibility-only, not a fused-CE proposal

The exact source already has a fused softcap-cross-entropy autograd path:
`_SoftcapCrossEntropyNKI` saves logits `z`, targets and logsumexp at lines
783-800, and the adaptive NKI install replaces its two primitives at lines
4701-4727.  A proposal merely to “fuse CE” is duplicate.

Full materialization remains.  GPT forms
`self.lm_head(x)[..., :vocab].float()` at line 2224, then invokes the CE path at
line 2231.  The original NKI backward allocates full `[N,V]` FP32 `dz` at line
758; the adaptive backward also allocates full-shape `dz` at line 4660.  At the
archived launch's 32 local rows, sequence length 1,024, and vocabulary 8,192,
one FP32 `[32768,8192]` tensor is 1,073,741,824 bytes (1 GiB).  This corrects
the memo's unsupported GPT-2-like 50k-vocabulary/approximately-6.5-GB estimate.
It is not evidence that logits and `dz` peak simultaneously.

The closest preserved profile/implementation aliases all predate exact 35c9
and reject particular layouts, not the entire design space:

| Record | Scope and result | Boundary |
|---|---|---|
| `runs/Trainium-v2-track-a-a1-r2-launch/research/run-logs/fs-w101-fused-head-loss-forward-component-r1.summary.json` (SHA-256 `48e33f52b1e9b15ee5ef1593f38a0a51ccb44514bd64f4394057346e26da3508`) | Shape `[16384,1024,8192]`; two native forwards median 12.8198 ms versus fused-transpose 16.2542 ms, +26.789%. | Commit `9cdfca3`; forward-only, no backward, optimizer, or whole model. |
| `runs/Trainium-v2-track-a-a1-r2-launch/research/run-logs/fs-w112-head-loss-cached-whole-step-comparison-r1.json` (SHA-256 `9698fff962979aaff99e33813567f3f8479a073da1d0d7256b619387ef44fe45`) | Control 500.727 ms versus cached candidate 689.956 ms, +37.791%; numerical postchecks passed. | Commit `5abe670`; candidate postflight failed disk reserve; no quality promotion. |
| `runs/Trainium-v2-track-a-a1-r2-launch/research/run-logs/fs-w123-head-loss-split-whole-step-comparison-r1.json` (SHA-256 `3798342d48d51d82c495bc3efd99964802c5b50577251d3dd018291a73972537`) | Original 501.265 ms, split-native 684.670 ms, split-cached 681.026 ms; regressions +36.588% and +35.861%; only 0.532% gain within split scope. | Commit `2f2ad50`; advancement false, no full-budget quality run. |

These are useful negative aliases: transpose, split-region and cached variants
did not convert component promise into whole-step speed.  They do not prove a
chunked/recompute implementation correct, faster, or impossible on exact
35c9.  No current-stack liveness trace, current-stack whole-step attribution,
correctness contract, or implementation exists in the audited evidence.
Therefore “remove full LM-head/CE materialization” remains **feasibility-only**
and must not consume a full training run on the panel forecast.

## Surviving queue and cheapest next check

No Bedrock-proposed idea is ready for a new full run.

1. Preserve local-convolution pre-attention ordering as the existing
   post-A1/A2 lane.  The cheapest next check is a read-only authentication of
   the already prepared `20260923-local-conv-pre-attn-serial-only-r3` candidate,
   diff, and verifier receipt against exact 35c9 bytes.  The output should state
   whether it is a one-factor current-base patch or has drifted.  Only after a
   fresh human-approved proposal and normal verifier gates could it be run.
2. Preserve QKV-normalization CSE only at its existing queue position.  The
   Bedrock records add no source or profile evidence to that lane.
3. For head/loss materialization, the next cheapest check is zero-host: search
   preserved artifacts for a binary-bound exact-35c9 profile that separates
   LM-head matmul, CE forward, CE backward, allocation/liveness, and spill cost.
   The W101/W112/W123 artifacts are older-source aliases and must not be relabeled
   exact-35c9 evidence.  If no such preserved trace exists, record the gap; do
   not implement or launch from these panels alone.

## Claims explicitly not made

- The Bedrock panels did not have complete ledgers.
- The overwritten root-source Astra answer is not recoverable evidence.
- Historical head-dimension or local-convolution results are not exact-stack
  causal estimates.
- A 1-GiB tensor size is not an observed peak-memory saving or speedup.
- Component and old-source timing records are not a current full-step profile.
- No scientific win, proposal approval, training authorization, candidate
  selection, or submission approval follows from this reconciliation.
