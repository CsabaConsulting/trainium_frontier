# Exact c239 bare-command ramp — Claude audit request

Audit attached train.py SHA c23935404ac44314e333ddf3608a6c671f4526263a3348eb901bb3f8d4ca05b0. This package is not approval or upload.
Only Poojith may upload after the required external ACCEPT for these exact bytes.
Source-audited BF16 comparison source: 54c835df60eb5d17ff49c6b79aa05aafd277027313fe6a07ed2955bdfaaebe3b. Prior acceptance does not
carry to this source. Human reports7077 uploaded, official score pending;
this package does not independently establish7077 external audit status.

## Source changes and semantics

Relative to BF16: norm-input cast removal, BF16 logits entering existing CE
with FP32 internal softcap arithmetic, runtime-weighted k1/k2 MTP supervision,
and a default-ON1->2microbatch accumulation ramp at one-third charged time.
MTP also defaults ON. Runtime gradient scaling follows actual accumulation.
Rankzero broadcasts the phase; native transition recompilation remains a
measurement concern. Native one-graph count is not established by CPU capture.
The port from2931 changes only --batch-ramp defaultFalse->True; its trn22
score0.9775917108277573 is context, NOT the c239 score or qualification.
targets=None/load_for_eval retain the same trained model: MTP and ramp add
no evaluation-only model. Inference is unchanged relative to7077, not BF16
(norm-input precision differs from BF16). Check these claims against source.

## Evidence-based precheck checklist (not external ACCEPT)

- Exact diff: DIFF_AND_MANIFEST.md is generated from the attached SHA-bound
  audited_parent.py and train.py, not a hand-maintained description. The diff
  contains norm/CE boundary changes, parameter-free MTP and ramp scheduling;
  prior parent acceptance does not certify these changes.
- Token-only inference: train.py:954 returns the
  same softcapped FP32 logits expression before the MTP branch. Inspect the
  inherited trained forward path, not only training-loss checks.
- Checkpoint/eval: load_for_eval at train.py:1312
  is AST-identical to BF16's loader. torch.load uses weights_only=True at
  line1318; state loading is strict at
  line1324. Saved config constructs GPT;
  trained weights replace initialization. Rotary-table sizing for evaluation
  is inherited. save_checkpoint is also AST-identical to the parent.
- Data/evaluator boundary: imports, prepare.py import at
  train.py:31, and all
  14 calls to imported prepare helpers are
  AST-identical to BF16. The complete inspected diff adds no external data,
  evaluator or tokenizer access. This is a delta claim, not a blanket waiver
  of inherited code; the shared prepare/evaluator is not modified or exported.
- Cold/charged evidence: exporter requires this source's authenticated six
  absent-then-empty cache roots, finite positive cold startup <=900 seconds,
  and positive charged-through-process-exit <=1800 seconds, with budget PASS.
  Cold startup here is the harness's measured fifth-boundary startup, not
  isolated compiler time. Measured values and receipt hashes appear below
  and in cold-evidence.json; no score or timing is inherited from2931.
- Identical command: byte-for-byte match to accepted BF16 command and archived
  launch-command.txt is checked before export. Full-chip/rank settings remain
  in that command; no research override is appended.

## Exact unchanged accepted launch

```text
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public
```

No added --batch-ramp or --mtp flags, timing override, source cleanup or edit.

## Own authenticated c239 measurement

Public20M/context1024: 0.978285555330, trn21 seed42.
Steps: 2687; cold startup 641.028370s;
strict cold-start limit900s: PASS (finite and <=900; not merely excess charged).
startup excess 0.000000s;
charged through process exit 1789.663936s,
budget1800 and over_budget=False. Fresh-process eval/checkpoint/source and
cold cache receipts are in cold-evidence.json. No early-phase projected
token count is represented as actual total training tokens.
Source/log-derived actual total tokens: 531759104;
accumulation1 steps: 1317; accumulation2 steps:
1370. See separate sampled phase rates in cold-evidence.
Last uploaded7077 research public20M: 0.981441493395;
7077 remains the last packaged source; the >=.001 gain threshold is unchanged.
c239 delta: -0.003155938065, required gain>=.001.
This is one c239 measurement, not a replicated mean or official portal score.

## Audit focus

Review cold budget/startup handling, full-chip launch, same checkpoint and
weights_only load, causal k2 labels/row boundaries, zero-weight MTP behavior,
BF16/FP32 gradients, rank-synchronized ramp and actual-accumulation mean loss.
Review inherited NKI/autograd code and unchanged data/evaluator/tokenizer
boundaries. CPU/source receipts are scoped evidence, not external compliance
approval; reviewers are not represented as fresh isolated sessions.
Exact c239 external audit is pending. No quota availability is assumed.
This export did not alter a host, job, queue, previous package or upload.
