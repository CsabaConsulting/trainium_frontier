# L3: Recursive schedule package, 2026-09-25

Implemented on exact research-v2 c5c bytes, behind default-OFF
`--recursive-schedule`. Candidate is `workflow/proposals/research-v2/train.py`;
SHA-256 `f9b6a9cbb1a1760fb8db422ffd79ffb91e9280899c40aa86028d935acf355f67`.
Only that training file changes executable candidate behavior. The remaining
files here are development evidence and must not be deployed beside a submission.

## Scope and exact source

Source: Recursive, `first-steps-toward-automated-ai-research`, Apache-2.0,
commit `a962ec43e2e3d7c018e59a2ece623fe6e232fdfb`,
[optimized_from_karpathy.py](https://github.com/recursive-org/first-steps-toward-automated-ai-research/blob/a962ec43e2e3d7c018e59a2ece623fe6e232fdfb/nanochat_autoresearch/solutions/optimized_from_karpathy.py).
Line numbers below are actual raw-file lines, including blank lines; some web
renderings remove blank lines and therefore show smaller line numbers.

| Piece | Exact constants / behavior | Source lines |
| --- | --- | --- |
| Muon LR | warmdown ratio 0.95, final multiplier 0.05 | 881, 884, 991-998 |
| Adam LR | warmdown ratio 0.65, final multiplier 0.05 | 882, 884, 1126-1131 |
| Table LR | no warmdown; existing `no_lr_decay` takes precedence | 883; existing c5c group semantics retained |
| Muon momentum | warmup 0.85 to 0.95 over 300 steps; quadratic warmdown to 0.79 | 1001-1002, 1010-1020 |
| NorMuon beta2 | 0.95 before Muon warmdown, linear ramp to 0.97 | 1004-1005, 1023-1030 |

The momentum rule follows the source precisely: when time progress exceeds 5%,
the quadratic value overrides the step-based warmup, even if fewer than 300
steps have elapsed. All progress uses the existing charged elapsed/budget
convention, with endpoint clipping. Existing optional step warmup is preserved.
The flag fixes its own LR floor and warmdowns, rather than reading the older
shared warmdown / floor knobs. Without the flag those knobs behave unchanged.

The dynamic beta2 value uses the existing fifth device-scalar slot in compiled
Muon, avoiding a changing Python constant in the compiled math body. No new
optimizer state, forward graph, model shape, evaluator, loader, or load hook is
introduced. `RECURSIVE_SCHEDULE` is saved as toggle metadata. Flag-off parameter
and loss identity does not imply serialized checkpoint-file identity: checkpoint
metadata has this extra explicit false toggle.

No WD pulses, Adam beta1 decay, table beta2 ramp, embedding-LR increase or x0
special LR group is ported. In particular, the A2 flat RMSProp table LR remains
the c5c value: embedding LR 0.30 times sqrt(768/1024), about 0.2598076211.

## Expected effect and risks

Earlier matrix LR decay, later Adam LR decay and longer variance memory may
improve final learning efficiency without adding model compute. BPB benefit on
Trainium is unknown. The NorMuon initial beta2 changes from c5c's 0.90 to 0.95
as a deliberate part of this package. These coupled schedule changes can lose;
a full 1800-second same-host comparison is needed. The CPU checks do not prove
Neuron compilation or absence of device recompilation overhead. Source recipe
success on another hardware/model configuration is not a measured local gain.

## Local evidence and its limits

`schedule_checks.json`: PASS. Tests compare flag-OFF schedule outputs against
the unmodified c5c source at multiple steps/progress points. Flag-ON checks cover
0, .05, .35, .5, .955, 1 and clipped endpoints; execute the actual production
param-group assignment; verify flat-table precedence, batch scale and dynamic
Muon tensor dispatch.

`tiny-g2-on.json`, `tiny-g2-off.json`, `tiny-g2-c5c.json`: each completed 3 real
training steps on all 4 CPU/gloo ranks, process exit 0, no timeout, finite losses,
identical parameters across ranks, all four tables changed, and strict checkpoint
load with deterministic logits. Width 128, table multiplier 1, table dimension
16, micro-batch 1x128; six layers and A2 layer positions are retained. These are
tiny synthetic CPU G2 checks, not exact-size A2 G2 or Neuron evidence.

`flag_off_identity.json`: candidate flag-OFF vs exact c5c is bit-identical over
three steps for all losses and all 74 final parameter digests; ownership plans
match. The baseline bytes are independently pinned in the receipt.

The existing laptop Torch runtime is 2.6.0+cu124, CPU only for these tests.
It lacks PyArrow, so `test_support/pyarrow` rejects any attempted real-data read.
Its Gloo rejects `ReduceOp.AVG`; the disclosed test-only `sitecustomize.py`
replaces synchronous Gloo AVG with SUM followed by division by world size,
identically for control and candidate. Existing `--stub-prepare` and G2 CPU
shims use synthetic data and deterministic time. No real BPB or causality result
is asserted from the stub. No package was installed. OMP/MKL/OpenBLAS each used
one thread; observed available RAM stayed above 7 GiB and swap near 1 MiB.

Two initial local failures are retained: an unsupported `--total-batch-size`
test argument (before model execution), and Torch 2.6's Gloo AVG error. They are
test-environment failures, not model results. The successful rerun uses the
unchanged exact candidate bytes with only the disclosed offline test shims.

An independent Verifier must review this diff and evidence. Exact-size CPU G2
with the production A2 argv remains outstanding; it must run on a CPU machine
with sufficient RAM. The laptop was deliberately not given full-size tables.

## Reproduction

Use the existing interpreter
`/home/poojith-devan/trtllm-vs-vllm-benchmark/.venv-vllm/bin/python`.
Run `check_schedules.py` from this folder. For the three G2 runs set
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` and set `PYTHONPATH`
to this folder's `test_support`. Invoke unchanged
`workflow/tools/preflight_dryrun.py --stub-prepare --steps 3 --nproc 4
--micro-rows 1 --micro-tokens 128 --timeout 180 --keep --trace-rmsprop`.
Each receipt records its exact source, argv and appended CPU arguments.
Compare c5c/off receipts with that tool's `--compare` option.

## Proposed full S-A command after independent gates

The current conditional schedule policy supersedes the original unconditional
S-A/A2 choice: if A1 or A2 improves same-host public-20M by at least 0.003,
select that winning table profile after its cross-host replicate and A3.
Otherwise select the dense control profile before architecture and the flag
bank. `PROFILES.json` and `proposal-{dense,a1,a2}.json` bind exact argv, source
bytes, unique per-host labels and comparison requirements for all outcomes.
No table win or missing table-control BPB is presumed. These are proposal
bindings, not a second queue or launch authorization.

`train.patch` is a zero-context patch against the exact c5c source. Apply with
`git apply --unidiff-zero` in an isolated checkout after confirming c5c; the
candidate file already contains the patch. The training file SHA above is
unchanged by this packaging step.

Deploy the candidate bytes as `train.py`, adjacent to the untouched organizer
files. A distinct Operator selects a unique label and performs machine checks.

```text
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --n-embd 1024 --ngram-ve --ngram-ve-layers=1,-1 --ngram-ve-table-mult 64 --ngram-ve-dim 128 --ngram-ve-trigram --ngram-ve-opt rmsprop --ngram-ve-flat-lr --ngram-ve-bf16 --ngram-ve-clip-exclude --no-qk-shift --compile-sdpa-direct --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public --matrix-lr 0.015 --recursive-schedule
```

Seed 42 and max-train-seconds 1800 remain source defaults. Research uses the
warm cache. Follow with fresh-process public-20M/context-1024 eval. Compare the
whole S-A stack against that host's A0; isolate the schedule contribution only
against that host's A2 table result. No host or external provider was accessed
by this implementer.
