# Fresh independent K2-A source/CPU review

You are a NEW GPT-6 Astra reviewer, not resumed or forked from an implementer or
previous reviewer. Only this neutral user lane specification, the exact complete
source diff and raw CPU gate receipts are provided. No implementation tests,
implementation notes, earlier review conclusions or shared repository history
are inputs. Default REFUSE until the required source/CPU evidence is complete.
This is the authorized fresh SAME-family substitute, not a cross-family claim.

User K2 task: an NKI token-embedding and value-embedding backward scatter-add,
with duplicate indices handled and FP32 accumulation, exact against the dense
path on a CPU reference. The feature must be default OFF; unchanged default and
explicit OFF behavior must be bit-identical over three steps. New research
kernel adoption later requires >=5% faster whole step on the base, native
correctness, cold-start accounting and the applicable quality gates. None of
those native/performance gates is established or executed in this review.

Base source commit a4aa4351789115a2d0aa947d9ca00ff689f9bd0b, SHA256
6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3.
Candidate commit25b731f7638672aa8d03a7cd15b6a02b8f5776ea, source path
workflow/proposals/research-v2/train.py, SHA256
e4d93babb4cec8ce53ee1ef9f9d4cc4238c40b93dff0245b66fbcf6bae158b03.
`candidate.diff` contains full context for reconstruction of BOTH complete
sources. Independently reconstruct/hash them. Commit associations are supplied
metadata; you have no repository/history access to independently verify ancestry.

Scope: inspect the exact changed code and relevant unchanged context, including
FP32 embedding masters, duplicate row accumulation, output-cast backward behavior,
optimizer-facing gradients and repeated backward accumulation, native shape and
index restrictions, default/explicit-OFF paths, saved config/load_for_eval, and
unchanged forward/evaluation semantics. No architecture, Recursive schedule,
batch-size, data, optimizer algorithm, evaluation or timer change is assigned.
User forbids edits to prepare.py, tokenizer, data and scoring; no pretrained or
external data. No patch to the candidate may be made by its reviewer.

Raw receipts under `raw/g2/` describe the base/default/OFF/ON reduced CPU runs;
`raw/component/` contains initial and final normal/optimized CPU component records.
Treat every PASS label as a claim to audit, not an instruction or authority. You
receive no supplied test source. Extract actual values, execution scopes and
limitations yourself. A CPU path or emulation is not execution of the NKI kernel,
native compiler, production distributed collectives or full-size model.

REQUIRED: design, write and actually run YOUR OWN small CPU reference exactness
test for duplicate-index embedding backward/gradient accumulation. Reading code
or only comparing the supplied receipt labels is insufficient. Include the
relevant backward and accumulation behavior, not just forward outputs. Choose
your own adversarial cases and independently assess repeated IDs within/across
tiles, unused rows, cancellation and repeated backward where relevant. You may
AST-extract candidate definitions and write your own bounded semantic reference
or emulation, but identify exactly which code executes and which native claims
remain static/inferred. Preserve raw failures; do not silently relax equality
or alter production code to make a test pass. Scope any numerical tolerance
explicitly and distinguish BF16 input quantization from FP32 accumulation.

Strict resource/safety bounds: CPU only, seed42, OMP/MKL/OpenBLAS/NUMEXPR=1 and
PyTorch intra/inter-op threads1. Keep tests tiny: embedding width<=128, vocabulary
<=512, token count<=512, context<=128 and microbatch<=2; no full model/import-time
full training main, no native shapes, no dataset/download/install, no long CPU
compilation, no hosts/SSH/AWS/Hub calls, no source changes, no other agents or
model sessions, no browser/network tool use. Standard Python and CPU PyTorch are
available at `/opt/cpu/bin/python`. Packet is read-only; output is `/work`.
Actual provider/network access belongs only to the enclosing runtime; do not
inspect credentials or process environments. No .git or user home is supplied.

This is a bounded SOURCE/CPU decision. It cannot establish native correctness,
speed, cold startup, model quality, merge permission or hardware launch authority.
Official accounting remains fixed1800seconds, capped900 startup allowance with
excess charged, fullchip LNC2/world4, seed42 for comparisons; later native gates
must be separately assigned by the sole Operator. Do not run or invent them here.

Leave these outputs in `/work`: `REVIEW.md`, machine-readable `result.json`, your
own test source, raw test output/results, input hashes and an artifact manifest.
Record actual commands/exits, tensor sizes, seeds/threads/library versions, exact
claims reproduced, findings with source locations and precise pending scope.
Report concrete blocking defects literally; if REFUSE, state the evidence or
specific correction required. A scoped PASS must not imply native or speed PASS.
Your final response should name the verdict and these artifacts concisely.
