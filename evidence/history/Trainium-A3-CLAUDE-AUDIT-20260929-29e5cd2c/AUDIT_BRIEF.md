# A3 NOPROJ-ID — evidence for Claude audit; NOT SUBMISSION READY

Actual ALL3 archive qualification and mean gate passed. This rendered audit material is not a Claude verdict, package-ready announcement, launch permission or upload approval.

## Source and command

Scored source: c5ef8414e2df998a9b52eeb2e249b66c0110a3c37869e676719e93975a3432b4.
Upload derivative: 29e5cd2cfd35bb52e5a955057a1661786239d0812d9a797c85dc0df7fe2d7446.
Only argparse freeze default286M->0 and two header sentences differ. Model,
optimizer, training, checkpoint and evaluation AST are otherwise identical.

Scored argv: `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0`
Proposed bare upload argv: `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py`

Independent stdlib parser/config/AST receipt813bb4bf proves the bare derivative
selects exactly the recorded scored configuration (including CONTINUE/every2).
Root reran it. This is not a new cold run on derivative bytes. R73 explicitly
requests the same default-only transformation accepted for D2 in R66.

Recipe versus D2: replace learned layer7 proj7(128->1024) with the fixed
repeated128 map (each head receives the same slice); retain existing learned
per-head gates. Its creation/initialization was inside torch.random.fork_rng,
so removal does not advance the outer RNG; common-parameter initialization
is separately CPU-checked. Add the authorized end-run rank-table diagnostic,
charged before checkpoint/exit. Table262144x256, embedding site, FP32
duplicate-safe sink, BF16 table/RMS state, every2, seed42 and CONTINUE remain.

## Required three-run evidence (fill only from actual archives)

| Run | Host | Public20M | Cold seconds | Charged-through-exit seconds | Status |
|---|---|---:|---:|---:|---|
| original | trn21 | 0.9719336457209543 | 676.520512084011 | 1792.6837460051756 | actual archive d38a9edd |
| repeat1 | trn21 | 0.9716331526259039 | 664.2176943281665 | 1793.5396360058803 | actual archive 5110ddf081bafcd248f76ebeb5e4c045ca48e6a602430aa641abbc0e1d62f752 |
| repeat2 | trn22 | 0.9715861357587537 | 687.1824835711159 | 1792.487295588944 | actual archive 9e933455e551979dd619f1e4f986c6579cb66258f82ef6d77b1fd55c07112b29 |

Mean ALL THREE: 0.9717176447018706. Delta versus verifier-recorded D2 mean 0.9725512952419127: -0.0008336505400421412.
All use seed42; this demonstrates same-seed repeatability, not across-seed
robustness. Do not discard a worse receipt or claim an expected official score.
For each run authenticate source/argv/config, original Guardian release,
six empty private training caches, raw fifth-boundary cold<=900, charged<=1800,
checkpoint identity before/after fresh20971520-token/context1024 evaluation,
and finite exact-zero table disagreement across all four ranks.

## Eight-item rejection checklist — research evidence versus pending compliance judgments

Line references below refer to upload29e5. FAIL means missing/contrary evidence,
not permission to omit the item. Claude makes submission-compliance decisions.

| # | Requirement | Current evidence status | Evidence/limitation |
|---|---|---|---|
| 1 | Only train.py; exact command; supplied dependencies | FAIL/pending clean-box review | Scored train/eval import success is in the actual archive. Imports at36–44 and bundled kernels unchanged from accepted D2. No installs/additional payload proposed. Bare argv equivalence independently proved, not separately cold-executed. |
| 2 | Trained model, strict load, shared causal forward | PASS research-source evidence; Claude review pending | 994–1049 model-owned lookup/core; 1502–1522 save full state and strict=True load. A3 reduced/full-table CPU and separate eight-head map reference, plus actual fresh20M. Preserve R71 Q2 resolution and original failed cross-shape BF16 prefix receipt; no submission tolerance waiver, full native FP32 parity or cross-shape bit identity claimed. |
| 3 | No validation/external/pretrained/network/cache dependency | FAIL/pending compliance review | Candidate obtains train split at1611, supplied prepare/tokenizer API44/1580s. Hash only uses tokens<=t. Six empty training caches measured. No clean-box no-network trace or universal environment-independence claim; source unchanged here from accepted D2 except map. |
| 4 | Cold<=900 and charged<=1800 on exact scored path | PASS exact-scored-path archive evidence; derivative cold replay not performed | Original raw676.520512/charged1792.683746, six empty caches. All three actual cold/budget/checkpoint/evaluation receipts are authenticated in ALL3_RESULT.json and the three complete archived run receipts. Declared evaluation caches may be warm; never called cold-eval evidence. |
| 5 | Disk/total bytes written and leftovers disclosed | FAIL/unknown cumulative I/O | Original checkpoint587788862B; all three actual checkpoint sizes and archived free-space observations are disclosed in cold-evidence.json; no new retained-file census performed. Checkpoint plus retrieved-text sizes are NOT total writes. Keep cumulative I/O unknown unless separately measured. No cleanup/protected-artifact deletion implied. |
| 6 | Default ON; required launch environment only | PASS bare-config proof; environment scope pending review | Parser1577 now0; main1587 passes it explicitly; independent checker matches all14 recorded config fields. Direct build_config default286M is unused by bare main and is preserved. Compiler/cache defaults36–39 are inherited; research wrappers supply declared extra cache environment. |
| 7 | Seed/time behavior disclosed | PASS research evidence | All three seed42; existing charged-time schedules and stop safety unchanged, record_step_boundary1691 retained. Same-byte/same-seed spread reported honestly. |
| 8 | Readability/CLEAN/kernel explanations | FAIL/pending Claude judgment | Two inaccurate header sentences corrected. Existing unreachable injection alternatives preserved to avoid unscored edits; no silent cleanup. Exact diff versus scored source and D2 supplied for review. |

## Correctness lineage and human boundary

Include full ACTUAL_CPU_RECEIPT, INDEPENDENT_REFERENCE, original conditional
INDEPENDENT_REVIEW_RECEIPT and effective R71 Q2 authority from the first scored
archive. Do not rewrite the original conditional verdict as unconditional.
D1 parity archive1e0f075a is historical lineage, not A3 production parity.
Structural derivative proof813bb4bf covers only three textual changes.

Required README framing: hashed bigram embedding table (modded-nanogpt /
nanochat / Engram lineage), zero-initialised, learned from the provided training
data inside the timed loop by RMSprop with gradient accumulation over two
steps, saved with the model, identical train/eval forward, a pure function of
token ids at positions<=t; harness untouched.

Poojith uploads only after Claude audit. D2 upload6d197b3c is the first morning
reviewer probe regardless of this A3 result. No upload by Codex.

## Scope carried forward

Cumulative writes remain UNKNOWN, clean-box/dependency/no-network compliance and readability judgments remain pending. Historical structural receipt pending strings are preserved verbatim, with actual ALL3 qualification supplied separately. No Claude ACCEPT is inferred. Research wrappers declare extra cold/evaluation cache environments; bare derivative configuration is proved, not separately cold-executed.
