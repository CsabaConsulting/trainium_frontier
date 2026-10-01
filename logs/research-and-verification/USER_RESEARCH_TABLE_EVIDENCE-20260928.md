# R49 attachment: value-path table evidence cross-check

Read-only local review, 2026-09-28. No model import/execution, host action, source edit, new proposal, queue change or reopening authority. Attachment SHA256: `508671df16c244e527e63b564d0b81e9179715a044742b109fd5d430dd495ee4` (`/home/poojith-devan/.codex/attachments/675ccd6e-ad69-4f57-af16-61617d747e24/Pasted text.txt`).

## Bottom line

The attachment correctly describes a historically implemented value-path design, but its suspected **dead double-zero initialization** and **slow Embedding lookup** do not describe the scored A1 run. A1 already had zero tables, random nonzero projections, direct `weight[rows]`, neutral zero gates, Muon projections, BF16 dense table gradients, bias-corrected foreach RMSProp, flat table LR and separate clipping. It lost against its same-host control. Moving a single table to the current 8-layer stack and freezing at epoch 2 are real differences, not demonstrated fixes. BF16 table-gradient accumulation conflicts with the later explicit FP32 decision.

The named `trainium-research-context` is **our context repository**, not authenticated competitor provenance. The distinctive quoted code, benchmark numbers and historical gains are present in our local source/logs. No missing-source request is needed.

## Exact A1, not a reconstructed default configuration

Here **A1 source** means `runs/Trainium-table-compile-research-20260925/submissions/research_v2/train.py`, SHA256 `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586` (locally rehashed). **Cleanup source** means `research/v2/operator/20260928-r37-saved-checkpoint-eval/artifacts/package/train.py`, SHA256 `6497d163d930275dfbc0bb494fbb546456478e3751119dcc1baf918f5638f9d3` (locally rehashed). Line numbers below refer to these exact files.

A1 label: `fs-20260925-v2-c5c-a1-m64d128-trn21-r1-full`. Its archived `research/v2/operator/20260925-trn21-a1-full-final/training-launch.json:2` specifies four ranks, width1024, last-layer bigram **and trigram**, table-mult64, dim128, RMSProp, flat LR, BF16, clip exclusion, NKI local conv, BF16 norm output, P32/T1024 and matrix LR.015. Seed42/1800s/LNC2 are in the same launch receipt. The log confirms depth6 at `training.log:52`, dtype/state at `:66`, and the complete two-table configuration at `:405`; these are not guesses from current defaults.

| Question | Exact A1 evidence | Difference in attachment |
|---|---|---|
| Initialization | A1:1919 constructs zero tables; :2004 zeroes n-gram gates; :2011–2014 zeroes tables and **uniform-random initializes projections**. | No double-zero defect. Zero gate gives `2*sigmoid(0)=1`; table has an immediate gradient path through the nonzero projection. Actual checkpoint table RMS was not inspected; no claim about its measured step50 magnitude. |
| Lookup spelling | A1:251 enables `NGRAM_VE_INDEX_GATHER`; :2109–2110 reads `tbl = module.weight; tbl[rows]`. `nn.Embedding` is the parameter holder, **not the executed lookup operation**. | Replacing the holder with `nn.Parameter` does not newly remove `embedding_dense_backward`; A1 already indexed its Parameter directly. |
| Capacity/hash/site | A1:1914–1929 and launch flags give **two 524288×128 tables**, 134217728 table parameters total, projected128→1024. Last layer means **layer5** in its six-layer model. A1:2058–2082 uses int32 previous/current (and previous2 for trigram), pad8192, layer-specific primes and prime modulus524287. | One 524288×128 bigram table halves table capacity/optimizer traffic; layer7 belongs to the current eight-layer base. Specific layer7 primes are not A1 layer5's pair, though the hash family is the same. |
| Injection | A1:1706–1715 adds projected bigram/trigram to attention **V**, using independent `2*sigmoid` gates over channels32:64 and64:96. | Value-path placement and the bigram gate formula are already tested in A1; not a correction from residual injection in that run. |
| Optimizer | A1:2163–2167 routes attention matrices (including n-gram gates) and projections to Muon. :2193–2195 gives tables β1=0, β2=.999, eps1e-10, WD0, embeddingLR×width scaling. :2806–2855 implements bias-corrected eager foreach RMSProp; :4594–4599 keeps these tables replicated and updates them on every rank after reduction. | Table optimizer/projection treatment mostly already tested. Current cleanup:896–909 explicitly routes gate matrices to AdamW, unlike A1's attention-gate Muon treatment; this distinction should not be hidden in any comparison. |
| Precision/reduction/clipping | A1 log:66 reports BF16 **parameter/gradient and RMS state/math**; FP32-state option was OFF. A1:2766–2775 allocates state in parameter dtype. :3069–3096 reduces then clips main and n-gram groups separately, maxnorm1. | “Excluded from global clip” did **not** mean un-clipped. Attachment's BF16 gradient recommendation reproduces historical A1 precision, not the currently approved FP32 accumulation contract. |
| LR/epoch2 | A1:3980–3987 uses warmup-only flat table LR; warmup0 and batch-rampOFF in this run. Freeze support exists (:229 default0; :3953–3958), but launch does not enable it. Log:317 onward reaches epoch2; log:412–413 reports3051steps/399.9M tokens. | Epoch2 freeze was **not tested in original A1**. LR0 at epoch2 differs operationally from its dormant `p.grad=None` freeze implementation (which also skips table all-reduce/update). Neither is proven to repair A1's quality loss. |

Cleanup6497 has **no hashed n-gram table**. It is depth8/width1024 with attention layers0,2,4,6,7 (:46–83), one ordinary unigram value embedding at layer7 (:825), standard `nn.Embedding(idx)` lookup (:937), neutral VE gate (:716–730, :854), and nonzero random VE initialization (:856). Its token/VE parameters and AdamW moments use ordinary FP32 storage (:907–911, :1123–1124); it does not have A1's table RMSProp state, table clip group or epoch2 freeze. Adding the proposed feature is not just repairing a current table implementation.

### Authenticated A1 result

Fresh public20M/context1024 result **0.9986312057378084** versus same-host c5c/LR.015 control **0.9914739939934344**, delta **+0.007157211744374**; 3051 versus3675 steps. This is a valid negative fixed-budget research result, **not** a cold-start proof and **not** a compiler failure. References: `research/v2/operator/20260925-trn21-a1-full-final/OPERATOR_RESULT.md:1`; `fresh-public20m.log:17`; independent result-use review `research/v2/analysis/20260925-a1-negative-a2-invalid-quality.md:5`; ledger row5.

Locally recomputed hashes match archived retrieval/ledger pins: launch `c0455ab1cc1611b2b648ce28dd435a07bd03603d0fa1e3bf52529a7c7e2a5866`; training log `b80583b9ae274e67a30e66d3bf69a59d73ab34956554512bcfa889253f7c20e5`; fresh-eval log `87f5077ed318bf1b4b722af24727707c676934de3e1d1ca16db0ee4935656657`. `retrieval-eval-and-postflight.json` records checkpoint `6f941a544887365962ab491ff372ac090b4981ad8e13c8febf8fff1654f9d24b`, 739322371 bytes, remotely rehashed then **not downloaded**. This review did not inspect checkpoint tensors or recontact the host.

## Local attribution and older gains

The attachment's source line numbers are not a pinned source revision. The relevant content is nevertheless located: A1:1668 gates, :2011 initialization, :2038 hash, :2085 lookup/benchmark, :2167 optimizer grouping, :2806 RMSProp. Current historical root `train.py` contains the same family at :2862–2899, :3029–3039, :3146–3175 and :3346–3351 (SHA256 `52ac9e6560a40592255e3499f4d42ac6e7073ea82da1b428e03dc984e6d4baa8`); it is **neither** current cleanup6497 **nor** exact A1.

The exact72.49ms/2.21ms numbers are in our `research/run-logs/embed-gather-bench.log:36` and `:43`, repeated in `research/trn2-experiment-results.md:2998` and A1:2096. They are **forward+backward**, not standalone backward times, for a historical isolated262144×128 benchmark with B4/T2048; not measured overhead on current B32/T1024. Raw log SHA256 `e937ac688cccf9dc0fb50fa428d9344333b21a7b5639b463bf30c6cc2cd29fc7`. The8192×640 comparison also exists in the same local benchmark. Its direct-index fix predates A1.

The older single-table gain is also our own local experiment, not competitor evidence. `research/run-logs/t31l-driver-2026-09-12.log:1–11` records exact launches; the model is six layers×640, training context2048. Raw logs clarify a misleading context shorthand in `research/trn2-experiment-results.md:3627`:

| Historical run | Main BPB at context2048 | Separate alternate BPB at context1024 | Steps |
|---|---:|---:|---:|
| `t31L-base-1800s` |1.034199|1.049507|2191|
| `t31L-l5bi-m64-1800s` |1.022205|1.036938|2202|
| `t31L-l5tri-m64-1800s` |1.018985|1.034050|2139|

Thus quoted single-table −.011994 is the **2048** comparison; its1024 delta is−.012569. Raw sources: respective `research/run-logs/<label>.log`, main/alternate scores at baseline:273/278, bigram:274/279, bi+tri:267/272. They are old in-run public evaluations, not the later fresh-process20M A1 protocol or a prediction for6497. Recomputed local log SHAs respectively `1f4359dec965713bbf0e3f491d980d0464cd48cd580df5618e73bc43862781fe`, `87da729ae63e1a01f413509ecc96e2c4f1f12326211ff025673418b35233ea94`, `bf604c61d01ea73ff189958052a91ff2f01b3350d20b59475d5e09c52dc4ed71`; driver `e600e266328f87f21c0fb10e47e486a033221e03a45280a67de33e1564a4e380`. These are local historical evidence pins, not a fresh remote authentication claim.

## R41 was not the executed R42/R46 failure

`workflow/ROUND42_2026-09-28.md:44` explicitly supersedes R41 before launch. `research/v2/proposals/20260928-bigram128-timing-probe/STATUS.md:1` records no released envelope and CPU/native NOT_RUN. Exact source532a971cb5a25f5e713cff4fa61f09869cbe6285e07b42362358592d8724a460 has zero FP32 table, nonzero projection (:860–894), projection AdamW (:928), and last-layer value-embedding addition after `F.embedding` of a BF16-cast table (:976–977). That R41 source is **not a measured value-path compiler failure**.

The actual native artifacts below contain exact source, training launch, log and terminal members; all four member-text hashes were recomputed and matched, as did each whole archive hash:

| Actual run | Science / executed timing source | Injection and observed outcome | Authenticated archive |
|---|---|---|---|
| R42 Stage1 trn21 |d5124600 / d93b0b01|FP32 direct lookup, per-block first256 residual channels; zero steps, NCC_IMGN901, terminal10:59:07IST|`research/v2/operator/20260928-round42-bigram256-after-r43r2-native60-trn21-preparation/candidate/artifacts/results-bundle.json`, SHA `7fb41cf4c16a33dfc0178854f6be3fc75d559d790609cb2c9bc2affcb4606a4c`|
| R42 one Embedding retry trn21 |02241534 / ae4c45ac|Same residual injection; zero steps, NCC_IMGN901, terminal11:16:09IST|`research/v2/operator/20260928-round42-embedding-retry-native60-trn21-preparation/candidate/artifacts/results-bundle.json`, SHA `7b07609bacb40d3bc2343cf0235a7f7f27ff3a2edc2f9004eff7bb372cbb54bc`|
| R46 Stage2 trn22 |377d7595 / a16d310f|Same residual site, custom duplicate-safe FP32 backward; zero steps, **NCC_IMCE902** `_reshape.1779`, terminal13:59:26IST|`research/v2/operator/20260928-round46-stage2-native60-trn22-preparation/candidate/artifacts/results-bundle.json`, SHA `26c77af3c2b17f45675d52ff1ed271b539bf97f725a60c07f41fb6651fcfee4e`|

Exact scientific lookup/injection lines: R42 `research/v2/proposals/20260928-round42-bigram256-stage1/train.py:947–955`; retry `research/v2/proposals/20260928-round42-bigram256-embedding-retry/train.py:947–955`; R46 `research/v2/proposals/20260928-round46-bigram256-stage2/train.py:990–998`. These are not last-layer V injections. The reviewed current-base records provide **no executed R41 value-path test**; do not convert R41's preparation into failure evidence or claim R42/R46 settled current-base value-path compatibility.

`research/v2/walls/round42-stage1-ncc-imgn901.md` maps the two R42 failed keys to backward graphs, but explicitly does not identify the offending operation. `research/v2/walls/round46-stage2-native-ncc-imce902.md` preserves the different Stage2 error. Their asynchronous loader traceback does not prove a loader defect. Nor do these records prove the attachment's proposed NKI conv/norm interaction: A1 itself ran with NKI local conv enabled (`training.log:48`), and a V injection still reaches residuals through attention output. Placement may affect compiler behavior; causation and a successful current8L rewrite remain unproved.

## Precision and inference boundary

`workflow/FINAL_SPRINT_GOAL_2026-09-26_1326_IST.md:24–33` explicitly requires **FP32 hashed-table gradient accumulation**, FP32 master/gradients/RMS state and BF16 *gathered* activations; it rejects casting the whole table before lookup. The attachment's BF16 Parameter/gradient all-reduce proposal conflicts. R41's BF16 lookup cast was expressly a timing-only diagnostic exception (`workflow/ROUND41_2026-09-28.md:11–14`), not permission to change the full-run reference.

The old epoch2 correlations do not establish why A1 lost; no randomized freeze comparison or checkpoint-RMS measurement was performed here. Projected5–15ms cost, −.003 to−.008 BPB, and ≥.005 jump are **unmeasured hypotheses**, not locally validated forecasts. A1's negative result does not mathematically rule out all future table designs, but it cannot be dismissed on the attachment's incorrect init/lookup suspicions. No new experiment is selected or authorized by this cross-check.
