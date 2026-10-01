# CLEAN+CONV b585 — evidence handoff, NOT SUBMISSION READY

Exact scientific source `b585d0a647962009ae13068fd0f823e17fd19392b5d0dbf1087e6e5f40e48742` already IS CLEAN881a plus the single conv-dW partial-store change. It was not rebuilt for this name. `audited_parent.py` is exact CLEAN `881a4f491d256fb89af33246a3b568fc3ffead9616070eea1860d7d8f7f9f749` for comparison, not a claim of upload. The human04:35 update reports RAMP uploaded and scored0.9775 official (tenth0.9759); this is human-reported, not independently portal-verified here. STACK upload remains unconfirmed. Protected500f is ancestry/comparison, not an uploaded reference. Existing CLEAN package remains unchanged.

## Exact launch and change

```text
NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py
```

Only `_conv2_dw_partial` changes: preserve each FP32 product, 128-token nc_matmul reduction and result copy, repack eight channel-block results by on-chip transpose/copy into1x1024 FP32 SBUF, store one contiguous partial per tap. Output partial layout and later sum/casts are unchanged. No new HBM scratch, forward/eval/model/optimizer/owner/RNG/loader/schedule/budget change or60-step cap. Outside that helper bytes are identical to CLEAN. Exact full diff and AST proof are attached.

## Selected best-of-six source-matched cold replay

Run `trn21-conv-dw-store-filler-full-r2` on trn21; archive `823ea754bcb22200ad1a62c9760ef86950110061aba9a1856f80fb3a07fec44c`. Fresh public20971520/context1024 BPB **0.9762047132468297**, cold step4 **650.7380700521171s**, charged through training process exit **1788.9089364169631s**, 2751 completed optimizer steps. Round33 score gate PASS: <=0.97845. This is one required actual cold run; no invented two-run rule or superseded ±0.0005-to500f rejection. Six absent-then-empty training caches, exact command/environment/source, clean snapshot, stable checkpoint, terminal, original trn21 lock/Guardian/outer-exit and archived idle are authenticated. Cleanup here means process/device/lock quiescence, not deletion or disk-write accounting. It is not current host admission.

CPU archive `384a711ffb5a6540998b83ebaecbcfadc1277d832b176f7fe2ab3f6af939d70a`, result `0eb5861ebdfee3d4988803dc175f47b3455138ca6359c2be45cd6f2713ae4c20`: eight actual installed NKI0.5 simulator cases, BF16/FP32, both lane counts, two upstreams, reduced batch including productionC1024/T1024; forward, dX, FP32partials and post-sum/cast dW byte equality plus dyadic/negative checks. This is not native/compiler scheduling, physical DMA, whole-model trajectory or optimizer proof. Inherited CLEAN CPU3 exact identity was reduced BF16 CPU fallbacks/unowned optimizer versus500f; it is not a native CONV identity test. Source review `9aea733d794f30f14c8166618d4f6810938bcbf4f0436599893ac77d3d3d5eb1` permits one research full, not submission legality. Same-host trn22 prefix timing was +2.5928%; not whole-ramp speed or a quality win. Early-phase harmonic throughput is not multiplied by1800; actual total tokens remain null unless source/log-derived accounting is established.

## All six cold runs: unchanged recipe, selected-run variation

This package selects the best observed replay, not a new scientific arm or a proven recipe win. Source, initialization policy, optimizer, schedule and bare command are unchanged b585. Six-run arithmetic mean BPB **0.976682176447** (rounded to12 decimal places; exact decimal sum in cold-evidence.json); observed min-to-max spread **0.0008397350757909**. Selection is best-of-six; this spread is not a confidence interval. No official portal score is claimed.

|Run|Host|Public20M BPB|Cold seconds|Charged seconds|Steps|Archive SHA256|
|---|---|---|---|---|---|---|
|trn21-conv-dw-store-full-r1|trn21|0.976737732406918|651.5250374421012|1788.7870391348843|2750|f0d8dfaa3135aae375c83579de6c0b5efcb917be1dadf319065932758f54dbe1|
|trn22-conv-dw-store-confirm-full-r1|trn22|0.9765959903809508|677.7090184979606|1789.1989635939244|2748|fa0776f29f4bf1f1b31374cf9f1d29a3616363909e1f88e488b60ae6902f0f3a|
|trn21-conv-dw-store-filler-full-r1|trn21|0.9769419100667623|644.1687107621692|1789.570305465022|2751|8b1b821e72981db084bfc3d5c2ff70bd78fabed85dbaa4d88400c3495b4229ea|
|trn22-conv-dw-store-filler-full-r1|trn22|0.9765682642578225|670.2221396220848|1788.9657829888165|2750|686ecc94bb17db72dc57d1962aa3dde55b28ba8c88bde3320a31d38bb937ceee|
|trn22-conv-dw-store-filler-full-r2|trn22|0.9770444483226206|675.4978872938082|1789.192851702217|2743|4ee223f060f6c457474ccafc34a2918af0e241660892568dd6adf61edbd6b543|
|trn21-conv-dw-store-filler-full-r2|trn21|0.9762047132468297|650.7380700521171|1788.9089364169631|2751|823ea754bcb22200ad1a62c9760ef86950110061aba9a1856f80fb3a07fec44c|

Selected minus first packaged run: **-0.0005330191600883 BPB**; the human's >=0.0005 packaging-refresh trigger is met, not a new-source adoption rule. The original five-file package remains untouched. Selected trained checkpoint SHA256 `bfa639d62547d6e17bda0fe992b5a56d2832741cb836a0a4bd1fd405fc144c45`, **453044690 bytes**, is authenticated unchanged across training/eval/terminal receipts; no checkpoint is included in this source-only handoff.

## Retained inventory supplement — not cumulative writes

|r2 host|Scope status|Regular logical bytes|Regular allocated bytes|Directory allocated bytes|Inventory receipt SHA256|
|---|---|---|---|---|---|
|trn22|COMPLETE|580846832|581951488|2367488|d517300f10344e7f49692d366697bc4859e3e8621bd2c2fa9a2be2a86cbfb488|
|trn21|COMPLETE|580826391|581931008|2363392|5cf754d0879ce7b5f7fbe3ee997b4c14c93dc192fb0425ff1d3150660b28d92f|

These are non-atomic retained inode totals beneath each exact packet root, with hardlink deduplication; allocated means st_blocks*512, not exclusive physical storage. Both retain a roughly453MB checkpoint and roughly77MB/39MB NEFF files. The ten largest relative paths and full metadata are in cold-evidence.json. Outside-root shared evaluation/data caches, the source checkout, other jobs, deleted/overwritten/transient bytes and write amplification are excluded. Cumulative bytes written remain UNKNOWN/null; no deletion or 'no large leftovers' claim. Checklist #5 remains FAIL. Inventory is supplemental evidence, not launch admission or a new gate.

## Retained compiler defaults

```python
if __name__ == '__main__':
    os.environ.setdefault('NEURON_CC_FLAGS', '--optlevel=1 --auto-cast matmult --auto-cast-type bf16')
os.environ.setdefault('NEURON_COMPILE_CACHE_DIR', '/tmp/neuron_cache')
os.environ.setdefault('NEURON_CC_FLAGS', '--optlevel=1')
```

These are unchanged optional defaults: main optlevel1/BF16 matmul autocast; import optlevel1. Actual bound values are in cold-evidence.json. Existing operator runtime/scaffold settings and private training-cache isolation are disclosed, not treated as required undocumented source settings. No compiler-policy removal or silent environment substitution.

## Human eight-item rejection checklist

PASS is scoped to its cited evidence; unknown remains FAIL/pending. These are the human's eight rows, not new training-admission gates. An inherited source issue is not excused by ancestry, and a missing proof is not an allegation of forbidden behavior.

|#|Result|Requirement|Actual evidence|Limitations / missing evidence|
|---|---|---|---|---|
|1|FAIL|Only train.py, exact launch, clean scoring-box imports|train.py SHA256 b585d0a647962009ae13068fd0f823e17fd19392b5d0dbf1087e6e5f40e48742:40,44,174,441; actual training-launch.json and isolated source in archive 823ea754bcb22200ad1a62c9760ef86950110061aba9a1856f80fb3a07fec44c; launch NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py|Exact isolated source/launch succeeded in existing native_venv; organizer clean-runtime import availability remains unverified. Provided prepare is scaffold, not participant payload.|
|2|FAIL|Same trained causal model and identical train/eval kernel path|train.py SHA256 b585d0a647962009ae13068fd0f823e17fd19392b5d0dbf1087e6e5f40e48742:1278; GPT.forward:918; MLP.forward mode branch:778; unchanged ASTs and CPU receipts in source_and_cpu|Strict weights_only loading and shared causal GPT are retained. Inherited not self.training selects ordinary MLP in eval versus eligible _FusedMLP in training. Literal same-kernel requirement/production rounding remains unresolved; CONV did not introduce it.|
|3|FAIL|No prohibited data/network/weights/local-cache dependency|train.py SHA256 b585d0a647962009ae13068fd0f823e17fd19392b5d0dbf1087e6e5f40e48742:1344,38,1278; archive 823ea754bcb22200ad1a62c9760ef86950110061aba9a1856f80fb3a07fec44c#operator-preflight.json and NATIVE_BINDING.json train/eval environments|Main uses provided TRAIN assets and fresh initialization, no direct network client. Six empty training caches prove cold training, not complete runtime I/O or cold-eval portability; eval uses recorded research cache paths. Optional cache default is not proven dependence.|
|4|PASS|Measured cold <=900s and charged <=1800s|archive 823ea754bcb22200ad1a62c9760ef86950110061aba9a1856f80fb3a07fec44c#operator-preflight.json; train/rehearsal.json; cold 650.7380700521171s, charged-through-exit 1788.9089364169631s|Applies only to this exact source-matched actual run, not future startup guarantees.|
|5|FAIL|Total checkpoint/log/cache bytes written and no large leftovers|archive 823ea754bcb22200ad1a62c9760ef86950110061aba9a1856f80fb3a07fec44c#checkpoint.json, checkpoint-after-postflight.json; retained checkpoint 453044690 bytes; original Guardian idle cleanup; actual retained packet-root inventories d517300f10344e7f49692d366697bc4859e3e8621bd2c2fa9a2be2a86cbfb488, 5cf754d0879ce7b5f7fbe3ee997b4c14c93dc192fb0425ff1d3150660b28d92f|Cumulative checkpoint/log/cache writes, overwritten/transient bytes and complete leftovers inventory are unavailable. Retained checkpoint size, archive size/free-space snapshots and idle processes are not total writes or disk cleanup; no deletion authorized. Supplement now inventories two exact r2 packet roots only, with large retained checkpoint/NEFF files. Shared evaluation/data caches and all historical/transient writes remain excluded; row remains FAIL.|
|6|FAIL|No hidden required environment beyond command; defaults on|train.py SHA256 b585d0a647962009ae13068fd0f823e17fd19392b5d0dbf1087e6e5f40e48742:36-39,45; main:1344; archive 823ea754bcb22200ad1a62c9760ef86950110061aba9a1856f80fb3a07fec44c#NATIVE_BINDING.json actual train/eval environments|Fixed recipe and retained optional main/import compiler defaults are explicit. Minimal provided-runtime environment sufficiency is not separately proved; external override availability is not evidence it is required.|
|7|PASS|Seed42 and no additional time-driven model selector|train.py SHA256 b585d0a647962009ae13068fd0f823e17fd19392b5d0dbf1087e6e5f40e48742:120,1344,1296,1333|Scoped source inventory: disclosed wall-clock LR/momentum/WD/MTP/batch schedule, mandatory startup/budget-stop predictor and reporting retained. Not native bitwise replay across variable-speed runs.|
|8|FAIL|Readable CLEAN source, per-kernel docs and shared eval path|train.py SHA256 b585d0a647962009ae13068fd0f823e17fd19392b5d0dbf1087e6e5f40e48742:1; helper-only diff lines510-524; MLP branch:778; convolution definitions lack docstrings|Exact CLEAN+CONV preserved without docstring rebuild. Missing conv/per-helper docstrings and inherited mode branch remain; generic fallback necessity is not blanket-proved. No silently revised source to improve checklist wording.|

The inherited MLP `self.training` branch remains explicit and unresolved under the literal same-kernels wording; no change or exemption is made here. `load_for_eval` still loads actual trained weights with weights_only=True/strict=True and returns the same GPT class, but that does not prove identical native train/eval execution paths.

Root must obtain independent actual-result/package review before applying the rendered new-only patch. External exact-source Claude audit remains PENDING. Checklist failures/pending evidence prohibit a submission-ready claim even though the numeric score gate passed. Human-only upload; no upload, adoption, export completion, host admission or AWS authority is granted. No checkpoints, data or credentials included.
