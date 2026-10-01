# e1bd CE2 + FLATCACHE + ZBUNDLE — R151 fallback review request

NOT UPLOAD-READY. Package audit PENDING; Claude exact fallback-package verdict UNKNOWN. Only train.py is the proposed model payload. Historical source-specific ACCEPT is preserved only in its original source/CPU scope, never promoted to package/compliance approval. No upload, host call, numerical run, source change, prepare/evaluator edit or checkpoint copy occurred.

R151 reports LaneB submission sub-2ba0961e INFRA_FAILED: "INFRA_NEURON_CRASH: Neuron runtime error during scoring (exit=1)". This is a controller-reported official failure, not an organiser raw log in this package. Fault cause is UNKNOWN. The earlier LaneB SDPA-fallback claim must not be inherited: its Neuron helper raises if bundled attention handles are absent. The supplied scorer calls model(x), so the logits return precedes participant CE/CE2; prepare.py computes its own cross entropy. The fresh scorer-condition reproduction is pending. Source inspection does not prove runtime compatibility.

Exact source: e1bd539f49658a01bf3188b70e5c04eecdb85427cc9e55db96abf487fad5a310. Parent comparison: 0c2bd7cc926e51d248873e72ffb344ba7dc49a30219ed5ec23b01130f345944e. Fixed command (do not execute during review):

`NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0`

## Actual research FULL receipts

| Host / archive SHA256 | Public20M BPB | Raw cold s | Charged through exit s | Steps |
| --- | --- | --- | --- | --- |
| trn21 / 088637f6f4db8ce993f5cc52e80ccc95f5e3441aacb5cc06ce88e59af2be1a0f | 0.9672630039667548 | 678.5742937349714 | 1793.4091268959455 | 2849 |
| trn22 / 7f45233b93aed4b120afe7d2149a02b24efa31cb0c14221878ee811c9c68294e | 0.9669715685328759 | 672.9102984368801 | 1793.388613705989 | 2857 |

Receipt count 2; arithmetic mean 0.9671172862498154; spread 0.0002914354338788838. Both e1bd runs use seed42; a same-seed repeat is not independent-seed evidence. ZBUNDLE has only one included FULL, not two. All values are research public20M/context1024 scores, not official/private leaderboard scores. Source/command/packet/archive/Guardian/checkpoint/budget are reauthenticated using preserved pinned metadata helpers. No additional source/math gate is invented.

## Required eight-item checklist

Full human wording is in REJECTION_CHECKLIST.md, SHA256 71717878b1c070bb7bce459579ae31f8b26b6339db9c7661859db1f19b9b9612. Unknown is FAIL, not assumed PASS. The following are preparation findings requiring separate review; only item4 is a measured receipt claim.

## 1. Only train.py; exact launch; no extra files/install/missing clean-box imports

FAIL — pending evidence/audit. Exact source e1bd539f49658a01bf3188b70e5c04eecdb85427cc9e55db96abf487fad5a310, command and all raw FULL packet members authenticate. Source imports are listed in SOURCE_INVENTORY.json. Only train.py is proposed for submission; audit files/parent are not dependencies. Existing research venv is not a pristine organiser image. Clean-box availability and minimal-launch sufficiency remain unproved.

Evidence: train.py SHA256 e1bd539f49658a01bf3188b70e5c04eecdb85427cc9e55db96abf487fad5a310; cold-evidence.json:1 SHA256 a836f8852c3c64f07ea4e0c79182f3179928f7b9329ab10bff669163022c687d; SOURCE_INVENTORY.json:1 SHA256 b51022aa05f9c8a9d95a598aa7d26924f3185b48323e2b7d100e58e7e6b4a6a7. Raw member paths, complete hashes and byte lengths are in RAW_RECEIPT_INDEX.json.

## 2. Trained checkpoint; same train/eval forward/attention/MLP/kernels; pure causal targets=None

FAIL — pending evidence/audit. load_for_eval at train.py:1850 strictly reloads supplied model_state into GPT and uses the same forward; GPT.forward:1287, GPT.forward_core:1305, attention:995, MLP:1083. No .training attribute was found. targets=None:1341 returns softcapped logits before CE/CE2. prepare.py:624 calls model(x); its own CE is at 628. Training core compile vs eager eval entry and native shape/LNC/environment behavior are not proven equivalent by source syntax or CPU surrogate checks. Scorer-condition diagnostic remains pending.

Evidence: train.py SHA256 e1bd539f49658a01bf3188b70e5c04eecdb85427cc9e55db96abf487fad5a310; cold-evidence.json:1 SHA256 a836f8852c3c64f07ea4e0c79182f3179928f7b9329ab10bff669163022c687d; SOURCE_INVENTORY.json:1 SHA256 b51022aa05f9c8a9d95a598aa7d26924f3185b48323e2b7d100e58e7e6b4a6a7. Raw member paths, complete hashes and byte lengths are in RAW_RECEIPT_INDEX.json.

## 3. No validation/network/external data or weights/hardcoded paths/local-cache dependence

FAIL — pending evidence/audit. Main:1916 initializes weights and requests only the supplied train loader; load_for_eval is the checkpoint reader. Environment/I/O calls are enumerated exactly. Six training caches were genuinely empty per raw receipt; separately recorded eval caches were not proved empty. Calling supplied ensure_tokenizer(build_if_missing=False) is not proof that the supplied scaffold has no mutation/download path. No dynamic network/file-access trace or clean-eval-cache portability proof. Optional /tmp/neuron_cache default alone is not a forbidden-dependence finding.

Evidence: train.py SHA256 e1bd539f49658a01bf3188b70e5c04eecdb85427cc9e55db96abf487fad5a310; cold-evidence.json:1 SHA256 a836f8852c3c64f07ea4e0c79182f3179928f7b9329ab10bff669163022c687d; SOURCE_INVENTORY.json:1 SHA256 b51022aa05f9c8a9d95a598aa7d26924f3185b48323e2b7d100e58e7e6b4a6a7. Raw member paths, complete hashes and byte lengths are in RAW_RECEIPT_INDEX.json.

## 4. Measured empty-cache cold <=900s; charged training <=1800s

PASS — measured exact research receipts only. Authenticated actual FULLs only: see the exact table below and cold-evidence.json, raw-cold-receipts/* members train/rehearsal.json and operator-preflight.json; fifth boundary step4, six absent-then-created-empty paths, charged-through-exit true, stable checkpoint, successful fresh20971520/context1024 eval and released Guardian. Not a future budget promise or organiser-environment test.

Evidence: train.py SHA256 e1bd539f49658a01bf3188b70e5c04eecdb85427cc9e55db96abf487fad5a310; cold-evidence.json:1 SHA256 a836f8852c3c64f07ea4e0c79182f3179928f7b9329ab10bff669163022c687d; SOURCE_INVENTORY.json:1 SHA256 b51022aa05f9c8a9d95a598aa7d26924f3185b48323e2b7d100e58e7e6b4a6a7. Raw member paths, complete hashes and byte lengths are in RAW_RECEIPT_INDEX.json.

## 5. TOTAL checkpoint/log/cache bytes written; no large leftovers

FAIL — pending evidence/audit. Each archived checkpoint is587788862 bytes. Exact SHA/path/stability witnesses are in cold-evidence.json and raw checkpoint members. Cumulative checkpoint/log/cache writes, transients/overwrites, present retention/backup and present large-leftover inventory are UNKNOWN. Collected text bytes, final checkpoint size and free-space snapshots are not total writes. Nothing is deleted or made cleanup-eligible here.

Evidence: train.py SHA256 e1bd539f49658a01bf3188b70e5c04eecdb85427cc9e55db96abf487fad5a310; cold-evidence.json:1 SHA256 a836f8852c3c64f07ea4e0c79182f3179928f7b9329ab10bff669163022c687d; SOURCE_INVENTORY.json:1 SHA256 b51022aa05f9c8a9d95a598aa7d26924f3185b48323e2b7d100e58e7e6b4a6a7. Raw member paths, complete hashes and byte lengths are in RAW_RECEIPT_INDEX.json.

## 6. No required env beyond launch; defaults ON; optional vs required distinguished

FAIL — pending evidence/audit. Exact launch is retained including explicit --tbl_freeze_tokens 0. Source defaults/features and every getenv/environ call are in SOURCE_INVENTORY.json. Main defaults optlevel1/autocast matmult BF16, import defaults optlevel1; optional inherited runtime/compiler/cache configuration is not proved unnecessary. No clean minimal scorer-environment receipt. Optional setdefault is not by itself a violation.

Evidence: train.py SHA256 e1bd539f49658a01bf3188b70e5c04eecdb85427cc9e55db96abf487fad5a310; cold-evidence.json:1 SHA256 a836f8852c3c64f07ea4e0c79182f3179928f7b9329ab10bff669163022c687d; SOURCE_INVENTORY.json:1 SHA256 b51022aa05f9c8a9d95a598aa7d26924f3185b48323e2b7d100e58e7e6b4a6a7. Raw member paths, complete hashes and byte lengths are in RAW_RECEIPT_INDEX.json.

## 7. Fixed seed; no timing behavior beyond time-based schedule

FAIL — pending evidence/audit. Main:1916 fixes seed42. Optimizer/objective/batch schedules use charged progress. startup_adjusted_schedule_elapsed:1868 preserves official startup accounting. The end-of-budget estimate also forces a table update, and changes stop/save/rank-diagnostic choices; exact time calls are inventoried. Audit adjudication remains pending; no bitwise replay or blanket timing-only-schedule claim.

Evidence: train.py SHA256 e1bd539f49658a01bf3188b70e5c04eecdb85427cc9e55db96abf487fad5a310; cold-evidence.json:1 SHA256 a836f8852c3c64f07ea4e0c79182f3179928f7b9329ab10bff669163022c687d; SOURCE_INVENTORY.json:1 SHA256 b51022aa05f9c8a9d95a598aa7d26924f3185b48323e2b7d100e58e7e6b4a6a7. Raw member paths, complete hashes and byte lengths are in RAW_RECEIPT_INDEX.json.

## 8. Readable CLEAN path; no dead/disabled experiments; per-kernel docs; shared eval path

FAIL — pending evidence/audit. Unchanged source retains explicit False/off compatibility paths and generic device/shape fallbacks. SOURCE_INVENTORY.json lists every definition and kernel docstring. Header remains inherited CLEAN-GATEFIX prose and is not rewritten to hide new bundle details. Per-kernel coverage and literal no-disabled-path wording require audit; no docstring/source cleanup is made. Same-kernel runtime proof remains item2.

Evidence: train.py SHA256 e1bd539f49658a01bf3188b70e5c04eecdb85427cc9e55db96abf487fad5a310; cold-evidence.json:1 SHA256 a836f8852c3c64f07ea4e0c79182f3179928f7b9329ab10bff669163022c687d; SOURCE_INVENTORY.json:1 SHA256 b51022aa05f9c8a9d95a598aa7d26924f3185b48323e2b7d100e58e7e6b4a6a7. Raw member paths, complete hashes and byte lengths are in RAW_RECEIPT_INDEX.json.

## Scope and follow-up

Original public raw archives remain at the exact canonical paths and SHAs in cold-evidence.json. Selected original cold/eval/terminal/log text members are copied without byte changes inside raw-cold-receipts JSON containers; each text's SHA/byte length is independently checked. Full raw member indices are retained. Model/checkpoint tensors, private data, credentials and caches are not packaged. This is not a self-contained copy of every external source/CPU/host receipt.

Existing source review is research/v2/reviews/20260929-r109-ce2-flat-z-independent/VERDICT.json SHA256 bf2046fe3947fe8fc741bcc3deeb594eff20a4c936b7afe8bac36eb5477afe12; its limitations are preserved. It is not a new fresh-session review or this package's Claude verdict. No R143 LaneB acceptance is inherited. Separate independent package integrity/compliance review and human submission authority remain pending. Poojith alone may decide whether and what to upload. All previous packages and diagnostic checkpoints remain unchanged/protected.
