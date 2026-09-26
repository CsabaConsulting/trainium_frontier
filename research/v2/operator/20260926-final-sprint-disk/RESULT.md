# Final Sprint disk recovery — 26 September 2026

Both hosts were idle at 06:55:42 UTC: no training/compiler/driver process, empty Neuron device, fuser exit1, STOP clear, and the original lock acquired exclusively then released without replacement (trn21 device66305/inode596823; trn22 device66305/inode832589). No training was launched by this task.

The new human Final Sprint directive authorizes removal of discarded **already-scored** checkpoints and private compile caches, never best/candidate/submission/shared-active artifacts or worktrees. The Supervisor additionally classified withdrawn3e79 and startup-truncated A2r1 as discarded. A2r2 remains unscored and outside this cleanup scope. No new science or selection threshold was changed.

## Actual removals

All checkpoint bytes were SHA-256 verified, regular uid1000/device66305/nlink1, with no-follow parent traversal and stable descriptor/name identity. Every selected host's files were verified before its first unlink, under the original exclusive lock and fresh ps/neuron-ls/fuser idle checks.

| Host | Exact discarded run / artifact | Archived score / disposition | Removed allocated bytes |
|---|---|---|---:|
| trn21 | `fs-20260926-arch3e79-primed-trn21-r1-full-packet/train/out/final.pt` under `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/` | 0.9931291951444726; DISCARD | 453,054,464 |
| trn21 | `fs-20260926-arch75b9-recursive-trn21-r1-full-packet/train/out/final.pt` under the same out directory | 0.9896621160468637; selector DISCARD | 453,054,464 |
| trn22 | `/home/ubuntu/trainium_l2_unet8x1024_93ae_20260925_trn22_r1/native/fs-20260925-l2-unet8x1024-93ae-trn22-full-r1/out/final.pt` | 1.017340669168314; DISCARD | 453,050,368 |
| trn22 | Six private children `compiler`, `neff`, `local`, `hlo`, `inductor`, `debug` of `/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-ct-arch3e79-cold-trn22-r1-packet/train/caches` | Retired3e79 lineage: scored0.9863403052402583 and0.9883603019664198; Supervisor explicitly discarded lineage after nonconfirmation | 373,571,584 |

trn21 checkpoint unlink times06:57:37.695858/06:57:37.712927 UTC; trn22 checkpoint06:57:41.409254 UTC; six-cache cleanup completed06:59:33.771701 UTC. The checkpoints were permanently removed from those hosts; no backup copies were made. The removed compile caches are regenerable. All run logs, scores, verdicts, source files, historical3e79 best-point checkpoint, and submission/current candidates were retained.

The private cache census matched exactly immediately before cleanup, including every entry identity, size, owner, and timestamps; no symlinks, hardlinked regular files, special files, or live process references occurred. Descriptor-relative symlink-resistant removal deleted only the six preselected child directories. The parent cache directory and its source/run root remain. Original locks were released normally and devices stayed idle.

## Provenance and protected exclusions

- Exact checkpoint hashes and ledger bindings are frozen in `cleanup_checkpoints.py` and actual cleanup receipts. trn21 archived proof: `research/v2/verification/20260926-architecture3e79-trn21-result-review/FINAL_RESULT.json` and `research/v2/verification/20260926-architecture-recursive-full-result/ledger-row.json`. trn22 original93ae disposition is in `research/v2/ledger.jsonl`.
- The retired3e79 private roots are bound by full binding SHAbb24dee24176414b014997c039b3465ed4a309bbef048ba952eaf414fcc30823, LR.012 binding SHA6e741a8d230ad35545ce90d6fdf25f797a975eed85abf30c0d5484ee4a2a5f17, and original cold binding SHAffc82b6a96779b6e76f4fa360e4276f76f79fa4de217445075a8c439a31514f7. Both scored run ledger environments identify exactly these roots. The old disabled3e79 control fallback must remain retired.
- Local independent provenance analysis found no current binding to those private roots. Adopted36, Recursive36, and e1e2 submission cache roots are distinct. All shared Sep24 train/eval caches were preserved; A1/A2, LR.0175, bcbd, original93ae and modern trn21 arms used those shared roots and therefore cannot justify wholesale deletion.
- Older autoloop discarded checkpoints already had prune receipts. Failed/unscored diagnostic caches, P native profile traces, worktrees, runtime packages, training data and evaluator files were not cleanup targets.

## Disk outcome and limits

- trn21: latest completed cleanup free **15,650,459,648 bytes** (15.650GB, approximately14.576GiB), an observed increase of906,108,928 bytes. Short of25GB by9,349,540,352 bytes.
- trn22: latest completed cleanup free **18,026,799,104 bytes** (18.027GB, approximately16.789GiB), observed increases453,050,368 +373,571,584 bytes, with unrelated small filesystem drift between reads. Short of25GB by6,973,200,896 bytes.
- No further high-confidence scored/discarded private cache was identified in this bounded task. The25GB maintenance target was **not achieved**, and protected storage was not deleted to manufacture it. The target is not a new launch gate: future assigned jobs still use their actual measured disk/RAM requirements and fresh machine checks.

Current instantaneous state: both hosts idle; no claim of ongoing training. Existing historical utilization receipt `research/v2/analysis/20260926-overnight-report/recorded-intervals.jsonl` proves only a lower bound for2026-09-25 19:30→2026-09-26 02:00UTC: trn21 >=70.6542%, trn22 >=62.7845% native compile/train/eval busy. It excludes CPU gates and missing intervals; it is not current/full utilization and its complement is not measured idle.

Raw receipt SHA-256: trn21 status826d634d87d63b5a545110f508266a3b71ac3494c225e27d4b09e6c723d1c457; trn22 status9f0e807eef30f8b8ead2ca4ea781f62b139cf711b46d7716aadd20d8b7f2a8ec; trn21 checkpoints2406960c5833f47427094250d93a4783a5754ee313c25d1cc2a28407556df94d; trn22 checkpoint5200927b725beb6ee098ae439b5f44b1055fc2b1ea60b622f10d94ff1bdb52fe; cache inventorya96c8d8665981f1c7faaf7d5cc1f609f86a373a43810c011493e0a4df855fe68; cache cleanup8de914225018447ec340736dafc62cae2156a2d2e8a223842e2d1870f1179eb5.
