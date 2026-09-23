# trn21 deterministic closed-artifact cleanup — 2026-09-23 r8

Status: complete. All authorized exact targets were removed and postflight passed.

Purpose: recover disk space on trn21 without changing any experiment evidence, cache, data,
configuration, circuit breaker, or AWS lifecycle state. The active Operator task authorizes only
closed controller-created run checkouts reconstructable from pushed evidence and 640-wide stack
checkpoints older than 24 hours with pushed evaluations and verdicts.

## Control-plane and preflight

- Controller repository: `/home/poojith-devan/Desktop/Trainium`
- Branch: `agent/control-plane-20260921`
- Local HEAD and `origin/agent/control-plane-20260921`:
  `f4ad6c5bd09098752f4a263d6cba47a41f505283`
- Worktree before this receipt: clean.
- Inventory timestamp: `2026-09-23T17:18:29Z`; target-hash timestamp:
  `2026-09-23T17:20:28Z`.
- Host: `ip-172-31-34-85` (`trn21`).
- Root filesystem before: 206,900,281,344 bytes total; 199,048,286,208 bytes used;
  7,835,217,920 bytes available; 24,996,074 inodes available.
- Memory: 133,975,027,712 bytes total; 131,419,213,824 bytes available.
- User service: inactive/dead, result `success`, `ExecMainStatus=0`, `NRestarts=0`.
- No `train.py`, `torchrun`, `run_experiment.py`, phase-timer, reconciler, or autoloop
  process was present. `/dev/neuron0` had no holder (`fuser` rc 1); `neuron-ls` reported
  the device and `neuron_processes: []`.
- A fresh `/proc/*/{cwd,root,exe,fd/*}` scan found no link into
  `/home/ubuntu/workspace/autoloop_runs`.
- No training, evaluator, cleanup service, breaker, config, or instance lifecycle action was
  started or changed.

## Controller-created run checkouts authorized for exact removal

Every path below resolved byte-for-byte to itself directly beneath
`/home/ubuntu/workspace/autoloop_runs`, was a directory (not a symlink), had the expected HEAD,
had exactly three expected uncommitted run artifacts (log, launch receipt, ledger), contained zero
`final.pt`, `*.pt`, or `*.ckpt` files, and had no process link/holder. The remote log, launch, and
ledger SHA-256 values matched the local retrieved artifacts exactly. All result, verdict, manifest,
and checkpoint-prune evidence cited below is tracked at the pushed control-plane HEAD.

| Exact path | Apparent bytes | Directory mtime UTC; age at inventory | Expected/actual HEAD | Remote/local log SHA-256 | Remote/local launch SHA-256 | Remote/local ledger SHA-256 |
| --- | ---: | --- | --- | --- | --- | --- |
| `/home/ubuntu/workspace/autoloop_runs/9a4caa7b--20260922-bf16-norm-output-policy-r3--al-20260922-bf16-norm-output-policy-r3-full-r1` | 190,093,336 | `2026-09-22T19:40:16Z`; 78,049 s | `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a` | `1570677fa084f37cac2b1e2585d083f60782a7b0877f158b1d419ad234fc0eed` | `9f0e4ba9b26a752f72b07fa8bac633869907bb946cc7f59cddfd146efef4ff0a` | `e9215099dcba019e4b488e66ca6878ef9bf702c19b3a7af7a6fec6cb224750e2` |
| `/home/ubuntu/workspace/autoloop_runs/9a4caa7b--20260922-no-ngram-ve-only-default-shape--al-20260922-no-ngram-ve-only-default-shape-full-r1` | 190,091,099 | `2026-09-22T17:29:17Z`; 85,908 s | `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a` | `8c3a37a4b1fdd4ef45931b52cc1c59e3166f1054562a6e4f6bfe4495b1c3b31e` | `2cd439013fcf2cc85aa52192f797b24e1e40d1875f3660a52880846d712ed1cf` | `812e1c440cd128d4d66dac8478722ac3c674fcb9fa884cf4d91007bbd8b32998` |
| `/home/ubuntu/workspace/autoloop_runs/9a4caa7b--20260923-ngram-ve-cooldown-lr-flag-only-default-shape--al-20260923-ngram-ve-cooldown-lr-flag-only-default-shape-full-r1` | 190,085,926 | `2026-09-23T15:38:18Z`; 6,167 s | `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a` | `9723d17ad2bc3c958cf9e53c5a5a7f61d9e62d259b4fb935e776354ca1b95fa1` | `8390018465647986f04a3ad6afdafbb5fd5970e6dca8e512879e0d3ccee254db` | `c21e10191261298dae0260e05fbca9408a97746f9e863ae13b53c19292c6ef1a` |
| `/home/ubuntu/workspace/autoloop_runs/9a4caa7b--20260923-ngram-ve-cooldown-lr-flag-only-default-shape--al-20260923-ngram-ve-cooldown-lr-flag-only-default-shape-smoke-r1` | 190,065,116 | `2026-09-23T15:19:09Z`; 7,316 s | `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a` | `28c2087fc0a9ab8fdc72e0821d38c28de32e297f41dcd66979e8c293af267e65` | `a7f39c18d623751349d0fa083a00f151ac93fa5dd682edbf17ad72ebd226fa8f` | `b526c8862aee0accefbbf8ef018e6b3277c3fb7db3d527fe2e58691b4da5c72a` |
| `/home/ubuntu/workspace/autoloop_runs/9a4caa7b--control-trn21-lnc2x4-9a4caa7b--al-20260922-control-trn21-lnc2x4-9a4caa7b-full-r1` | 190,085,320 | `2026-09-22T05:06:03Z`; 130,502 s | `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a` | `23f29f1f2e665eea59187ff081a6d13d2da33ac9df4a8605bdc10500c65faaf6` | `d1738780e7571019394b94d64abfbbf4c46b9b466467d78e3819cda7eb035aaf` | `854fa04af392982d37836cdea095d28d74780589ee579629596358745cd8a073` |
| `/home/ubuntu/workspace/autoloop_runs/9a4caa7b--control-trn21-lnc2x4-9a4caa7b--al-20260922-control-trn21-lnc2x4-9a4caa7b-smoke-r1` | 190,064,508 | `2026-09-22T04:52:48Z`; 131,297 s | `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a` | `2d64ed8b188f1bf58535ed23b76edb46a4ea63a087717c0a35399fcc91ffe0b0` | `5f91b436279f549ab751fb7ad239d740e085286de3a46e10d7c412c3be85a243` | `f41e3ab9b2057c3ca7980fc327c308ae759ef2c1e27f7a54395bb6b3c6bef3f3` |
| `/home/ubuntu/workspace/autoloop_runs/b2a5bf5b--20260922-bf16-mlp-input-norm-r3--al-20260922-bf16-mlp-input-norm-r3-full-r1` | 190,113,464 | `2026-09-22T22:11:42Z`; 68,963 s | `b2a5bf5ba7e6f3c071a4fa941f95b7723534237b` | `ebb8604eef98e49520aa2551e2935c0fdc18b71cf583c86353d1e8d753cfcaf9` | `59527f1c976c7c9cce8be19fd4996b57a2adb3239341471c7163716f85763e0a` | `eb35576962387b3cd033356c22f4f721f572939864cad5da00ff49dbbe449248` |
| `/home/ubuntu/workspace/autoloop_runs/b2a5bf5b--20260922-bf16-mlp-input-norm-r3--al-20260922-bf16-mlp-input-norm-r3-smoke-r1` | 190,089,905 | `2026-09-22T21:16:41Z`; 72,264 s | `b2a5bf5ba7e6f3c071a4fa941f95b7723534237b` | `4ca5ea970290b15bc5a5c4cd6b343caa9ad502e840c76605b44a22d8c6d93d0f` | `1ced2303be3733042b539d91d04269f21e722be3f672518bea28bdce9ba655b7` | `566b5b138648f14a9545d8776efd52b37ea7437585305e7a226b921ffccec6f4` |
| `/home/ubuntu/workspace/autoloop_runs/b8fab55a--20260922-bf16-mlp-input-norm-r2--al-20260922-bf16-mlp-input-norm-r2-full-r2` | 190,166,337 | `2026-09-22T21:32:38Z`; 71,307 s | `b8fab55a4a6a21098cad7200a0001000aa9d76f3` | `e5339fb55518728543962c887a7485b3d21007eecd499b3890286930b03a454a` | `fa5217328934140e30dbadddfd1ac0053e5e0091c56e3f6cf8802e50eabced13` | `0cfb94f49b705316c834067916670d9e3e9971e7185fe1692a17ca3917e6b3c7` |
| `/home/ubuntu/workspace/autoloop_runs/fff4b9ba--20260922-bf16-norm-output-except-mlp-input--al-20260922-bf16-norm-output-except-mlp-input-full-r1` | 190,147,772 | `2026-09-23T00:20:37Z`; 61,228 s | `fff4b9ba808e18f8012537cb80b445889f2a9608` | `1bd980ec66849f3a4d0815e38f4f60fd6031858ce58e97cc871570e8c8b04fcd` | `d6f7bb7a1a0af17c47a62a4572921a8dcb65bf767f052166fd833d81c894f548` | `dce213b79d05adf1e0d1f1731a8c61393b48f462ee1b5a86b1ff75cb2265724a` |
| `/home/ubuntu/workspace/autoloop_runs/fff4b9ba--20260922-bf16-norm-output-except-mlp-input--al-20260922-bf16-norm-output-except-mlp-input-smoke-r1` | 190,124,094 | `2026-09-23T00:09:09Z`; 61,916 s | `fff4b9ba808e18f8012537cb80b445889f2a9608` | `db18314f1d65a59777e9ea3208ec20f518028940b7c20bfd11355d1fc1de877e` | `3a5f0d1bb07e57a45bd18090fa4494106498cfb332ec8c3c07cde33682ba66fa` | `8281beeeead9d597d5d4552f691791184975b3e6d1e91c49592502c2efced3dc` |

Total apparent bytes for these 11 directories: **2,091,126,877**.

### Pushed closure and reconstruction evidence

The evidence roots are `research/agentic/experiments/<proposal>/`; each cited tier has tracked
`artifacts/<tier>/`, `<tier>-result.json`, and `<tier>-verdict.json`. The exact verdict/prune
receipt hashes and commits are:

| Proposal/tier | Verdict SHA-256; pushed commit | Checkpoint-prune receipt SHA-256; pushed commit |
| --- | --- | --- |
| `20260922-bf16-norm-output-policy-r3/full` | `521e0ad86afcb6a515b6a680dc55b4c7b55948f1b3047dbd5881303c62748802`; `fe6f234251b658b3c695f25cac3828cdc695704c` | `a3e453329923bd1280ea2a880b91265ee729caabdec231459012e0c5eefa5075`; `ada5c4f95453d7c36302e1fa3e149a368654bf4e` |
| `20260922-no-ngram-ve-only-default-shape/full` | `b924767e15499e260a944a1d5ad527360a833890930adceb32f7b890dad88b81`; `475c345ad8bbe05e56659074fe3c19d2bd334568` | `3fd7d78d9cd81fb65094c372b0fcb4fb29e79ce3b7e9af1c9ce92c17a47f7fe7`; `83849903dbe29044c93539d22307e47f0a68bfae` |
| `20260923-ngram-ve-cooldown-lr-flag-only-default-shape/full` | `6890e14353d683adc055339b5c5ed88b89bc9d8b8646f353832cee110273cbbd`; `ac84104e21af4db475f04ebd44645ebcb2d3474c` | `7bec1be6c1d099ead37fb606c989f67e49325480ea965232a0f37656246c8189`; `2245e840e085d600ec1e97aa131905d4f1721325` |
| `20260923-ngram-ve-cooldown-lr-flag-only-default-shape/smoke` | `6a48507253d9f1ee178d8b3fe5ec99dce8545ba0faf804e767404ff2d38c61cd`; `d6779966190dcd7c26f4e180894d35f79f63ade4` | `05ee7243b69c9761aafea49ad86b6a91975876837ab1d7066682cd415aea4228`; `83ff55e93e978a016dea0a1484e62e234cd58010` |
| `control-trn21-lnc2x4-9a4caa7b/full` | `d5c4fa73ddc1fbab52c1d93d57bf6b1034c446829ddf6e540b519b93773c4554`; `a812b2c0155e4a92b004a4a63e84ebbcfaa7ffcd` | shared cleanup receipt `f65d0f0c91f86b641254b1c728679b5b901b718ae0137eeb95bf5f6794fcadcc`; `a4dedebab233542ce7724d6b9c01773a938f1760` |
| `control-trn21-lnc2x4-9a4caa7b/smoke` | `b65522276c0e0b16455b9c261e48db08f9bf4e5712d72be433f308617605f35c`; `bc91599c250dd483946f142218fdb6773dbf5faf` | shared cleanup receipt `f65d0f0c91f86b641254b1c728679b5b901b718ae0137eeb95bf5f6794fcadcc`; `a4dedebab233542ce7724d6b9c01773a938f1760` |
| `20260922-bf16-mlp-input-norm-r3/full` | `c70dbbd51690c9e39e811c71f88a5d2c9b5a45588a21a80ec9c4dc3965d0047c`; `3321605b6c6cb021d6e3261390f5e8941153c4b2` | `dde94d681bf37f44abcaa361973aac96db614febf2b5276bbeb13654a895eee0`; `83b10f71546dcf2c39f7fd267966680cc208b803` |
| `20260922-bf16-mlp-input-norm-r3/smoke` | `d1eb5fcb055b3f96caa051c8d389537b5b3aae0371ad115e7b54ce749a6dd331`; `4bcf5c753c3c2fe974e7ad483b2e2e1ae9dbc58c` | `6a64db75b111d17f4eb3710b0f5d07d2462fb34818a43094c727582e2c011834`; `4fefd8a0cf0b5ba861e75a351940581667ec404d` |
| `20260922-bf16-mlp-input-norm-r2/full` | `a4262529be0bf666e136e33fe4018d704bc8762ee0fdfa396254b9edc86bfeec`; `f19a968d6aeb3c83bbfd09a8189410cb0df3347f` | `48b5bf8261ce6cdf8872cbe4b7b2d1f9cf3a00973afa543cb68aff5b6e5db7a5`; `91164450f344afec5084307d213ddab94750c430` |
| `20260922-bf16-norm-output-except-mlp-input/full` | `9c4fb68d9fa8daf465ee8bfae576d681ab5956ddfa74a8444973ea82ebfd3b98`; `30dcdaf7301b93f8fe68bd6f377e5aeaad1ed1a9` | `6ec0f25a1a7f1ba0c1a260bc34ebfdf331df1125dd79bdfdef4c9d483f2e91b9`; `067e7d748c9c63808d2b5f3dbaee39fed61f5814` |
| `20260922-bf16-norm-output-except-mlp-input/smoke` | `5ccca63417e795c1d2c0071753ce8185d9e3281c796f97956aa19868b292bcaf`; `c20a45dead7aea58625f2b313ab8ef210395f3b7` | `a58eb4cd35f0176cb021852a8a6227b042d41051a58b0f3802ebbd2a3f3c6943`; `25157e48a2bb2f732c4bbea7ee41758e76fd8e15` |

The three candidate HEADs not currently named by a remote branch are still byte-reconstructable:
each has parent `9a4caa7bfab26c1d1bb59f7ef2bb8c5392d7503a`, which is on multiple origin branches, and
`git show --binary <candidate>` exactly equals its tracked, pushed implementation patch:

- `b2a5bf5ba7e6f3c071a4fa941f95b7723534237b` ->
  `research/agentic/experiments/20260922-bf16-mlp-input-norm-r3/implementation.patch`,
  SHA-256 `fc1865d6e05d07627c02b5cd463d1baaa83f252dd61fb5b4de3af0757194b947`.
- `b8fab55a4a6a21098cad7200a0001000aa9d76f3` ->
  `research/agentic/experiments/20260922-bf16-mlp-input-norm-r2/implementation.patch`,
  SHA-256 `bb2f8569c8fc8396205eba22e7f4c9157aabfa1809048b6b654a5337baae8ad0`.
- `fff4b9ba808e18f8012537cb80b445889f2a9608` ->
  `research/agentic/experiments/20260922-bf16-norm-output-except-mlp-input/implementation.patch`,
  SHA-256 `edff764a516f42dbae690670a46e275b1a7c9d6eab744a91da14532d5a3c31b5`.

## 640-wide stack checkpoints authorized for exact unlink

Both are regular, non-symlink files, are more than 24 hours old, and have no holder (`fuser` rc 1).
Their retrieved logs explicitly record `n_embd: 640`, `n_layer: 6`, and 190,099,500 parameters.
The source experiment rows and complete public/alternate evaluation results are in pushed
`research/experiment-log.jsonl`; the closed-negative full-budget comparison/verdict is in pushed
`research/trn2-experiment-results.md`.

| Exact path | Bytes | mtime UTC; age at hash time | SHA-256 | Archived result |
| --- | ---: | --- | --- | --- |
| `/home/ubuntu/trainium_frontier/out/t39-stack-900s/final.pt` | 491,984,869 | `2026-09-13T01:03:52.523095548Z`; 922,596 s | `ddd83745e4e7ac4e224f2a8dddc2e1f48a46b04127b7a013a773cee31d9d089f` | 1,143 steps; public `val_bpb=1.058920`; alternate `1.074905`; causality passed |
| `/home/ubuntu/trainium_frontier/out/t39-stack-1800s/final.pt` | 491,984,869 | `2026-09-13T02:46:48.390151548Z`; 916,420 s | `a247de8718e46f9212c47ac4d759e29c4411e64c5fae78ea586eab1f0eb069eb` | 2,288 steps; public `val_bpb=1.021278`; alternate `1.036844`; causality passed; closed negative vs `t39-base-1800s` |

Archive integrity:

- `research/run-logs/t39-stack-900s.log`: remote/local SHA-256
  `0ea64d0143ec027ccb32fd739945011e45bdac17edd75d0a520583ad64d76572`, pushed commit
  `a2b38fba6afde09c716b8bdc847d861a97f6761e`.
- `research/run-logs/t39-stack-1800s.log`: remote/local SHA-256
  `b0e4dc14f794266e9ac4e5002e62b313709cb56b6d10439537f78edfa57ee211`, pushed commit
  `2242ea52c2413605ef09969a1eb22ab463190695`.
- Exact JSONL row SHA-256 values: `t39-stack-900s`
  `4095baa79bee936a11452a2ce79b50d6f200df257ad454c7372311ed20cf7896`; `t39-stack-1800s`
  `af7515d96bbf780b0d8dfe843d18be329bae37bf4108f9353b3522f345a7683d`.
- `research/experiment-log.jsonl`: SHA-256
  `e7d9554d9e89b49b40fea356e82c5361851f0d5a1290c04cc2109202de0d47bc`, pushed commit
  `934f3cbd96fbadd7d0a043c5230f6a1fb8c79167`.
- `research/trn2-experiment-results.md`: SHA-256
  `d9204f939c42f4ba23e0bd0a05e97a966e8583e7a9bb59b41c53960261a0e966`, pushed commit
  `7a74a8fe689d1c18db6d233ce6894bb0c68413df`.

Total checkpoint bytes authorized: **983,969,738**. Total apparent authorized target bytes across
the 11 directories and two checkpoint files: **3,075,096,615**.

## Retained exclusions

- All run logs, JSON/JSONL, manifests, result rows, verdicts, proposals, and cleanup/prune receipts
  are retained locally and in pushed history.
- All data and cache paths are excluded.
- The inventory found 107 `final.pt` files under `/home/ubuntu/trainium_frontier/out`; only the two
  exact T3.9 640-wide stack files above qualify under this authorization. The other 105 are retained.
- No other controller run directory existed under `/home/ubuntu/workspace/autoloop_runs` at the
  inventory boundary; paths elsewhere on the host are out of scope and retained.
- Any target whose identity, hash, holder, pushed evidence, or reconstruction gate changes before
  removal must be retained fail-closed.

## Post-delete result

- Final fail-closed preflight: `2026-09-23T17:23:25Z`.
- Available bytes at final preflight: 7,835,193,344.
- Every identity, type, resolved-path, HEAD, artifact hash, checkpoint hash, age, no-checkpoint,
  no-holder, idle-service, idle-process, and Neuron-idle gate passed before the first removal.
- Deletion completed: `2026-09-23T17:23:28Z`.
- Only the 11 exact directories and two exact files listed above were removed. Every literal target
  was then confirmed absent.
- Available bytes after: 11,073,384,448; used bytes after: 195,810,119,680; filesystem use 95%.
- Available-space increase from the final preflight: 3,238,191,104 bytes. The difference from the
  3,075,096,615 apparent target bytes is filesystem accounting/checkout metadata overhead.
- Remaining directories directly under the bounded controller base: 0.
- Remaining `final.pt` files under `/home/ubuntu/trainium_frontier/out`: 105, all retained.
- Postflight user service: inactive. Relevant process count: 0. `/dev/neuron0` holder check:
  `fuser` rc 1. `neuron-ls` process count: 0.
- Logs, JSON/JSONL, manifests, result rows, verdicts, data, caches, configuration, circuit breaker,
  and AWS lifecycle state were not changed. No training or evaluation was launched.

The removed run checkouts are recoverable from the pinned parent/HEAD and pushed implementation
patches/evidence documented above. The two deleted checkpoint binaries themselves are not
recoverable from the host after unlink; their exact hashes and all experiment/evaluation evidence
remain durable in pushed repository history.
