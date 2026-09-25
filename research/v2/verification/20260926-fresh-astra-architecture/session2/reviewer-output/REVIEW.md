Verdict: **PASS — limited to the supplied source change and CPU evidence.**

I am a fresh same-family reviewer under the user's substitute policy. I independently reviewed only this packet and wrote and ran the bounded CPU test retained here. I did not consult prior reviews, implementation notes, other agents, historical branches, or external services. This is not a cross-family review, native-readiness clearance, submission decision, or hardware launch. The default was REFUSE until the source identities, receipt checks, and independent exactness test were complete. No blocking finding remains within this review's scope; no additional evidence is required for this limited PASS.

The original source is associated in the packet with commit `1e21c3d797eed764f108c41546b270065ad5650c`, and the candidate with `b4d83b5db66971f3c348837f04f53cba65aa6d33`. There is no Git repository, so those commit associations are supplied metadata. I reconstructed both complete source texts from the single full-context diff and independently verified these exact content identities:

| Source | Lines / bytes | SHA-256 |
|---|---:|---|
| Original | 4944 / 323017 | `93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5` |
| Candidate | 4948 / 323377 | `3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c` |

All 26 manifest-listed input files match their recorded sizes and SHA-256 hashes. The manifest itself was also hashed, and no unlisted packet files were found. The complete input hash inventory appears below and in [input_audit.json](/work/input_audit.json). Reconstructed copies are [original_train.py](/work/original_train.py) and [candidate_train.py](/work/candidate_train.py); no packet or candidate source was modified.

The change consists of exactly four added lines, with no deletions, in `workflow/proposals/research-v2/train.py`:

- [Line 3507](/work/candidate_train.py:3507) adds `--initial-cache-refresh` using `argparse.BooleanOptionalAction`, with literal `default=False`. The independent test executes this actual parser statement and verifies omitted, explicit OFF, and explicit ON arguments.
- [Lines starting at 4032](/work/candidate_train.py:4032) refresh the existing `cached_linears` only when `step == 0` and the flag is ON. With the flag OFF, this branch performs no tensor work. The existing cache selection, forward, backward, optimizer, and post-update refresh paths are unchanged.

The existing [Linear implementation](/work/candidate_train.py:1463) initializes `_cached_weight` to `None`, casts the FP32 master weight to the input dtype on an uncached forward, and uses a populated cache only if its dtype matches. `refresh_weight_cache` performs the same differentiable `.to(dtype=...)` operation; it neither detaches nor mutates the master weight. The new call executes with gradient tracking enabled. The existing [post-optimizer refresh](/work/candidate_train.py:4096) remains in place, so the ON flag makes the first forward encounter the populated cache state already used by subsequent steps. The [installed MLP adapter](/work/candidate_train.py:4540) also selects these caches. The submitted adapter requires BF16 compute, uses the native-forward/custom-backward path with weight mode zero, and does not invoke packed-weight refresh in this configuration. These facts support the intended removal of a `None` versus Tensor cache-state distinction. They do not prove fewer native compilations or faster startup.

Cache priming is after [the first-step timer starts](/work/candidate_train.py:4023), after gradient zeroing, and before the first forward. It adds no preliminary training step, optimizer update, data fetch, RNG draw, synchronization exemption, or clock reset. Model architecture, initialization, scientific settings, optimizer equations, learning-rate scheduling, evaluator, tokenizer, data, scoring, and gate thresholds are unchanged by this diff. The required training budget remains **1800 seconds**. The unchanged [startup accounting](/work/candidate_train.py:4104) caps its allowance using `TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS` and charges excess, including the existing conservative 30-second launch margin. The lane fixes the permitted exclusion at **900 seconds**. The imported `prepare.py` is not supplied, so its constant definitions cannot be independently inspected here; the receipts record the same prepare hash and 1800-second budget. Fake-clock CPU runs do not validate real elapsed-time accounting.

The independently authored [reference_exactness.py](/work/reference_exactness.py) extracts the actual `Linear`, MLP adapter/autograd definitions, parser statement, cache collection, initial-refresh branch, and post-update refresh from both hash-verified source texts. It does not import or allocate the full training model. Six cases cover BF16 linear, FP32 same-dtype aliasing, transposed BF16 linear, dtype-mismatch fallback, the small-token MLP fallback, and the active custom-MLP backward path. Each compares original, candidate default, candidate explicit OFF, candidate ON, and a cache-disabled control. Linear cases additionally compare against direct functional linear/matmul computation. The MLP control retains the source's existing custom backward math, isolating cache effects rather than claiming a new audit of that algorithm.

Every arm runs three tiny AdamW steps with two independently backpropagated microbatches per step. The test compares exact tensor bytes, including outputs, losses, input and accumulated weight gradients, parameters, and AdamW state. It checks cache freshness, differentiability, RNG and master-weight invariance, parameter identity, checkpoint schema, and refresh counts. ON refreshes four times per module over three steps; default/OFF/original refresh three times; the cache-disabled control refreshes zero times. The custom MLP path is explicitly asserted to execute. This local optimizer is a comparator, not a reimplementation or independent gate of the production Muon/owned optimizer.

Actual successful command, exit **0**:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /usr/bin/timeout 120s /opt/cpu/bin/python -u /work/reference_exactness.py > /work/reference_exactness.raw.log 2>&1
```

Result: **6/6 cases PASS, 9,248 assertions, 2,582 bitwise tensor comparisons, zero mismatches.** Python 3.14.4 / PyTorch 2.13.0+cpu; seed 42; one intra-op and one inter-op CPU thread; largest width 64, context 128, microbatch 2; no compilation and no competing tensor job. Full stdout/stderr is in [reference_exactness.raw.log](/work/reference_exactness.raw.log), structured results in [reference_exactness.results.json](/work/reference_exactness.results.json), and all compared tensor snapshots in [reference_exactness.tensors.pt](/work/reference_exactness.tensors.pt). The raw log retains a CPU-library `/proc/cpuinfo` parsing diagnostic emitted at import; execution and assertions completed successfully. I did not inspect `/proc` or process environment.

One exploratory test assertion initially failed and is preserved transparently in [reference_exactness.attempt1.py](/work/reference_exactness.attempt1.py) and [its raw output](/work/reference_exactness.attempt1.raw.log). It assumed caches never enter `state_dict`, which is too broad for the pre-existing FP32 same-dtype path: `.to(torch.float32)` returns the master `Parameter`, and `nn.Module` registers `_cached_weight` as an alias. ON exposes that alias before the first forward; original/OFF expose it after the first optimizer step. This behavior is at the unchanged refresh assignment in both sources, outside the submitted FP32-master/BF16-compute path. The corrected test explicitly checks the alias schema in those auxiliary cases, keeps the stricter non-parameter-cache assertion for BF16, and still requires exact computation, parameter identities, and optimizer state everywhere. No source fix or numeric tolerance was introduced. This is an informational scope limitation, not a blocking regression in the submitted configuration.

The independent [receipt audit](/work/audit_receipts.py) completed **315 assertions**, exit **0**. It checks every raw rank JSON against its aggregate record, source hashes, explicit flag arguments, candidate launch/request/terminal consistency, exit codes, timeouts, rank completeness, finite recorded losses, parameter-change counts, ownership plans, checkpoint-check fields, raw log tails, and log loss values recomputed using the source's smoothing formula. PASS labels are checked for consistency only after examining their underlying supplied records.

```bash
/opt/cpu/bin/python /work/reconstruct.py > /work/reconstruct.raw.log 2>&1
/opt/cpu/bin/python /work/audit_receipts.py > /work/receipt_audit.raw.log 2>&1
```

Both commands exited **0**. Their full outputs and structured artifacts remain in `/work`.

| Recorded rank | Step 0 loss | Step 1 loss | Step 2 loss |
|---|---:|---:|---:|
| 0 | 9.009672164916992 | 9.021651268005371 | 9.053120613098145 |
| 1 | 9.012287139892578 | 9.020283699035645 | 9.050683975219727 |
| 2 | 9.013776779174805 | 9.019458770751953 | 9.050694465637207 |
| 3 | 9.012662887573242 | 9.0178861618042 | 9.058394432067871 |

All 12 loss values have identical parsed floating-point encodings across reference, candidate OFF, and candidate ON. All 65 supplied final rank-0 parameter digests also match exactly across the three runs. Every rank reports 65/65 changed parameters, no cross-rank mismatches, and three optimizer steps. All three recorded torchrun exits are zero without timeout. The ownership plan and checkpoint claims match; rank 0 reports deterministic checkpoint logits of shape `[1, 256, 8192]`. OFF is therefore supported as identical over the three recorded steps by the numerical receipts, final digests, unchanged source path, and the independent tiny exactness run. ON meets the supplied CPU G2 evidence requirements and additionally has the same recorded numerical results.

The evidence has specific limits. The production-size receipts describe four CPU ranks, an 8x1024 architecture, a synthetic tokenizer/loader, a BF16 override, a no-op replacement for the Neuron CE installation, a deterministic 1 ms fake clock, and training inputs sliced to 2x256. The submitted full-size gate was **not rerun locally**; only the authorized tiny model fragments were executed. Whole-model compilation is disabled, but logs still show three Muon body compilation specializations and `compile_muon=True`. The RMSprop trace has zero calls and no tables, so it proves no RMSprop update exactness. Logs' throughput, budget usage, and startup numbers are not performance measurements.

The gate harness, original prepare source, checkpoint files, full tensors, and per-step parameter digests are absent. I cannot recompute the supplied production digest values or checkpoint checks. Final digests are only 16 hexadecimal characters (64 bits), and only rank 0 supplies their values; the other ranks supply empty mismatch lists. Thus recorded equality is not a byte-for-byte audit of every intermediate production tensor or all rank checkpoints. The local PyTorch version also differs from the receipts' Python 3.12.3 / PyTorch 2.11.0+cpu. These limits are explicit and do not extend this PASS to native behavior or the unchanged architecture as a whole.

Pending scope: a separate **cold startup-only native run**, stopped around the fifth-step boundary, must meet the **<=900-second startup gate before any full quality rerun**. This session neither performs nor clears that gate. Native graph/compilation behavior, cold startup duration, steady-state speed, full-budget quality, evaluator results, submission authorization, and cross-family review remain unproved. No hardware run or submission is authorized by this report.

The hashes below identify the exact packet inputs read, rather than authenticating external provenance. The prepare SHA-256 recorded consistently in receipts is `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`; the candidate G2 harness hash recorded in OFF/ON requests is `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`; and the recorded ownership-plan hash is `df750f7f11b4b8dabc233cb476e486d0b6b997da2a1b32fe5af13142a394668a`. Their underlying external files are not in the packet and were not accessed.

| Packet-relative path | Bytes | Verified SHA-256 |
|---|---:|---|
| `candidate.diff` | 328630 | `3c490e56f5c7b1db834982802402dc79359eb3328aeeac2e019390ea294f96f3` |
| `lane-spec.md` | 4071 | `1d2eccf7a27c10047046254fa6dbf09c9618856e9de3a1489b9d33ef3947e5e3` |
| `raw/off/g2.json` | 18957 | `47b7b33f942670f88594e9167960f02f1929b6af5b7045fe7f08a0714d23633d` |
| `raw/off/launch.json` | 2309 | `b9d7ec55658f572ec745c0acd677e7476a0926d2d3ff4b071fc11829d50c9032` |
| `raw/off/rank0.json` | 5358 | `5563bef2e53f4ec338f9f4e8f66d1e5ca1cd797e6188a9f1cfffe6c859a2bfa4` |
| `raw/off/rank1.json` | 1461 | `623c5a9f5804f3ee24f5486e705d4026b502fdaccd7884e151328d22cbd3c253` |
| `raw/off/rank2.json` | 1461 | `853cac9f1116c6a314a699c00dfc5ed0d019604baf93f2d89a19722742c2e61a` |
| `raw/off/rank3.json` | 1459 | `c33dd4311b2c4ab126d81b504fbf42af643b826df4d9950d075c3a17f1eb6533` |
| `raw/off/request.json` | 2791 | `45dc34e378736ba84fd22c5bb4e79ceef29ac8ac4012f82432548f919e7d876c` |
| `raw/off/terminal.json` | 324 | `b1aee7bdf20db5986665fc7075dd8e1e9790cd2f00e5d2efd5876683097dc29b` |
| `raw/off/torchrun.log` | 8868 | `cf35c06180236763b1776274e76eb0528907c1f1c6ec71e3b883dbc92eb89cf5` |
| `raw/on/g2.json` | 18948 | `52cd0aba1efbec8ab23c756d0132316a3fc69ef3278aa529ac09965675914101` |
| `raw/on/launch.json` | 2302 | `f9aaa85ea5f2c1ce782581e0d44316aabe15b520381a32a6537dbed060ecd209` |
| `raw/on/rank0.json` | 5357 | `0e89d81e17fd57d45510d44b2594b835981e7c87587bfecbac10555ae169c74c` |
| `raw/on/rank1.json` | 1460 | `347d35b4c0b1c0a4f8ec65e85f797a925c5df68ca8fc09896fa843a47aa9f633` |
| `raw/on/rank2.json` | 1460 | `87bf32afd0bc1cb43c2658e07693154fedf6b4dae9923f1b77aa754e0fd4b91f` |
| `raw/on/rank3.json` | 1458 | `b6d27256c968bdf32c9c64f5cad221c0de1d59d56bb4dcbf231fa706f88e0788` |
| `raw/on/request.json` | 2779 | `63b6ce3cddb8ce3100846440674cd60510c27397f795ac2a6c165a14ac0ef5d9` |
| `raw/on/terminal.json` | 323 | `fa81dee99cd4521dc5ee23700e67a6fce8c8103925be516009e95b0320efa44c` |
| `raw/on/torchrun.log` | 8867 | `bbadab99e45ddd274fb13a206970464632adf25e663507f38e952aaf106a2e37` |
| `raw/reference/g2.json` | 18860 | `c2fee697d02322198d61951363861fb31201bd7b68d4b91f159dbd44001392b0` |
| `raw/reference/rank0.json` | 5349 | `a4f56b5fe083990c22fecbfd8abae9a8da8e9199001ce9d535cac95ad3f73b67` |
| `raw/reference/rank1.json` | 1452 | `0e1541056532a22021fbaacf0e7ece8b4ee96c68e5c60944ec03d4c203fdf2cb` |
| `raw/reference/rank2.json` | 1452 | `dbf3b9e0a63f633d19facfa10b564fe14994a8f02ba79de6ad8c45167889e83a` |
| `raw/reference/rank3.json` | 1450 | `a1601b64e87aaa5afad24d10218f33a3bf16816ad0bc052550364018bb8c3bdf` |
| `raw/reference/torchrun.log` | 8859 | `7e3d10c8ab565d637a20adac838c78988f99e9ea310d63056fa3848d8c3121ac` |
| `INPUT_MANIFEST.json` | 4148 | `96a204756ce2f465f6cf9dbee24dab4997703107a9c5494696faa933cb4354c5` |
