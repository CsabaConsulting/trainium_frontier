Decision: PASS for suitability to undergo later exact full-size CPU qualification, limited to the assigned BF16 compute / FP32 master path. No full-size qualification, native execution, performance claim, or scientific adoption is approved by this report. The decision remained REFUSE until source authentication, packet comparisons, and the independent bounded reference completed.

The review used only `/packet/lane-spec.md`, its supplied full-context diff, input manifest, and four raw receipts. Paths and historical notes mentioned inside the source/receipts were not followed. Neither full training module was imported; no full model, distributed process, compilation, accelerator, network access, or training launch was used. Candidate source was not edited. Reconstructed copies, reviewer-written programs, raw outputs, and failed exploratory attempts are preserved in `/work`.

Source authentication and causal scope

- The single full-context hunk reconstructs 4,890 base lines and 4,894 candidate lines. All six manifest-listed packet files match their recorded byte lengths and SHA256 values.
- Base SHA256: `6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`.
- Candidate SHA256: `bcbd148dd51b5a014289cbecf3f0e684e35cf537e9588c878e15b265fcbfd522`.
- The only additions are the BooleanOptionalAction at candidate line 3474 and the three-line first-step branch at 3976. Removing those two AST statements reproduces the complete base AST. The parser was executed independently: omitted and `--no-initial-cache-refresh` select False; `--initial-cache-refresh` selects True.
- `Linear.refresh_weight_cache` at 1464 performs `self.weight.to(dtype=dtype)`, under enabled gradients in the normal loop. It neither replaces nor detaches the FP32 master. `Linear.forward` at 1467 and the MLP adapter's weight selection at 4486 use a cache only when its dtype matches the input. BF16 caches retain the cast's autograd connection to the original master.
- The new branch follows gradient zeroing and precedes the first microbatch forward. It is restricted to step zero and the existing `cached_linears` list (3917). The ordinary refresh at 4042 remains after optimizer updates, including the existing skipped-update handling. No architecture, data loader, physical training shape, LR/schedule formula, total batch, optimizer ownership, or budget code changes are present.
- The loop/budget clock starts at 3937; the step clock is read at 3967, before the new refresh; synchronization and step-boundary recording precede `dt` at 4047. Preparation therefore remains within the existing measured step and existing startup-accounting rules. The unchanged startup allowance/reset still applies. This is a source-order finding, not proof of actual runtime, compiler cost, cold start, or budget compliance. The separate `--compile-only` early-return path is unchanged and was not qualified.

Independent exactness test

`reference_test.py` extracts only the named Linear/MLP definitions, cast/forward helpers, pure Muon update math, and the added CLI/refresh statements. It executes none of the training module's import-time installation or main routine. Its independent arithmetic oracle uses ordinary `torch.nn.functional.linear`, fresh differentiable casts on every forward, ReLU squared, and PyTorch autograd; it does not use the cached Linear forward or custom MLP backward.

The final run used Python 3.14.4 / PyTorch 2.13.0+cpu, one intra-op thread, one inter-op thread, deterministic algorithms, and a 60-second timeout. There were eight cases, each with base/default/OFF/ON/direct-oracle arms, four updates, and three sequential forward/backward microbatches per update. Fixtures contain 12 or 64 FP32 master elements. Linear/eager-MLP inputs are 2x3x4; adapted-MLP inputs are 256x4 so the actual extracted custom-autograd dispatch is taken without large weights. The adapter's installer and full model are not exercised. The adapter's `forward_backend='native'` name means its ordinary PyTorch reference forward in this test, not native accelerator execution.

Result: **PASS; 3,183 bitwise tensor comparisons, zero nonexact comparisons.** Comparisons require matching dtype, shape, and tensor bytes, with no tolerance. All compared maximum absolute differences are zero. Coverage includes:

- Default and explicit OFF identity against reconstructed base, and ON identity against both base and the fresh-cast arithmetic oracle.
- Forward values, scalar losses, input gradients, FP32 master gradients after each microbatch, clipped gradients, optimizer state, and updated FP32 master values across repeated steps.
- Standard Linear, eager two-layer MLP, adapted MLP custom autograd, transposed Linear, and cache-disabled Linear.
- AdamW with momentum state/weight decay and the source's extracted pure Muon update math with momentum/second-moment state. This establishes cache-arm update equivalence in the small fixtures; it is not an independent qualification of the complete distributed optimizer adapter.
- Direct traversal from each populated cache's autograd node to its actual master Parameter, exact cache values after every update, replacement of BF16 cache tensors, preserved Parameter identity, and nonzero updates.
- Hooks attached to the actual custom backward node fired once for each of the 12 backwards per adapted-MLP source arm. Across two adapted cases and four source arms, 96 executions were directly observed. These are reviewer-owned observations, independent of packet counter names.
- BF16 caches do not enter the saved state dict. Serialization followed by strict loading into fresh tiny modules leaves caches empty and reproduces evaluation outputs exactly. An intentionally stale BF16 cache is correctly bypassed for FP32 input, including exact backward results.

State/load boundaries and retained counterexamples

1. Hot-loading new weights into an already cached Linear does not invalidate its old cache in either source. A concrete diagnostic caches an all-ones 3x4 master, loads an all-twos master, and feeds two rows of ones: the stale output is `[4,4,4]`, while a fresh cast gives `[8,8,8]`. Explicit refresh restores `[8,8,8]`. This pre-existing behavior is not used by the checkpoint evaluation path: `load_for_eval` at 3211 constructs a fresh model before loading, and warm start at 3189 occurs before the new first-step refresh. Arbitrary hot-loading without refresh is outside the supported scope.
2. An extra FP32-compute probe failed the initial suite's no-cache-state assertion. In both sources, `weight.to(torch.float32)` on a FP32 Parameter returns that Parameter itself; assigning it to `_cached_weight` registers an alias and adds `_cached_weight` to the state dict. The complete failed attempt is preserved in `attempt1/`. The final suite explicitly reproduces and records this behavior in both sources as an out-of-lane diagnostic. It does not silently treat FP32-compute caching as supported. The requested assigned compute path is BF16; the supplied integration itself asserts BF16 at 4873, and the historical CPU harness explicitly substitutes BF16 for CPU's ordinary FP32 default.
3. A detached-cache negative control produces no master gradient and fails the connectivity predicate, showing that the checks distinguish an actual differentiable cast from a detached value copy.

Neither counterexample changes the required BF16 fresh-model checkpoint behavior or introduces a candidate-only regression in the reviewed lane. Extending the claim to same-dtype FP32 caching or hot-load invalidation would require REFUSE on the concrete examples above.

Raw packet evidence

`audit_receipts.py` compares recorded values and structures rather than adopting the receipts' PASS strings. Across base versus default/OFF/ON for all four ranks (12 paired rank comparisons), it found zero mismatches in per-rank losses, pre-clip and post-clip gradient digests, optimizer-state records, parameter-after-update digests, final digests, parameter-change counts, owner-plan records, and checkpoint records where present. Each rank reports 62 parameters changed. Default/OFF cache-observation records equal base. ON's later caches and refresh records equal base after removing the initial 40 refresh records. The initial audit assumed checkpoint fields existed on every rank; that schema error and its source are retained as `audit_receipts.attempt1.py` and `receipt_audit.attempt1.*`. The corrected audit handles the actual rank-zero checkpoint fields.

At the first recorded forward, base/default/OFF have 40 absent caches per rank. ON has 40 BF16 caches, with FP32 master dtypes, `requires_grad=True`, `ToCopyBackward0`, and `connected_first_forward=True`. ON adds 40 records at `(completed_steps=0, forward_calls=0, grad_enabled=True)`; all arms then have 40 records at each of `(1,1)`, `(2,2)`, and `(3,3)`. Thus there are 120 base/default/OFF and 160 ON refresh records per rank. These are descriptions of records, not separately reconstructed instrumentation semantics. Rank zero reports an identical checkpoint-file digest across all arms: `1bc4752a2f82da10dd944c1a58bcdc2859ccb4bedea59222d3e77bb41efe098c`. The checkpoint files themselves are not supplied; their byte equality is only a receipt claim corroborated by equal recorded checkpoint contents.

The observation-driver implementation is absent; only its hash is supplied. In particular, the fields `custom_mlp_nodes=6` per forward and `mlp_custom_backward_calls=18` are not independently established execution counts. A graph-node count or aggregate can have the same values without measuring executions. Later `connected_first_forward` values are null and cannot establish later connectivity. The independent test supplies its own execution hooks and repeated connectivity checks instead.

All historical receipts remain reduced synthetic CPU observations: four ranks, three steps, width 128, depth six, physical 2x128 inputs, synthetic prepare/data, no compilation, and a deterministic fake clock. Their reported configured token counts, tok/s, timing and budget fields are not physical full-size work or real timing evidence. No conclusion about full117M standard G2, real data, native correctness, speed, compiler graph behavior, cold start, or budget compliance follows from them.

Separate gates

| Gate | Finding |
| --- | --- |
| Suitability to proceed to later exact full-size CPU qualification of this authenticated candidate in the assigned BF16 lane | Supported by this limited review and reference test. |
| Exact full117M standard G2 CPU qualification | Not run for this exact source; remains required before claiming that qualification. Tiny-width equality is not a substitute. |
| Fresh-session policy gate | Not separately certified by these numerical tests. This report is a new supplied-packet review; the input manifest's `reviewer_session_started:false` is packet metadata, not evidence that another policy/session gate has passed. No separate policy source is supplied or consulted. |
| Native execution or scientific adoption | REFUSE / not authorized by this evidence. Manifest `native_authority` is false. Native/compiled correctness and real timing/budget gates remain separate. |

Artifacts and reproduction

- `authentication.json`, `reconstruct.py`, `base.py`, `candidate.py`: authenticated reconstruction and exact four-line change list.
- `extracted_base.py`, `extracted_candidate.py`, `extracted_controls.py`: definitions/statements actually selected for the CPU reference.
- `reference_test.py`, `reference_results.json`, `reference_raw.pt`, `reference.stdout.txt`, `reference.stderr.txt`: reviewer-owned executable reference, every tensor comparison, raw input/weight/output/gradient/update/state tensors, and actual process output. `reference_raw.pt` contains tensor/container data and can be read with `torch.load(..., map_location='cpu', weights_only=True)`.
- `audit_receipts.py`, `receipt_audit.json`, `receipt_audit.stdout.txt`, `receipt_audit.stderr.txt`: reproducible packet comparison and detailed per-rank structural summaries.
- `attempt1/` and `*attempt1*`: retained earlier code/results/errors described above. The final reference stderr also preserves the CPU library's processor-information parse diagnostic; the test completed with exit code zero.
- `commands.txt`: actual substantive reconstruction/audit/reference commands and exit outcomes, including failed attempts. `artifact_manifest.json`: hashes of generated artifacts except itself and the finalizer's open output streams.

The evidence supports only the precise limited qualification scope stated at the start. It supplies no wider exactness or authorization claim.
