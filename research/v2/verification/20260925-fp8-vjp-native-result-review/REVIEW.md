# Independent native FP8 VJP diagnostic audit

**Diagnostic evidence authenticated; FP8 native correctness remains HOLD.**
All-zero dX/dW in compiled execution is a real backward bug, not a tolerance
dispute. The original component failure is not superseded. Diagnostic exit0
means observations completed, not component or performance acceptance.

## Binding and bounded execution

Raw archive commit `c39844eb`, prefix
`research/v2/operator/20260925-trn21-l1-vjpdiag-r1/completed/`.
Label `fs-20260925-l1-921271-vjpdiag-r1`; four LNC2 ranks, seed42, only256x256x256,
Torch2.11.0+cpu with native Neuron backend. Source commit50a90137 SHA256
`921271d713f53a9d952633ba98a1704c11937235ce974f6b9fcd2cbded4ad858`;
reviewed checker commitfe781c22 SHA256
`d8187ad2f3e68d204203a6252e16062e52521e8b93f119c7f0a9fbe82e4367fd`.
Independent local git-blob hashes match all four rank receipts.

The fixed300-second outer diagnostic,60-second kill grace exited0 in
31.26167626096867 monotonic seconds. All four cases on every rank are OBSERVED,
not caught ERROR. Terminal includes the exact four individual receipts and is
`DIAGNOSTIC_COMPLETE_NOT_COMPONENT_PASS`; each receipt says
`DIAGNOSTIC_ONLY_NOT_COMPONENT_PASS`, `full_component_gate_passed: false`,
`whole_step_measured: false`. Postflight records no active workload/device holder.
All13 retrieved files match their size/hash manifest; no host access/retry or
candidate/checker edits were performed by this reviewer.

## Results — identical on all four ranks

All forward outputs have relative RMS0.005053456407040358 and are exactly equal
to the direct native FP8 forward output. That does not validate backward.

| Execution path | dX relative RMS | dW relative RMS | Nonzero dX/dW entries |
|---|---:|---:|---:|
| Original compiled bound apply, uninspected |1.0|1.0|0 /0|
| Compiled enclosing wrapper, uninspected |1.0|1.0|0 /0|
| Compiled bound apply, inspected replay |1.0|1.0|0 /0|
| Eager apply, inspected |0.004092592280358076|0.0029846138786524534|65536 /65536|
| Manual eager native BF16 VJP using native effective operands |0.004092592280358076|0.0029846138786524534|65536 /65536|
| Ordinary eager native BF16 VJP using CPU-reference operands transferred to device |0.000002118581505783368|0.0000008900373131837114|65536 /65536|

Compiled grads are present and finite, but their RMS, L2 norm, max absolute
value and nonzero count are all0. The **error** maxima4.875/326 equal the
nonzero reference maxima; they are not actual gradient maxima. Thus the previous
zero-gradient signature is now directly observed, not merely inferred.

The supplied upstream BF16 tensor has65536 nonzero entries, L2
256.6750183105469, RMS1.0026403665542603, max4.53125 and zero device-roundtrip
error in every case. This observes the Python-supplied tensor after backward;
it does **not** dump the actual input slot consumed inside the compiled backward.

## Actual saved contexts and casting evidence

Both inspected eager and inspected compiled contexts expose two finite,
nonzero BF16256x256 tensors, with no inspection errors:

- Effective X:65249 nonzero, L2 275.3784484863281, RMS1.0756982564926147,
  max100; relative RMS0.002658416284248233 versus CPU effective-X reference.
- Effective W:65536 nonzero, L2 16.04619598388672, RMS0.06268073618412018,
  max0.2890625; relative RMS0.0038507836870849133 versus CPU effective-W reference.

Eager order is X,W; compiled context order is W,X. Compilers may reorder saved
arguments: this alone is **not a swapped-operands bug**. The backward graph's
input mapping has not been captured. In particular, matching exposed context
tensors does not prove the device backward executable consumed those values.

These are the same nonzero summaries as the independently called native
prepare/project outputs. BF16 saved operands are therefore not visibly zero in
the inspected compiled replay. Saved-W versus unquantized FP32 master has
relative RMS0.025974595919251442; saved-X versus unquantized input0.024319767951965332.
Those comparisons include intentional FP8 quantization and are not STE gradient
acceptance checks. Direct packed-weight error0.004039883613586426, scale error
6.442103739345839e-08, and no reserved legacy exponent are observed. No native
pre-quantization cast tensor or compiled-backward IR was archived.

## Supported inference and remaining uncertainty

1. The failure is not unique to passing bound `.apply` directly to the compiler:
   an enclosing compiled Python frame also produces zero gradients. The wrapper
   matches the inner `Linear.forward` call pattern, but is not a whole-model test.
2. The candidate's eager STE formula and eager native BF16 matmuls produce
   nonzero, within-original1%-RMS gradients for this case. Ordinary eager BF16
   arithmetic and native FP8 forward/operand generation do not explain total
   gradient annihilation. This does not prove a compiled ordinary-BF16
   autograd path: the ordinary control is deliberately manual/eager.
3. Inspection/materialization does not restore the compiled gradients. Visible
   saved-operand zeros are ruled out for the inspected replay, not retroactively
   observed in the first uninspected run.
4. Evidence narrows the fault to the **compiled differentiation / backward
   lowering, dispatch or input-consumption path for this custom-function/NKI
   integration**. It does not distinguish AOT capture, code generation, argument
   mapping, compiled BF16 VJP or runtime execution. No specific SDK or candidate
   source fix is proven, and generic compiler blame would overstate the evidence.

The precise missing evidence for a fix is the captured backward graph and its
consumed tangent/saved-input mapping (or a narrow equivalent probe), followed
by validation that the repaired production-shaped path produces real gradients.
This receipt requests no automatic native retry. The original four-shape1% RMS
gate remains failed; no threshold was changed. A future user-directed
three-step per-parameter cosine>=.99 check is a separate acceptance request,
not grounds to relabel these all-zero gradients acceptable.

If the Supervisor follows the newer instruction to time before fixing gradients,
such OFF/ON measurements must be labeled **broken-backward whole-step
diagnostics**, not component PASS, valid complete-work speedup, quality authority
or proof that a corrected implementation will gain>=3%. Missing backward work
or compiler elimination must not earn the speedup claim.

## Independent local audit

`audit.py` passes110 stdlib-only checks in normal and optimized Python. It
rehashes the13 artifacts, checks exact source/checker/argv and monotonic exit,
matches terminal to four raw receipts, verifies case scope/zero-versus-reference
norm consistency, saved-operand matches, eager controls and all-rank identity.
It does not recompute native tensors from summaries or claim to inspect IR.

Audit SHA256 `79e8b6c49f6c7a3ec41cac9bcc295181d9791f2cd64b7de5cc1e7436eae64c92`;
output SHA256 `ed23996275cce9507b0c533ee726ffc86ae93ed7a5f89f8f152bd9585c14a028`.
The output contains the complete manifest hashes and representative numeric
summaries. Original failure reference remains `c1ac162c`.
