# K2 numerical-domain disposition (Researcher only)

2026-09-26. HOLD correctness; no native timing/full release. Existing failures
are immutable. This note neither replays nor qualifies the failed packet.

## Evidence and inference

`operator/20260926-trn21-ordinary-subnormal/completed/ordinary-control-summary.json`
records all four ranks returning zero gradient versus 1,408 expected nonzeros
for both ordinary eager and compiled embedding. Maximum absolute discrepancy
is 1.2611686178923354e-44. Input and forward identity, normal and sentinel
controls pass. The diagnostic exited 1; `analysis/20260926-ordinary-subnormal-result/`
preserves independent archive checks. Earlier registered-candidate observations
are in `operator/20260926-trn22-k2-registered/`; its required ordinary control
failed, so candidate observations cannot turn its aggregate result into PASS.

This localizes a mismatch shared by ordinary native routes. It rules out the
registered candidate or torch.compile being necessary causes of *this control
failure*. It does not identify the exact failing primitive, prove a universal
Trainium FTZ policy, or establish candidate correctness over all inputs.
Shared backend lowering, arithmetic underflow handling, a conversion boundary,
and operator implementation remain distinguishable hypotheses.

## Four different contracts

1. Real-number scatter sums have no rounding. They are not the test oracle.
2. `proposals/20260926-k2-registered-native-trn22/reference.py::dense` models
   an ordered binary32 addition after each occurrence using explicit pack/unpack.
   It preserves gradual underflow. An alternative reduction tree is not
   necessarily identical, even with FP32 accumulation. Current receipts use
   this specific oracle, not every possible mathematically equivalent sum.
3. Deployed BF16-output / FP32-weight-gradient paths must first represent the
   upstream tangent in BF16 and then accumulate in FP32. The FP32 fixture
   magnitude 2^-149 rounds to zero in BF16; its BF16-output variant therefore
   cannot test nonzero FP32-subnormal transport. BF16 minimum subnormal 2^-133
   is nonzero in both types, but is still subnormal in FP32. A BF16 pass alone
   cannot prove the unrestricted FP32-input contract. Actual production cast
   placement must be established from the frozen candidate, not assumed from
   the phrase 'BF16 training'. FP32 normal operands can also cancel to subnormal
   results, so excluding tiny inputs does not establish closure of a domain.
4. `adapter.py` registers a forward embedding and routes backward explicitly
   to the consumer. `consumer_nki.py` gathers into FP32 SBUF and performs one
   Vector Engine `tensor_tensor(add)` per occurrence before returning FP32.
   This differs from ordinary embedding lowering; exact observed candidate
   gradients and failing ordinary gradients can coexist without contradiction.

Primary AWS documentation describes FP32 internal arithmetic for relevant NKI
operations, but that statement alone does not promise gradual underflow:
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_modules/nki/isa.html
and https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/api/generated/nki.isa.tensor_reduce.html
(the latter is a reduction example, NOT the candidate's primitive).
An official-domain search for denormal/subnormal/flush-to-zero did not locate
a primary guarantee for this ordinary native embedding path. Therefore
**documented native FTZ is unestablished**, not a conclusion of this note.
Versioned installed implementation and compiler flags still matter.

## One prospective proposal: CPU-only boundary/provenance audit

Status: proposed, non-dispatchable; no new host run requested. Separate
Implementer and Verifier should audit the exact frozen ordinary/native packet
and production embedding cast/autograd chain, without changing either. Produce
a small independent CPU table of input bits, output-tangent bits, ordered FP32
intermediates and final bits at 2^-149, 2^-133, the normal boundary, and normal
cancellation to subnormal. Include duplicate ordering and both signs. Compare
archived candidate and ordinary results case-by-case; do not allocate training
tensors or reconstruct a training run. Check whether any claimed BF16
subnormal success is actually a cast-to-zero fixture. Preserve the explicit
FP32 oracle and every failed gate. This cheap test can falsify a mistaken
dtype-domain explanation without host occupancy.

If source/receipt inspection cannot locate the loss boundary, leave HOLD and
request an exact installed-version backend explanation. A later uniquely
proposed primitive diagnostic could distinguish transfer, cast, add and scatter
boundaries, but is NOT authorized or packaged here. Repeating the unchanged
ordinary embedding test adds no information. Replacing the CPU oracle with
observed zeros, deleting the ordinary control, loosening absolute tolerances,
or selecting only successful cases would weaken correctness and is rejected.
Resolving a domain mismatch scientifically does not itself resolve the required
gate: a separately reviewed complete contract and valid gate remain necessary.

## Workspace incident

An attempted full isolated checkout failed while writing tracked artifacts;
Git removed the incomplete checkout. No host action or artifact deletion was
performed. A second `--no-checkout` isolated branch holds only this note.
This avoided repeating the checkout and created no new numerical evidence.
