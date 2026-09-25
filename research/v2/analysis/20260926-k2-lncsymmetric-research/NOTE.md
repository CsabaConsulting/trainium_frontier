# K2 after the parked consumer: a distinct multicore design hypothesis

Read-only research, 2026-09-25 23:20 UTC. No patch, SDK installation, native
retry, source acceptance or host reservation. Current resident-consumer r3
remains PARKED_COMPILER_FAILURE. This note does not re-arm it.

## Observed local facts

Frozen d0f0d7fb consumer4939265b uses two row groups and a runtime loop bound
loaded from each group's metadata. Its checker89189024 invokes the JIT kernel
without an explicit multicore launch grid. The actual compiler error reports
9 basic blocks on one core versus1 on core1; arithmetic was never validated.
These facts support investigating multicore lowering, not a proven root cause.

## Primary-source constraints

AWS documents that explicit `kernel[2]` selects two programs, while a bare
call defaults to one, and warns that dynamic control flow must agree between
cores. [AWS LNC guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.0/nki/get-started/about/lnc.html).
The same control-flow constraint appears in the [2.32 NKI release notes](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/release-notes/components/nki.html).
The 2.32 LNC-guide URL could not be retrieved. Installed framework/JIT behavior
must be checked against these versioned claims before a new native proposal.

Follow-up: authenticated r3 rank0 receipt reports installed
`nki=0.5.0+28631259367.ga768afa6` and `torch=2.11.0+cpu` (native extension).
The same release-notes page explicitly says the matching SPMD launch-grid
requirement still applies in NKI0.5.0; its removal is a future deprecation,
not an already-shipped behavior change. This narrows the version mismatch
and strengthens the design question, but still does not prove compiler cause.

Inference, not diagnosis: a one-program custom kernel inside LNC2 may leave
one core without the other's dynamic-loop blocks. The raw failure alone does
not distinguish that from another compiler defect. Do not change global LNC,
world size, compiler version or production source on this inference.

## Candidate for a later independently reviewed proposal

Use an explicitly symmetric two-program kernel: both programs process the
same row groups, counts, token order and runtime trip count, but own disjoint
feature columns. No core-specific condition may bypass a dynamic loop; every
output element has exactly one writer. FP32 additions must preserve token
order, including cancellation, empty rows and repeated tokens. This is a
different multicore ownership design, not another relabelled namespace retry.

Before any host slot: prove with an independent small CPU reference that
feature sharding preserves every output byte and that both programs take
identical loop counts for adversarial metadata (including unequal row-group
maxima). Verify the installed JIT launch contract from archived code/docs.
Then require fresh input-only source review and a new completed proposal.
No transfer, metadata-building or whole-step speed claim follows from this
consumer-only design; those costs still need measuring after correctness.

## Upstream scatter is not currently a safe shortcut

AWS's published scatter API still requires unique indices within each128-row
tile. Ordinary embedding gradients violate this precondition.
[AWS scatter API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/library/api/scatter-add.html).

In [upstream PR11](https://github.com/aws-neuron/nki-library/pull/11), a maintainer
reports reproducing the duplicate-index bug and replacing the proposed
one-row workaround with deduplication, citing a large regression for the
original workaround and saying the replacement was merged. These are upstream
reports, not our measurements or proof that available bytes include that fix.

The directly retrieved public main commit was92d11f63a9a8ec1ade34e6e1a3b8db66ef31307e;
its scatter source SHA is67c9422f2e05a19ab91da81b15f290b114e2b9d263f21c0af079931867135237.
It still has the uniqueness precondition and tile-wide gather/add/store, with
no duplicate-safe branch. [Exact inspected source](https://github.com/aws-neuron/nki-library/blob/92d11f63a9a8ec1ade34e6e1a3b8db66ef31307e/src/nkilib_src/nkilib/experimental/misc/scatter_add.py).
This mismatch blocks a blind upstream import. Locate actual corrected bytes
and verify duplicate semantics, accumulation order, installed-SDK support and
Apache2 attribution before considering reuse. No source was copied into the
training file and no package was installed during this research.

Priority: finish active architecture/schedule scores, cold-clock qualification
and saved-profile decode. This note offers a supported design question for the
next available implementation lane, not a reason to interrupt either host.
