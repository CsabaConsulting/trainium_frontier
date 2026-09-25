# Completed proposal: distinct symmetric K2 tiny consumer

Prepared before code on 2026-09-25. This is a new design, not a retry or adoption of parked d0f0. Its raw compiler refusal and all previous REFUSE receipts remain unchanged.

Both programs execute groups 0 and 1 in that order and load the same group maximum into their runtime loop bound. Program 0 owns columns 0–63; program 1 owns 64–127. No program-dependent branch or row/group offset exists. Each output element has one writer; within that element, additions follow original token order with FP32 rounding. Unused rows preserve positive zero. Final output dtype remains FP32 exactly as in the prior consumer; no new low-precision output cast is introduced.

The input is ONLY synthetic V256/C128/T256 with supplied stable routing metadata validated by the unchanged checker. This does not implement metadata production, general shape handling, training integration or optimization. Both programs repeat metadata reads; no intercore reduction or partial sums occur.

## Contract and evidence

Installed native rank0 reports nki0.5.0+28631259367.ga768afa6. [AWS NKI0.5 release notes](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/release-notes/components/nki.html) explicitly retain the matching SPMD grid requirement; the future removal is not current behavior. [AWS LNC guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.0/nki/get-started/about/lnc.html) documents explicit grid2, program IDs and matching dynamic control schedules. The 2.32 LNC URL is unavailable, so version applicability is cross-checked against the 0.5 release note rather than presumed.

Archived installed torch_neuronx code at research/v2/operator/20260926-trn22-arch-initial-cache-cpu-r1/native/cache-readers/nki_kernel.py.txt lines118–120 documents kernel[grid].dump_config and CompileOptions uses self.lnc at133. This is evidence of framework grid plumbing, NOT a successful grid2 call or full installed frontend source.

[AWS access-pattern documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/deep-dives/nki-aps.html) supports a compile-time offset plus indirect row selection. The planned gradient view uses the unchanged base row stride128, feature width64 and offset program_id*64, avoiding nested-view offset ambiguity.

Raw prior r3 compiler evidence: research/v2/operator/20260926-trn22-k2-namespace-native/completed/results-bundle.json. Rank0 failed NCC_IXGM002 with 9 versus1 basic blocks; other terminal rank receipts are missing after termination. No prior arithmetic pass exists. The asymmetric one-program explanation is a hypothesis, not a demonstrated cause.

## Alternatives and risks

Row sharding would give different trip counts for uneven group maxima, so is rejected. Merely adding grid2 while retaining full-output writes creates duplicate writers and is rejected. Grouped partial sums violate retained cancellation cases and are forbidden.

Runtime-loop lowering, SWDGE with a feature offset, output allocation across programs, compiled custom-autograd integration, FP32 subnormal/signed-zero behavior and actual SBUF allocation are all unproved. Four source-level half-width FP32 tiles total128KiB/program plus small metadata; actual peak is compiler-dependent. Maximum group-trip-count sum is256 for256 logical tokens. Logical gradient/output bytes do not increase, but metadata and transaction overhead can increase. No speed estimate is justified.

## Gates and interpretation

Own stdlib tests must prove complete unique ownership, exact old scalar-reference bytes, retained grouped-sum failures, stable occurrence order and identical control schedules for uneven group maxima. Disabled factory must not import NKI; unchanged checker must remain opt-in.

Future fresh isolated Astra receives full-context kernel diff, neutral spec, exact source/checker/reference and original raw failure evidence—not implementer tests or conclusions. Independent source review, disabled binding review and separate sole Operator assignment precede any native attempt. Required future native result is all four ranks, all20 cases, all150 comparisons per rank; any failure/incompleteness remains REFUSE/HOLD. No native execution, full model, data, evaluator, source adoption or performance acceptance is authorized here.

Undo: leave this separate experimental directory unused. Original d0f0 and all old bindings remain untouched.
