# Independent scoped P r2 mechanics review

**PASS for the narrow readiness-limit repair and its frozen binding only.** No blocking defect found in99d7c261316a6597d8324c7804912365e67ce02b after independent source-scope inspection and own tests. This is not installed Go/decoder compatibility, ingestion/coverage proof, source adoption, or host launch authorization.

Proposal-first7519c517 is byte-identical to the bound proposal. New r2 root/label are distinct from r1; source profile root and all86 input paths/sizes (all80 NEFFs;4,310,546,731bytes), decoder options, binary/version, shared guard/dispatcher, support helper and every resource ceiling are unchanged. The remaining complete driver AST equals b529468f after root/proposal-pin substitution and exactly two wrapper substitutions; only readiness_guard is added.

Pins:
- Driver616fc37c756a5be2cc04f8fb4c13e551a8e0bae4e07e2eeda00475fd1db5e1df.
- Binding3c08ce4b48d98596bd8c8594fc2f9f8fcb92e41669779cd93732de335ce856cb.
- Manifest4e6bc2f3264d9d481713fe88ec8d2ce1e1ecd221f3370ddb9286bed14b6a40d7.
- Unchanged containment helper96aebc1e9a43843f57691e67053c75ecbe904f2cec6899688a9e3036c863efd9.

## Own tests and failure paths

audit.py loads exact immutable Git blobs and does not import/run the candidate main or native binaries. **116 own checks pass normal/-O**, including all packet manifest/runtime pins, proposal chronology, complete inherited AST identity, every input/limit/command identity and the11-file original failure archive's bytes/hashes. Eight independently written mocked paths cover success, original guard exception, BaseException, receipt failure, failed elevation, raised restoration error, silent restoration failure/readback mismatch and unexpected initial limits. No failing path returns success; no false restored receipt is emitted after failed restoration. Original guard exception types/messages propagate when restoration/receipt succeed.

Separate short local Python subprocesses also exercise **real Linux RLIMIT_AS inheritance**, not only mocks: before=(512MiB,16GiB); the wrapper's child observes(16GiB,16GiB); parent restores(512MiB,16GiB) on both success and an injected guard rejection. The same cases pass under -O. No large allocation, tensors, Go, SDK, device, remote host or trace bytes are involved. Supplied82 checks were separately replayed in each mode.

The driver's existing outer BaseException/finally handling remains byte/AST-equivalent: initial wrapper failure leaves a refusing exit and does not progress to version/copy/decode; post-wrapper failure resets status/exit to refusal. The same FD check, original lifetime lock and external explicit quiescence obligation remain. No guard result is cached, suppressed or bypassed.

## Original failure and limits

Original b29da2cf shows locked ps succeeded, neuron-ls returned2 with Go page-summary memory-reservation failure after inherited512MiB soft AS, and version/copy/decode outputs did not start. The exact terminal/refusal/postflight copies remain. This supports the narrow inheritance diagnosis; it does not prove16GiB will be sufficient for installed Go.

The readiness wrapper temporarily raises only the parent's soft AS to its existing hard16GiB ceiling during the bounded pre/post guard calls, then restores512MiB before normal parent copying/hashing/decoding. CPU, wall, disk/RAM floor, file/output/scratch limits and decoder hard AS do not increase. The temporary parent soft-limit exception is explicitly declared, not described as unchanged parent AS throughout. RLIMIT_AS is virtual-address-space per process, not RSS/cgroup containment.

Another Go reservation failure, restoration failure, incomplete cleanup or unsupported trace layout remains REFUSED_OR_INCOMPLETE. No automatic larger limit, third attempt or fallback decoder. Installed Go16GiB compatibility, decoder ingestion and all-model matching coverage remain **UNPROVEN**. Independent Supervisor acceptance and a new sole Operator assignment after predecessor quiescence remain required.

## Independence and review scope

I authored the original decoder/helper but **did not author this r2 patch**. This review independently tests the new wrapper and exact unchanged scope; it does not replace the separate prior containment review22233aea or purport to freshly approve my own original helper. No candidate/support/source/profile/cache/checkpoint or host was changed.

The first reviewer test assumed the original failure bundle used text fields; it actually uses base64. audit-preparation-attempt1.json preserves that reviewer-only fixture error. The fixture now decodes validated base64 and rehashes all11 files; no candidate change was requested. Undo: leave the disabled packet unassigned.
