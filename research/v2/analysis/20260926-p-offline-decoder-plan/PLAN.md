# P rank0 offline decoder: disabled bounded plan

Status: PLAN ONLY. No decoder, host command, large-file transfer/parse, device capture, training, or provider request was performed. Supervisor assignment and Operator preflight remain necessary. This is not a new launcher.

## Exact existing evidence

Capability archive: canonical commit `cd298eba`, directory `research/v2/operator/20260926-trn21-profile-recursive6ef7/offline-cli-capability/`.

| Metadata file | SHA256 |
| --- | --- |
| neuron-explorer-version-and-prefix.json | 2a5c2ac045ea2be595f3ec77f3fcf1012dc76ff1c38ab18a41651836cd0daa80 |
| neuron-explorer-view-help-complete-normalized.json | e35c5342947390367013768b477c2a54d9035bbf039425e69d463f21564c2c12 |
| neuron-profile-and-inventory.json | 25b5b94bca799741cfc7e8eed35cc37cb578074aef1d338d55f1d3a352274454 |

Installed explorer is `2.32.0.498-b1f3998`. `neuron-profile` is a deprecated, failing migration shim, not the decoder to use. Help success is only option evidence, not an ingestion test.

The original root is `/home/ubuntu/trainium_profile_recursive6ef7_20260926_trn21_r1/profile/rank0/native/`. Its session subdirectory is `i-0329acb9a6244986b_pid_3008377/1790365944906458876/`, containing:

- `profile_nc_0_session_0.ntff`: 4,130,966,280 bytes;
- `ntrace.pb`: 1,529,888 bytes; `trace_info.pb`: 5,929 bytes;
- `trace.json`: 10,181,471 bytes; zero-byte `cpu_util.pb` and `host_mem.pb`.

The sibling `native/neffs/` tree contains **80 NEFFs, 167,863,163 bytes**, mostly in hash-prefix subdirectories, plus a barrier NEFF. These are inventory sizes, not newly authenticated content hashes. Future assembly must record source/destination hashes; do not claim existing metadata authenticates all 4.13 GB of NTFF content.

## Correct multi-model input

Use the **whole rank0 native tree**, not the timestamp leaf alone and not an arbitrary `--neff-path` with the NTFF. AWS 2.32 documents the process/timestamp session plus sibling `neffs/` layout and automatic NTFF-to-NEFF matching. This establishes the appropriate directory-mode mechanism; it does not prove that the installed binary recursively discovers this particular nested hash layout. [AWS 2.32 profiling layout and matching](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/tools/tutorials/performance-profiling-vllm.html#step-4-view-profiles-in-neuron-explorer).

Assemble a new uniquely labelled scratch input tree preserving the original relative layout, using independent copies or copy-on-write reflinks, **never hardlinks, moves, renaming originals, or in-place writes**. Include all 80 NEFFs and all six session files. Copy/reflink bytes and their hashing count against the same bounded diagnostic slot. A reflink fallback may physically copy approximately 4.31 GB; require 12 GiB free scratch space first. Make assembled input read-only. No symlink-following assumption and no undocumented flat-name rewrite. On missing/ambiguous model matching, stop and report the exact diagnostic; no automatic arbitrary pairing or second decoder attempt.

AWS directory upload requires the system metadata and can include device profiles; standalone device-only matching is a different mode. JSON export can expand into a file per model, so it is intentionally not the first attempt. [AWS 2.32 input and export modes](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/tools/neuron-explorer/get-started.html#step-5-upload-your-profile).

## One prospective attempt

After a separate explicit assignment, Operator alone reserves a no-other-work host slot: no training/evaluation, full-model CPU qualification, compilation, or second decoder. Hold the existing common `/home/ubuntu/.cache/trainium_freshstart/device0.lock` for assembly, decode, and cleanup/quiescence; fresh `ps` and `neuron-ls` checks under that live lock both fail closed. This plan grants no staging/launch authority and does not interrupt current work.

Core argv, with `SCRATCH_INPUT` resolved to the new assembled tree:

```text
neuron-explorer view --session-dir SCRATCH_INPUT --output-format summary-json --ignore-nc-buf-usage --ignore-instruction-hierarchy
```

Installed help explicitly says summary-json prints to console without a database write. Capture stdout rather than assume `--output-file` applies to this format. Do not add `--data-path`, `--force`, `--ingest-only`, UI/port options, or a database service. Keep instruction/event/DMA traces enabled: skipping those would defeat device attribution. The selected omissions remove buffer-fullness and hierarchy analysis, so neither may subsequently be inferred from this output.

Proposed ceilings for this **single diagnostic**, including copy/hash work: 600 s overall wall, decoder at most 300 s wall plus 10 s termination grace, 300 CPU-seconds, one allowed CPU and `GOMAXPROCS=1`, hard 16 GiB virtual-address-space limit, 64 MiB per-file write limit, 16 MiB combined retained stdout/stderr (stop the process group at this limit), and 256 MiB scratch *generated-output* stop threshold excluding the authenticated read-only inputs. The last threshold is a sampled stop condition, not a filesystem quota; record sampling interval/overshoot. Require at least 24 GiB available memory and 12 GiB free disk; stop instead of reducing these margins. A virtual-address limit is not a cgroup total-memory guarantee and may cause an immediate Go/runtime failure; preserve that as a bounded diagnostic failure, not permission to raise limits. Limits must be enforced in the eventual reviewed binding, not merely written in a receipt. If existing machinery cannot enforce them, leave this plan disabled rather than improvise a new controller.

No automatic retry, larger parse, JSON/Parquet expansion, installation, device recapture, or deletion follows failure. Preserve originals and the bounded scratch/log artifacts; no protected-artifact cleanup is authorized.

## Required small receipt and interpretation

Record exact binary/version, source-to-scratch inventory/hashes, argv, resource limits, common-lock inode, both idle checks, start/end, exit/signal, CPU/RSS high-water marks, bytes written, bounded stdout/stderr, and post-exit quiescence. The summary must identify models actually decoded and unmatched/missing/ambiguous NEFFs. Merely supplying 80 files is not evidence that all 80 were used or that every observed graph was mapped. If summary-json lacks mapping coverage, retain `coverage_unknown`; do not call attribution complete or automatically expand output.

Only fields explicitly emitted as decoded device measurements support engine/instruction attribution. Prior runtime graph intervals and host phase annotations are not those measurements. This rank0 trace has known capture skew; a successful decode is not matched whole-step throughput, all-rank timing, a K1/K2 speedup, internal matmul/SDPA dtype proof, or an MFU estimate. Buffer fullness and instruction hierarchy remain intentionally absent. Decoder success would guide the next hypothesis, not authorize a kernel or quality adoption.
