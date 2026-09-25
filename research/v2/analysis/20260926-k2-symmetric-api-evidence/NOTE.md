# Symmetric K2: version-matched API documentation evidence

Read-only research, 2026-09-25 23:45–23:55 UTC (26 September IST). **This addendum does not change the fresh source REFUSE, accept native correctness, or authorize a launch.** No host, SDK installation, tensor execution, provider call or production/source change occurred. Mechanical review `f4aeca7111aa192bfb886fbc7748fdafc68fbce5` remains a separate, disabled-binding disposition.

## Version and provenance

The old r3 rank0 raw receipt records `nki=0.5.0+28631259367.ga768afa6`. Its byte-exact copy at candidate9ce2e6a0, `research/v2/proposals/20260926-k2-symmetric-consumer/review-input/original-native/rank0.json`, has SHA256 `16d811e9c72df549fc1ce6726347d834f8c6291f740bba170327f8e827b327ad`. This is an archived trn22 observation, **not a new observation of the future trn21 environment**.

AWS's [version2.31.0 release notes](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/release-notes/components/nki.html) identify NKI0.5.0, released7 July2026. They retain the matching SPMD-grid/LNC-degree requirement; the announced removal is future behavior. Thus version2.31.0 is the direct documentation match, not an inference that the installed package implements every2.32/NKI0.6 behavior. The prior note's2.32 release-history citation remains useful corroboration, but is not needed as the API authority here.

`SOURCES.json` contains11 official, versioned URLs, section/parsed-line locators and short exact rendered excerpts. `EXCERPT_HASHES.json` authenticates those UTF-8 strings, **not entire HTML pages**. Markup is removed as described; no publication date was invented for individual API pages. Versioned documentation can still be corrected later. `MANIFEST.json` pins the archived local bytes, not a live website or installed wheel. Failed guessed `.ap` URL was replaced by the actual `NkiTensor.html` page; no failed retrieval is used as evidence of API absence.

## Factual evidence matrix

| Candidate use | Version-matched documented behavior | Remaining distinction |
| --- | --- | --- |
| Lines25–26,75: grid2, no-argument count, core ID | The [LNC guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/get-started/about/lnc.html) gives a two-program bracket launch, count2 and IDs0/1. It requires shared inputs/outputs and identical basic control flow; divergent dynamic control flow is undefined. The [count API](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/api/generated/nki.language.num_programs.html) documents its default, and [ID API](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/api/generated/nki.language.program_id.html) axis0. | This documents the requested launch/ID contract. It does not prove the installed compiler fixes NCC_IXGM002 for this new source or that two compiled cores follow identical paths. |
| Lines49–58: `.ap` pattern, ordinary `offset`, vector index, `indirect_dim=0` | [NkiTensor.ap](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/api/generated/nki.language.NkiTensor.html) defines stride/count pairs, an additional element offset, per-partition vector indices, and scaling by the selected original tensor dimension's stride. [DMA](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/api/generated/nki.isa.dma_copy.html) explicitly permits HBM vector indirection using a `(P,1)` SBUF index and includes the same row-stride/free-stride pattern. | Applying those definitions to contiguous `(256,128)` gives the inferred address `128*token + 64*program_id + feature`, for feature0..63. The first gather similarly addresses `positions[pointer]`. This is a derived mapping, not an executed native address trace. No exact combined program-ID-offset/vector-gather example was found in these bounded pages. |
| Lines50–58: software DGE, OOB skip, BF16 input to FP32 tile | The [DMA API](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/api/generated/nki.isa.dma_copy.html) supports SBUF-indexed gathers under software DGE, skips OOB transfers when requested, and converts via FP32. [Supported types](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/api/nki.api.shared.html) include BF16, FP32 and INT32. | Retention of initialized destination lanes is the consequence of skipping their writes, not a new native result. Sentinel256 with the combined offset, index typing, signed zero/subnormal behavior and exact cast/add/select bytes still require the unchanged native checker. |
| Lines27,72: shared output, disjoint feature stores | [shared_hbm](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/api/generated/nki.language.shared_hbm.html) is visible across grid instances. The LNC guide demonstrates program-ID-specific stores to one shared output. | Source ownership is disjoint under the ID contract:0..63 versus64..127. Documentation does not certify this compiled allocation/store result or any speed benefit. |
| Lines35–42: loaded group maximum, runtime loop | [register_load](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/api/generated/nki.isa.register_load.html) accepts a one-element HBM/SBUF value up to32bits; [dynamic_range](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/api/generated/nki.language.dynamic_range.html) accepts virtual-register bounds and positive compile-time step. | Candidate `(1,1)` INT32 maxima and default step1 fit those documented inputs. Both programs read the same group maxima; actual symmetric lowering, allocation and termination remain unexecuted. |

Important terminology: the candidate uses ordinary `.ap(offset=...)` together with `vector_offset`, **not** `scalar_offset` plus `vector_offset`. The DMA documentation forbids the latter pair, not the former. NKI0.5's newer on-chip tensor-indirection API is a different, Trn3-only feature; it is not being used to justify Trn2 HBM DMA gathers.

The [published API source view](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/_modules/nki/language.html) corroborates the signatures/docstrings but has ellipsis bodies. It is not the installed wheel implementation and cannot fill that gap. The documentation default DMA engine spelling differs between2.31 and2.32; no newer default was projected onto this candidate's installed environment.

## Exact support and unchanged limits

`support/` contains only exact files from `9ce2e6a037212aaa76bd85b9b4ab7d311a481757`, with no Implementer tests, notes or previous reviewer conclusions:

- `consumer_nki.py`: `2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae`.
- `check_native.py`: `89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850`.
- `reference.py`: `14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f`.

The latter two are byte-identical to the already-bound oracle/checker; these archive copies were not executed or edited. They let a later assessor inspect the full support that was absent from the initial compact input. This research note is an interpretation, not a neutral replacement for the primary evidence or an instruction to a fresh reviewer.

The concrete public-documentation gaps have narrowed: the APIs and their requested parameters are documented for NKI0.5. The remaining evidence gap is compiled/native realization of their combination on the exact wheel/compiler/hardware, including OOB+offset, FP32 recurrence rounding and all output bytes. There is no new demonstrated source defect and no claim that the old compiler failure's cause is proved. Original077, d0f0/r3 and symmetric source refusals remain preserved. All four native ranks/all20 cases, unchanged byte-exact finite acceptance, failure/descendant cleanup and separate source/Operator release remain mandatory. No routing, full-autograd, whole-step performance, BPB, adoption or submission claim follows.

Local archive verification only:

```text
python3 -I -B research/v2/analysis/20260926-k2-symmetric-api-evidence/audit.py
python3 -I -B -O research/v2/analysis/20260926-k2-symmetric-api-evidence/audit.py
```
