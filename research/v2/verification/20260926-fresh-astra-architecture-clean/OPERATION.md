# Fresh isolated clean-architecture review — exact REFUSE preserved

Actual fresh reviewer verdict: **REFUSE**, finding `EVAL_BATCH_WRAPPER_OMITTED`. This operator is not that reviewer and did not supply previous conclusions, implementer tests/notes, shared branch/history, or follow-up feedback. No source correction, reviewer retry, threshold change, provider-refusal rewrite or Opus/Fable request occurred.

Candidate `bd9e33eee671ee489a1201485faf748bf0a35dea`, source `3f5035da293d984e15df915e2348258aea7f399bbf4d0e47aeb5c9896c35a47f`. Reference exact research3e79; baseline vehicle116188. Input manifest `802a246a3fc0d816e9801c5b73fccb8dc0cb1b8b5e71214246eb5fff33907d92` contains seven input files: two complete source-context diffs, neutral lane specification, and four raw reduced receipts. No checkpoint, repository, prior review, implementer test program, README or implementation note was mounted.

New session `01a0da6d-9512-7d90-9fc8-7088070ac3c2`; requested/CLI-selected model `gpt-6-astra`, high reasoning. No separately echoed provider-resolved model identity is invented. Actual start `2026-09-25T21:16:49.652326Z`, end `2026-09-25T21:24:11.108161Z`, exit0, 441.4558382719988 seconds. The bounded CPU slot was released on completion.

Actual reported usage fields: input1,810,562; cached input1,670,424; cache-write input140,072; output20,540; reasoning-output5,374 tokens. These are event fields, not mutually disjoint billable totals or a cost calculation.

## Isolation and provenance

The already-reviewed runner was reused with only its packet path and manifest pin changed. Fresh empty work/home, ephemeral CLI, ignored user configuration/rules, no session resume/fork; external bubblewrap mounts only read-only packet/system/CPU libraries and writable review output. Original repository/home/SSH and process-environment filesystem are absent. Existing API-key authentication is loaded in memory for the provider-facing process only, never copied/printed/committed; shell tools receive an explicit credential-free environment. Provider networking is allowed; tool networking is forbidden by the lane spec, not falsely claimed to be isolated with a network namespace. Logged commands stayed within packet/work and standard CPU libraries. CLI help/version and [official fresh-session controls](https://learn.chatgpt.com/docs/developer-commands) were checked before launch; the requested [Astra model](https://developers.openai.com/api/docs/models/gpt-6-astra) was preserved.

`session/prompt-input.json` is a no-provider composition diagnostic, not a copied prior conversation. It includes ordinary CLI built-in instructions plus the new prompt. Its debug default permission text is not the actual execution setting: the actual command in launch.json uses externally enforced filesystem isolation with redundant nested CLI sandbox disabled, as in the previously repaired and accepted runner.

## Exact finding and independent evidence

The reference loader constructs `_SubmissionEvalGPT`, invoking base GPT separately for each row when B>1. Candidate loader constructs ordinary `GPT` and processes the full batch. Actual hooks on independently trained state-identical tiny checkpoints witness `[2,32,128]` once for candidate versus `[1,32,128]` twice for reference. Ten route comparisons differ across five context lengths and two CPU dtype settings; every corresponding CPU logit comparison has max absolute and relative error0. The reviewer explicitly does **not** claim measured CPU/native numerical failure. Its REFUSE is for not preserving the specified reference's effective evaluation route. See the verbatim report and decision, rather than treating hook counts as failed logits.

The fresh reviewer independently wrote and ran one CPU exactness program, exit0, one thread, no distributed workers: 596 passing numerical/diagnostic records, including all65 initialized tensors, three tiny gradient/update steps, actual optimizer arithmetic, isolated256-token custom MLP backward, tiny checkpoint/eval/casual-prefix comparisons, schedule/ownership probes and two negative controls. Positive comparisons use atol=rtol=0; not all596 records are numerical comparisons. The detached-cache control exposes missing master gradients despite identical forward outputs. The FIFO-versus-LIFO skip control produces max absolute0.016597986221313477 and relative56.75460122699386. Inputs, outputs, states, gradients, checkpoints and raw stderr are preserved.

Own models are width128/depth8/vocab256/context<=32/batch<=2, one live training model at a time. Tracked peak live tensor storage43,590,044 bytes; conservative retained+workspace52,968,792 bytes, each below64MiB. Torch2.13CPU, no native/compiler/collective/real data. Test-only prepare/pyarrow stubs and temporary nonpersistent eval table floor32 are disclosed. Imported startup placeholders were2steps/600s; they were **not** used as validation of the official5-step/900s contract. Full113M count and ownership used static shape objects, not full tensor allocation.

## Gate interpretation — no verdict substitution

The decision's literal scope is “Source and bounded reduced-CPU suitability for next qualification only,” and its decision is REFUSE. Required resolution in the report is preservation of reference row-wrapper behavior followed by loader verification. It does **not** positively approve the current file for full113M CPU or a native checkpoint diagnostic. `launch_authorized:false`, `submission_authorized:false`, `native_executed:false`, `full_size_cpu_executed:false`, `candidate_quality:unmeasured` are explicit. Later fullCPU/native/cold/quality gates being absent are not its reason for refusing. Any separate investigation of whether ordinary-GPT batch behavior meets the competition contract must be a new explicitly scoped diagnostic decision; it must not relabel or conceal this fresh REFUSE. The mechanical e96 CPU binding review is separately archived in commit08e3dda5 and does not supersede this source decision.

## Durable archive

All89 archive files,108,392,147 bytes, were copied verbatim after exit. Archive manifest SHA256 `7b4bc5724fb9820d3f08b6db841c3fd6974da5e1a3e3992985fab6d413a38e4d`. All71 files listed in the reviewer's own manifest also authenticate. Supplementary operator provenance/validation files are outside that earlier immutable manifest. `validate_archive.py` passes normally and under -O without tensors.

Key SHA256s:

- Report: `5ec4839f88e831c48ae6784bd629132d9ca6c89d6aaa084b25b6688cb6857dd0`.
- Decision: `14a1ab1974d17e8cee29b14ee3b314fb3fdf6794e4232f9a6e69e610950d4c83`.
- Own test: `b60699847859d4b5f52cb27f9e59af6f48483dc012fc3d4011684000e51c338a`.
- Test results: `4a81055a2a378b03c353d96c63f8922ccabb4d0869db81e12fd363c185ea4295`.
- Structural source audit: `55342b58cb9af99233768bc13bdd89b4e55a3813c2afa696a3144b519aa03666`.

No host, source edit, ledger mutation, native execution or submission was performed by this operation.
