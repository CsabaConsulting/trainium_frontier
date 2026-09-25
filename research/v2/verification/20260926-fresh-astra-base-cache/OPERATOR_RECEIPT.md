# Fresh isolated Astra review: exact basebcbd CPU-qualification scope

Outcome from the new reviewer: **PASS_LIMITED_CPU_QUALIFICATION_SUITABILITY**, assigned BF16 compute / FP32 masters only. Full117M standard G2 is absent; native execution/scientific adoption remains **REFUSE**. This operator did not supply a source verdict, reference test or implementation/review notes to the reviewer and does not substitute its own technical review for this result.

## Actual independent session

- Session: `01a0da4a-5cd6-78e2-b0dc-33267b8ab74b`.
- CLI-selected/requested model: `gpt-6-astra`, reasoning high. The event format does not separately echo a provider-resolved model identifier; no different model was selected or retried.
- Started `2026-09-25T20:38:21.489014+00:00`, finished `2026-09-25T20:44:20.868969+00:00`, exit0,359.3799513349986s. The dates are UTC; local task labels use26September IST.
- A new CLI exec was created outside every repository at `/tmp/trainium-fresh-base-cache-review-20260926.O9in3e`. No fork, resume, inherited conversation, shared branch or previous session was used. No Opus/Fable call or provider-refusal retry occurred.
- Only the prepared full-context exact diff, neutral lane specification, four raw reduced CPU receipts and input manifest were mounted read-only at `/packet`. Fresh `/work` was writable. No implementation notes/tests, earlier technical review, repository/history, user home, SSH files, auth files or process environment filesystem were mounted.
- Existing reviewed bubblewrap containment was reused. Provider credential was loaded in memory solely for the CLI process; generated commands received an explicit credential-free environment, one-thread CPU settings and fresh HOME/CODEX_HOME. No credential value/file was copied into the packet, outputs or repository. Outer filesystem isolation remained active while redundant nested CLI sandboxing was disabled, as previously authorized.
- `--ephemeral --ignore-user-config --ignore-rules --skip-git-repo-check`, fresh home and disabled external integrations were used. Filesystem/environment preflight passed. Built-in generic CLI instructions are present; no user/implementer context was passed. Actual logged commands stayed within `/packet`, `/work` and ordinary CPU tools/libraries; no host, network, native, full-model or training launch occurred.
- CLI0.155.0-alpha.16.4 help and [official command documentation](https://learn.chatgpt.com/docs/developer-commands?surface=cli) were checked for the fresh-session controls. The OpenAI Docs skill informed this control check; no new control plane was built.

Input manifest SHA256: `56172b63b560d7a437a2fbf601b175be7b872e2e2e265adce7302daa03956fae`. Six payload files,5960489bytes. It is unchanged from the prepared packet and reconstructs base6ef7d090 and candidatebcbd148d exactly. Its immutable `reviewer_session_started:false` describes preparation time, not the later session; the actual launch/terminal/events here are the authoritative session evidence.

The reviewer's `fresh_session_policy=NOT_SEPARATELY_CERTIFIED` correctly means its numerical test did not self-certify a separate policy document that was not supplied. It is **not** an additional source defect or demand for full117M evidence before CPU qualification. This operator's launch/preflight/input/access records establish the agreed fresh-session conditions; the source reviewer separately supplies its own limited source/test verdict. Supervisor retains policy/assignment authority.

## Actual reviewer-owned test

The reviewer independently reconstructed both sources, removed only the two added AST statements to verify all other source code unchanged, audited raw receipts, and wrote/executed `reference_test.py`. Eight tiny fixtures ran four updates and three sequential backwards per arm across base/default/OFF/ON/direct-oracle paths. It used12or64master elements and at most256x4activations, no full model or distributed process. Intra/inter-op threads were1.

All **3183 bitwise tensor comparisons** passed, with zero nonexact comparisons and maximum absolute difference0 throughout. Tests include ordinary and transposed Linear, eager MLP, actual extracted custom MLP backward, cache-disabled mode, AdamW and source-extracted pure Muon state updates, FP32 master/input gradients, repeated updates, connectivity, and fresh strict weights-only loads. Its own hooks directly observed96custom-backward executions across the adapted cases. The custom adapter's CPU `native` forward name is ordinary PyTorch reference execution here, not a Trainium claim.

The reviewer retained and explained its two exploratory fixture failures: an audit initially assumed checkpoint fields existed on every rank; a broader FP32-compute probe exposed a pre-existing same-dtype Parameter alias outside the assigned BF16 lane. Both attempts and their raw output are preserved. It also recorded pre-existing hot-load stale-cache behavior in both sources, plus detached-cache and dtype-mismatch controls. None was hidden, corrected in production source, or converted into a wider exactness claim.

After completion, this operator checked all30entries in the reviewer's artifact manifest, all3183comparison records (exact and zero-error), eight result cases and the actual successful test/finalizer command exits. The reviewer's raw tensors were independently weights-only loaded by its finalizer before exit. This operator did not re-run tensors or alter its report. The CPU slot was released after the session exited.

Support hashes:

| Artifact | SHA256 |
| --- | --- |
| Reviewer test | `8c50dcf84dcbd11c4bd5a8115f4e06eb9642ace05c62e81b63daaeb5dcea8f9c` |
| Reviewer numeric result | `fe8321386562dfa2dce11b10aba11dd7ea7a770b1683a54f30c4e783451718d0` |
| Reviewer report | `fa1d2d27c24f81542c7697ab27d6c3813b2b09e6a575f9637e88fb65f5627cb7` |
| Reviewer decision | `0528eb4184ab01a6d1037af3e10382375219b7f287c19de15b9124a033e532cd` |
| Verbatim archive manifest | `e926638a3c28eb684e68a802801f9acc5ca8fdf12004ff23a29f7df0ab68c199` |

Actual CLI usage, not a cost estimate: input1128795, cached input1044645, cache-write input84090, output20383, reasoning output4242tokens. These are reported usage fields, not distinct totals to add together. Exact events and terminal evidence are preserved under `session/`.

## Remaining gates

This evidence supports the justified next **exact full-size CPU qualification**, subject to the separately reviewed CPU binding and sole Operator's assignment/preflight. It does not claim full117M G2, native compiler correctness, cold startup, performance, budget compliance, quality, adoption or submission. No host/source/ledger changes were made by this review operation. The separate cold-five mechanical review is commit `189e9d7f5850b345c1f494bee446e88d8d4750a0`; it remains distinct from these numerical/source findings and confers no launch authority.
