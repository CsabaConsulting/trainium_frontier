# Fresh Astra architecture review operation

This is an **operation/archive receipt, not the operator's source review**. The
fresh reviewer returned **PASS for exact source/CPU scope only**. No native
startup, performance, quality, full-run or submission permission follows from it.
The separate cold fifth-boundary startup gate `<= 900 s` remains pending.

## Identity and isolation

- Original source: `93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5`.
- Candidate source: `3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c`
  from supplied commit `b4d83b5db66971f3c348837f04f53cba65aa6d33`.
- Input manifest: `96a204756ce2f465f6cf9dbee24dab4997703107a9c5494696faa933cb4354c5`.
- CLI `0.155.0-alpha.16.4`; explicitly selected model `gpt-6-astra`. The JSON
  stream does not independently echo the provider's model identity. This is a
  fresh **same-family substitute**, not a claimed cross-family review.
- Successful fresh session: `01a0d9fa-c5f1-7522-b5ed-4b7905cc1000`.
  Start `2026-09-25T19:11:25.516694+00:00`; completion
  `2026-09-25T19:17:25.677664+00:00`; elapsed `360.16096499299965 s`; exit `0`.
- Reviewer received only the neutral lane specification, the complete exact
  source diff and raw reference/OFF/ON G2 records. No existing implementation
  tests, summaries, review conclusions, branch, conversation or CT research were
  shared. CLI built-in instructions are not inherited implementation context.
- A fresh non-repository directory and fresh ephemeral CLI session were used;
  no resume/fork, daemon, user configuration, repository rules, external skills,
  apps, browser, memory or agent delegation. The exact flags and tool calls are
  retained in `session2/launch.json` and `session2/events.jsonl`.
- Outer bubblewrap exposed only read-only OS/CPU libraries and `/packet`, plus
  the reviewer's writable `/work`. The original home, repository, SSH files,
  private configuration and credentials were not mounted. There is no process
  filesystem; only a fixed `/proc/self/exe` symlink needed by the CLI.
- An existing API key was loaded only into the Codex runtime environment, never
  copied into the packet or emitted. Model tool commands inherited no credentials
  and used an explicit safe environment. Preflight verified these boundaries.
  Provider networking was allowed; model tool networking was forbidden by the
  lane specification, not separately blocked at the network layer. The event
  log records no host, AWS or external tool-network operation.

## Infrastructure failure retained, not a source result

The distinct first session `01a0d9f7-00e2-73f0-8d41-c20e767de0dd` ran from
`19:07:18.510196` to `19:08:02.125107 UTC`. It returned REFUSE because the redundant
inner workspace-write sandbox could not construct its bubblewrap command
(`No such file or directory`), before reading the packet or running a test.
Its raw launch/terminal/event/error files are retained under `session1-failed/`.
This was not a provider refusal or source finding. Following explicit supervisor
authorization, a **new**, unresumed session used the same packet and unchanged
outer filesystem/secret protection with the redundant inner sandbox disabled.
There was no retry of the earlier unrelated Opus provider refusal.

## Actual reviewer-authored work

The reviewer reconstructed and independently hashed both sources, wrote its own
AST-based tiny reference tests, and executed them in the isolated CPU environment
with seed 42 and one intra/inter-op thread. Six cases covered BF16 linear,
FP32 same-dtype alias behavior, transposed BF16, dtype fallback, small-token MLP
fallback, and active custom-MLP backward. It compared original/default/OFF/ON and
cache-disabled paths through three tiny AdamW steps and two independent backward
microbatches per step: **9,248 assertions; 2,582 bitwise tensor comparisons;
zero mismatches**. Maximum test width was 64, context 128, microbatch 2; no
full-size model or native compiler was invoked. The independent raw-record audit
made **315 checks**, including all 12 recorded losses and 65 final rank-0 digests.

The first exploratory test contained an over-broad checkpoint assertion for the
pre-existing FP32 same-dtype Parameter-alias behavior; it failed. The reviewer
preserved that version and raw failure, corrected only its own auxiliary test
assertion, and reran successfully without editing candidate code or relaxing
numerical equality. Both attempts are archived. A missing `/proc/cpuinfo` library
diagnostic is also preserved; the final test completed with exit 0.

Key unmodified reviewer hashes:

| Artifact | SHA256 |
| --- | --- |
| `REVIEW.md` | `6f0b7b0708e382147e4ca44fdadc85c3097c56836e388e14470318d52cb2ae9f` |
| `result.json` | `93a4f8b860d50bb1f5c8b112338c5613966d97510b30114db11d6a441d1d4c26` |
| `reference_exactness.py` | `af94408e554683235c7ac65e12a12027571041c7f08d707918bde7ee9bdee5fe` |
| `reference_exactness.raw.log` | `5dc533fda6c3a92cae1173372a8acb82617c9c0e92a24fa26cf1503c7f596066` |
| `reference_exactness.results.json` | `4fa4ed659fa0efa3f198f9830225178b520b1b3f53a215cc46aaf4dc34218449` |
| `receipt_audit.json` | `adee2d084e6b07dbb3302e1992912e871f4aefee59b1eaaf05cc2d11a7540e69` |

All text, scripts, JSON, logs and reconstructed sources are preserved unmodified
under `session2/reviewer-output/`. Its original absolute `/work/` references map
to that folder. The 8,406,093-byte tiny diagnostic tensor snapshot is deliberately
not committed to Git; it remains at
`/tmp/trainium-fresh-architecture-review-r2-20260926.Hx9821/output/reference_exactness.tensors.pt`,
SHA256 `d02460fd09e4dc3a7e4f4b03686e8ee50ba47255dd67920ed226f4042e180dcf`.
It is not a full-size model checkpoint. No artifact was deleted.

## Bounds and disposition

Read the reviewer's `REVIEW.md` and `result.json` for the authoritative independent
findings. In particular, full-size raw G2 records were audited, not rerun locally;
their synthetic input, fixed clock, CE no-op, 2x256 slice, digest and checkpoint
availability limits remain. Local PyTorch 2.13 CPU evidence is not installed
Neuron/PyTorch 2.11 compiler proof. This operation does not approve the native
binding, waive cold startup or timing gates, or authorize a quality/full run.

The archive validation is stdlib-only hash/structure checking, not another
numerical review. Reviewer scripts and conclusions were not edited or rerun by
the operation agent. The next independent P session receives no files from this
review and remains a separate packet/session.
