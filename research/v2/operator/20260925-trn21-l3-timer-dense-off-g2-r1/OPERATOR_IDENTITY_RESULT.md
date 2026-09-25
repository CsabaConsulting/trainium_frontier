# Production-width default-OFF identity: Operator result

The Supervisor separately assigned these two synthetic CPU-only probes after
the 6ef7 schedule-ON probe passed. Both were host-detached and serialized by
the same `device0.lock`; no Neuron job overlapped either probe.

| Probe | Exact source | UTC start/end | Outcome |
| --- | --- | --- | --- |
| `fs-20260925-l3-timer-dense-off-trn21-g2-r1` | 6ef7d090... | 16:18:37 / 16:18:57 | PASS, exit 0, 19.9 s |
| `fs-20260925-c5c-dense-identity-trn21-g2-r1` | c5c410af... | 16:18:57 / 16:19:17 | PASS, exit 0 |

The unchanged pinned `preflight_dryrun.py --compare` exited **0**, returning
`identical: true`: all four ranks' first three losses are exactly identical,
all **62/62** rank-0 final parameter digests match, and ownership-plan SHA
matches. Both probes independently report no cross-rank parameter mismatch,
62 changed parameter tensors and a successful strict checkpoint load with
deterministic logits. See `identity-vs-c5c.json` for the comparison output.

Additional direct binding checks confirmed equal exact dense argv, seed 42,
width 1024, Torch 2.11.0+cpu, four ranks, three real optimizer steps and 2x256
synthetic training micro-batches. `--recursive-schedule` is absent in both
argv; the 6ef7 default-OFF branch is actually tested, not inferred from the
earlier tiny f9b6 evidence. The harness's CPU compile/NKI shims and fake clock
remain explicitly recorded. This does not prove Neuron behavior or BPB, and
does not claim identical serialized checkpoint files.

All raw aggregate/rank reports, full torchrun logs, launch commands,
thread-limited environments, preflight and terminal receipts are retrieved
under this directory and sibling
`../20260925-trn21-c5c-dense-identity-g2-r1/`. Their remote SHA-256 values
were rechecked after byte-exact retrieval, including original no-final-newline
JSON formatting. Synthetic checkpoint URIs/hashes remain recorded on the host;
neither checkpoint was downloaded or deleted.

Final distinct-SSH postflight found zero matching workloads/Neuron holders,
the common lock available, 49,950,412,800 bytes disk free and 126,742,408 KiB
RAM available. The empty conditional waiter was alive and unresolved. The
organizer prepare SHA remained 5807c46e... unchanged. Independent evidence
review and a separately bound full-run proposal remain required before a
native schedule run; this Operator does not claim a score or choose a base.
