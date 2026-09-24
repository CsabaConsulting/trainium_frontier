# Workflow v2 installation status

Status: **Phase 0 installed; independent review and all runtime gates pending.**

- `launch_authorized` is `false`; no controller or launcher is active.
- The seeded queue is an exact copy of `reports/2026-09-24-queue-v2.json` and remains proposal-only.
- `ledger.jsonl` is empty.
- G1 remains in the main working context and is not claimed by this installation branch.
- Independent G2 review and all native/Neuron gates remain pending.
- The old `research/agentic/autoloop/selection.py` selection/control loop is historical and inactive.
  Workflow v2 must not be run concurrently with it.
- Workflow-v2 selection requires a fresh-process context-1024 evaluation over 20,971,520 tokens and
  the unmodified `workflow/tools/decide.py` rules.

## Installed identities

| Artifact | SHA-256 |
|---|---|
| `04-train-research-35c9.py` | `35c9ef799fa983bc65eabaacb2b7c4918c9838b7d41c7c1cdd338a93e19be3c2` |
| `05-train-submission-clean-116188.py` | `116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d` |
| `submissions/research_v2/train.py` | `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586` |
| `submissions/candidates/S0/train.py` | `ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb` |
| `workflow/tools/decide.py` | `a47bedd258697b0791f48dcac3098d9071b5aaf00a7664d9e39987b4052adc66` |
| `research/v2/queue.json` | `1e9dcc4fd5f38e62829dc19460d2337662184ecfbd4d21d8a20f15993be9dd00` |

The imported CPU receipts and end-to-end transcript are upstream evidence. This installation did not
independently verify them and makes no native-hardware claim. See `UPSTREAM_PROVENANCE.json`.
