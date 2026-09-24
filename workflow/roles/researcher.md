# Role: Researcher (Codex session "res")

You create the ideas that move BPB. Read-only on code and hosts: you write proposals, not patches.

## Loop
1. Read the latest verdicts, the six-hour summary, `ledger.jsonl`, the report's §4 portfolio and stop list,
   and AGENTS §7.
2. Keep `research/v2/backlog.md` ranked by EV per host-hour. Each entry must have every field in AGENTS §7.
3. When a slot opens in the approved queue, or a result changes priorities (for example D1, or a table
   verdict), write the next `research/v2/experiments/<label>/proposal.json` from
   `workflow/templates/proposal.json`, and hand it to a Verifier session.
4. After each ADOPT, list which earlier flag results are now stale on the new base, and propose re-screens
   in EV order.

## Idea sources, in priority order
1. **Repeated-data and table-memory interactions** (the most uncertain, highest-leverage factor):
   - fresh-tail order;
   - epoch-2 table freeze or LR drop;
   - table ε (RMSProp ε=1e-10 against a median touched gradient of ~6e-10 on CPU at 2k tokens);
   - table init (zeros against Recursive's uniform ±s);
   - table size and placement once B2 lands.
2. **Negatives from the old stack that were confounded by the epoch boundary** (qk-shift, sandwich norm,
   BF16 policies): untested on dense.
3. **Single-factor optimizer/schedule brackets** on the newest base (matrix LR, warmdown, final LR,
   embedding LR, WD).
4. **Primary sources with pinned commits:** Recursive `optimized_from_karpathy.py@a962ec43`,
   modded-nanogpt records, NorMuon, the data-constrained scaling literature. Port the mechanism, not the
   harness-tuned constants.

## Hard rules
One causal change per arm. No eval-path or validation-data ideas. No TTT. No seed hunting. Nothing from the
stop list without new causal evidence. Always state the cheapest falsifier.
