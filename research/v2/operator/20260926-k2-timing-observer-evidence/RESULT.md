# K2's precommitted whole-window timing is already available

The exact pinned harness7a55cb87 records **all** observed boundary timestamps,
not only the first five. Its live `steps.jsonl` contains integer step indices only;
the harness keeps corresponding `time.monotonic() - started` readings in memory
and writes the entire `observations` list to `train/rehearsal.json` after successful
training. The first-five subset is used only to calculate the startup allowance.

Verified exact existing source:
`/home/poojith-devan/clean-fullcold-review-DrcoWi/submission_rehearsal.py`, SHA256
`7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b`.
The complete source is archived unchanged inside `EVIDENCE.json`; it was not imported
or executed. No remote read was necessary.

Two actual terminal receipts demonstrate the behavior:

- Batch105-step timing receipt0fd7d8cf:105 sequential timestamped observations,
  step9=799.5914532509632s, step44=832.1402450909372s.
- Architecture full receipt4c55d387:3629 sequential timestamped observations,
  including both steps9 and44, and the final step3628.

Thus the unchanged precommit `35 * 131072 / (t44 - t9)` is supported for each
45-step arm, covering complete steps10–44 including producer/backward work.
Retrieve terminal `train/rehearsal.json`, require the actual sequential45 boundaries
and valid finite positive interval, then apply the already-selected ratio>=1.03.
These are existing timing validity requirements, not a new review or experiment.

Resolution caveat: timestamps are when the existing observer notices appended
indices, polled every50ms, not instrumented exact device timestamps. The harness
does not drain a final index after child exit, so a missing step44 would be missing
evidence, not permission to invent a timestamp or substitute printed throughput.
Failed/abrupt runs may lack the full observations list and cannot pass this timing
comparison. No prospective statistic change, extra observer, model/evaluator/harness
edit, current-run polling, new source review or host action is needed.
