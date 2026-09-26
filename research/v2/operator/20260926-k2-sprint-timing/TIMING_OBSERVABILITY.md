# Existing timing support — unchanged

Exact helper SHA256:
`7a55cb87e6900cb6106090e3dba0ff28ca763892fb7a67218716d0c1e3dc105b`, locally archived at
`/home/poojith-devan/clean-fullcold-review-DrcoWi/submission_rehearsal.py`.
Its lines94–96 append every boundary using `observed_seconds=time.monotonic() - started`;
line110 retains `observations=observed`, andline112 writes it to `train/rehearsal.json`.
The live `steps.jsonl` has indices only; the complete timestamps persist at terminal.

Existing actual105-step receipt:
`research/v2/operator/20260926-trn21-batch262144-timing/completed/train/rehearsal.json`,
SHA256 `0fd7d8cf1176df31be69691099c7a52cf43803705e65480b58b346c69dd2e251`;
105 sequential observations, step9=799.5914532509632s andstep44=832.1402450909372s.
Therefore retain the exact precommitted `35 * 131072 / (t44 - t9)` statistic.
Require all45 sequential timed observations; no substitution if terminal timestamps
are missing. Existing50ms observer precision applies. No new observer, test suite,
source change, evaluator/harness edit or remote action is needed.

Full local evidence (including unchanged helper text):
`research/v2/operator/20260926-k2-timing-observer-evidence/EVIDENCE.json`, commit235f0230.
