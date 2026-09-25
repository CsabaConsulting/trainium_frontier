# trn21 first-shape FP8 diagnostic binding (not launched)

Exact packet fe781c22, unchanged source921271, checker d8187ad2.
The existing a26b7414 manual driver is mechanically rebound to one new label,
new paths, 300-second timeout and DIAGNOSTIC_ONLY receipt/terminal names.
No train.py/SDK/evaluator/kernel/threshold change or new launcher framework.

Independent review and current trn21 training+fresh20M terminal are pending.
No host pause, stage or launch was performed while preparing these files.
The prior component failure remains; successful diagnostic collection cannot
authorize timing, BPB training or a component PASS.

Per Supervisor's review clarification, every rank must contain all four named
paths with status OBSERVED. A caught per-path ERROR makes the overall terminal
DIAGNOSTIC_INCOMPLETE even if the checker exits zero. Numeric mismatches from
completed observations remain diagnostic evidence, not a component verdict.
