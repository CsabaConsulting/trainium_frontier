# Speculative source prepared; author CPU checks passed

Prepared 2026-09-26 10:17 UTC by the assigned Implementer.
Source SHA256: `c2e27ccde59e619d74102429e340716dbec16a6ba538ea9beca9c714d0d33f87`.
The full diff is in `source.diff`; input identities and scope are in `SPEC.md`.

This is the exact50fa Recursive schedule port onto exact8c6 batch/LR source.
Removing the six original batch-port statements reproduces exact50fa byte
for byte. No clock, model, evaluator, optimizer ownership, data, K2 or table
change was introduced. Existing batch/LR defaults remain 131072/.015;
Recursive is OFF by default. The proposed 262144/.021/Recursive argv parses.

Bounded author test PASS: 70 donor boundary cases, 560 group-routing cases,
three-step OFF identity at default .015/accum1 and target .021/accum2, and
donor ON three-step identity at target .021/accum2. Synthetic eight-layer
width128/vocab256/context32 model has 1,475,892 parameters and uses the actual
MuonAdamW optimizer. Loss/input/gradient/model/optimizer/RNG/config receipts
are bit-exact. The rerun took 18.47 seconds with 406,792 KiB maximum RSS,
one CPU thread and a 4 GiB address-space cap. Local heavy slot is released.

The first attempt's failed observer assertion is retained in
`test-attempt-1.log`; it indexed `to_bf16` instead of `one_beta2`. Only test
instrumentation changed. `CPU_RECEIPT.json` and `test-attempt-2.log` bind the
complete successful rerun and unchanged scientific source.

No actual 131072/262144-token global batch, distributed ownership, full113M
model, Neuron compilation/native timing, checkpoint I/O or BPB was tested.
These checks are Implementer evidence, not an independent review.

Actual batch confirmation and any launch release remain with the Supervisor.
The single scoped independent review/composition acceptance, exact source/argv
and selected-host comparator binding, and normal machine protocol still
apply. K2 and tables retain priority. No adoption or result is claimed.
No proposal, queue/state/ledger/selector, host binding or permission was
edited, and no SSH, host launch, cleanup or submission was performed.
