# DISABLED symmetric K2 native binding — trn21

Proposal-first commit dfdcfdd12eff3791f75ee6a600074955a88104d2. Preparation only: no host reservation, staging, launch, old-route retry or production adoption.

Exact new source: 9ce2e6a037212aaa76bd85b9b4ab7d311a481757, research/v2/lanes/k2-symmetric-consumer/consumer_nki.py, SHA2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae. Checker89189024/reference14572563 are unchanged. This source partitions features over explicit grid[2] with equal per-core row-group control; it is not the parked d0f0 route and its earlier failures remain unchanged.

## Scope and mandatory ordering

Only the source/host/project/label/root and derived consumer/entry/driver hash constants change from frozen84d9f94f namespace-native r3. The driver body, rank-entry body, validator, shared guard, dispatcher, four-rank LNC2 setup, fixed shape, strict full-output comparison, nonfinite handling, compiler options, resource floor, timeout and completion rules are unchanged. All original r3 packet files remain byte-identical.

Frozen label: fs-20260926-k2-symmetric-consumer-trn21-r1.
Frozen root: /home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-k2-symmetric-consumer-trn21-r1-packet.
Hostname: ip-172-31-34-85.

Do not stage or dispatch until BOTH current full75b9 with attached fresh20M AND separately assigned P decoder27768550 have reached terminal states, had required artifacts retrieved, and have explicit process/device quiescence. This packet does not schedule itself after those jobs. Fresh isolated source review, independent binding mechanics review and a separate Supervisor assignment to the sole Operator remain mandatory. A terminal failed P decode is not permission to ignore its cleanup/quiescence gate.

Use the original existing /home/ubuntu/.cache/trainium_freshstart/device0.lock, device66305/inode596823. Sole Operator must authenticate that historical identity before dispatch, not create or replace it. The inherited-FD shared guard records/authenticates the held current path identity and fresh in-lock ps AND neuron-ls. The expected historical inode is a packet/Operator gate; no extra driver branch or guard change was introduced.

## Exact existing execution mechanics

The frozen command is in bound-command.txt and the disabled JSON. One synchronous timeout600 seconds plus60 seconds TERM/KILL grace wraps torchrun world4/LNC2, with the original lifetime lock FD inherited. No extra launcher or controller was added. Six new per-label SDK cache roots are required absent, created empty before the child and recorded; compile URL equals DIR. Rank-entry captures all declared environment keys before Torch/NKI and executes the unchanged checker.

The existing resource floor remains10.5GiB disk headroom. No new RAM/output-limit promise is invented; fixed case/sample counts and the inherited child timeout are unchanged. Full raw logs/graph_stats and all bounded available compiler evidence must be retrieved, including failed runs. All model/checkpoint/training/evaluator/data paths are outside this diagnostic.

Preserve the prior packet's three optional component repeats per unique/all_same dtype after correctness. This is an intentional preservation of the old driver/validator contract, selected by the present binding instruction; it supersedes the source proposal's illustrative timing0 command. These12 samples per rank exclude metadata generation/transfer and cannot be interpreted as whole-step gain or accepted_speedup.

Mechanical completion needs all4 exit0 rank wrappers and checker receipts,20 cases/150 exact comparisons per rank, all4 three-sample component groups, exact pins/argv/environment, and finite/noncontradictory values. Missing/STARTED/REFUSE cases, signed-zero or other byte mismatch, NaN/Inf, changed threshold or missing gradient cannot pass. Exit0 alone is insufficient. Actual device correctness still needs independent artifact interpretation.

## Postflight is explicit, not inferred from timeout exit

Sole Operator records dispatch/flock/driver/timeout/torchrun/all4 worker identities and any compiler descendants while active. After any exit, retrieve postflight-explicit-pids.json proving the recorded PIDs and PPID/PGID/SID descendants absent with unfiltered process evidence plus fresh ps AND neuron-ls, and the original lock free. Name-filter emptiness or released parent lock alone is not enough. Unknown surviving/ambiguous work parks the route; no next job, automatic retry, unrelated kill, cleanup, full run or eval.

This old driver does not itself claim universal process-tree cleanup. Its synchronous child, lifetime lock and unchanged guard do not replace the explicit Operator descendant postflight. The new packet preserves that requirement rather than adding another containment framework.

## Staging list for a later separately authorized Operator

From this packet: host_driver.py, rank_entry.py, validate_receipts.py, proposal, disabled binding, bound-command and manifest. Exact new candidate consumer/checker/reference from Git9ce2e6a0 at source_module_path. Shared idle_guard.py/dispatch_driver.py from research/v2/operator/20260926-common-host-idle-guard with frozen hashes. Never copy the old consumer under the new label. Check every pin, absent unique root outputs and all role/source/binding gates before using dispatch_command. No Python-O for actual execution.

Own stdlib tests:4 mechanical tests plus85 binding/refusal checks pass normal/-O. Tests use synthetic dictionaries and fully mocked driver effects, no real driver subprocess, Torch/NKI, tensors or host access. test_rebind.py proves exact mapped runtime bytes and all original files unchanged; test_binding.py reuses prior negative fixtures with only its expected root mapping extended for trn21.

Undo: leave disabled/unassigned. No host, queue, model or source adoption needs reversing.
