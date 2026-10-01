# Round63 final paired CPU transfer diagnostic — COMPLETE; lane closed

Completed Sep29 02:41:36 IST; worker2234.788s, driver2237.461s.
Exact archived result: research/v2/operator/20260929-r63-transfer-cpu-preparation/collect-trn21-both-20260928T211252421039Z.json
SHA256 961f2245b04e01b4c4bfe91834b6914b1ec564b55c51e5cdd03331a36e723f6e; result SHA256 2d81bf4d4d26064fe1a2594f135d9bbcdf47bb2ea19abbcf15b8e6439e32da61.

Each text set: 1,048,576 paired tokens,1024 contiguous1024-token windows.
Values are diagnostic BPB, not the official-setting public20M score.

| Text | Saved D1 | D1 table zeroed in RAM | Saved b585 | D1-base | D1-zero | Bytes |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.9700936654600395 | 1.1415287507487928 | 0.971518958619925 | -0.0014252931598854879 | -0.17143508528875329 | 4201934 |
| train | 0.9390189575330712 | 1.1274054709312553 | 0.9454853673744772 | -0.006466409841406007 | -0.18838651339818413 | 4192497 |

Interpretation: zeroing the trained table materially worsens both splits. The table
is active in evaluation; these observations do not support a disconnected lookup.
D1 gains .0064664 on the sampled seen training text versus .0014253 on public
validation. This is consistent with a generalization gap, not proof excluding
every possible defect. Ablation cost is not the quality gain from training with
a table, because the other weights co-adapted. No .005-per-site claim follows.

CPU compute dtype was FP32 under the exact frozen load_for_eval policy; explicit
BF16 norm/table casts remain unchanged. This is not an all-operations-FP32 run.
Training rows reproduce the original rank/microbatch loader order; historical
per-shard/per-batch hashes were absent from the D1 receipt, as disclosed in
LOG_FINDINGS.md. No raw tokens, tensors or checkpoint copy were exported.
All protected checkpoint/source/data hashes match before/after; ownCPUchild
reaped/lockreleased, no device lock or training signal. No further diagnostic run.

CE-only losses at500/1000 and exact adjacent pass-boundary batch losses were
not logged. Existing figures are MTP-weighted EMA nats and cannot be converted
to gross validation BPB. See LOG_FINDINGS.md for the exact available log lines.
