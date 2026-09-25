# Architecture initial-cache refresh: exact-size CPU packet review

Verdict: **PASS for conditional CPU-only assignment**, not a numeric G2 result or native launch authorization. Independently reviewed operator packet `9cbedf6bb0470ba9b2aed08bb8f617134fe9034a`. No candidate, host, queue, ledger, or authority state was changed. No tensors were allocated in this packet review.

## Exact bindings

- Packet: `research/v2/operator/20260926-trn22-arch-initial-cache-cpu-r1/`.
- Plan SHA-256: `fb622153f463f3fdb67456380a01c61ff804fa4d0a90c2a1e008ee913a3f634c`.
- Candidate commit: `b4d83b5db66971f3c348837f04f53cba65aa6d33`; source SHA-256: `3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c`.
- Unchanged G2 tool SHA-256: `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8` at `1e21c3d797eed764f108c41546b270065ad5650c`.
- Trusted, unmodified prepare SHA-256: `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- Authenticated original architecture receipt SHA-256: `c2fee697d02322198d61951363861fb31201bd7b68d4b91f159dbd44001392b0`, previously independently accepted in `02ed18964128050192036ad1ceaa000918c01ffc`. It remains labeled as original `93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5`, not new-candidate evidence.

The separate source/reduced review `74889f48b00aef3654911b62c097ea3d5b48f111` covers the four-line default-OFF change and independent tiny base/default/OFF/ON gradient, update, optimizer-state, and checkpoint identity. That evidence is not relabeled full-size. This packet correctly avoids adding its tiny observer to the exact-size G2 route.

## Findings

No actionable packet bug found. The full parameter shape is eight layers by width 1024, with architecture attention layers `0,2,4,6,7`, 113,256,724 parameters and 65 parameter tensors. The original reference is this architecture, not dense six-layer or tiny width-128 evidence. All original flags are preserved, including seed 42, matrix LR 0.015, pack factor 32, nominal context 1024 and maximum train seconds 1800.

The two new command lines equal the original reference argv plus exactly one explicit flag: `--no-initial-cache-refresh` or `--initial-cache-refresh`. Each uses the same real prepare import, unchanged standard G2, four CPU ranks, three steps, physical synthetic 2x256 micro-batches and deterministic fake clock. CPU/no-compile/three-step overrides are supplied by G2. This is full parameter-size CPU evidence, not the production 131,072-token update, actual-data training or native scheduling identity.

Both driver bodies are AST-identical to the prior reviewed architecture driver except for their literal per-probe ROOT. The outer command retains the shared nonblocking device lock. Fresh preflight checks, distinct absent roots/labels, one-thread settings, serial execution, retained artifacts, 3600-second G2 timeout and stop-on-failure behavior are preserved. The runbook requires the prior job's terminal/quiescent state and current role assignment; this review does not supply either. Request/source/tool/prepare bindings must be verified during staging as the runbook states. No second launcher is introduced.

Candidate OFF must finish with exit 0 and G2 PASS and compare identically against the authenticated original reference before ON. ON has the same terminal/identity rule. The unchanged comparator checks per-rank losses, rank-0 final parameter digests and ownership plan. It does not independently authenticate argv or every exit field, so the packet's explicit profile/hash/all-four-rank/exit checks remain part of acceptance; the receipt audit below confirms their static bindings. Returned numeric results still need independent review.

The standard harness does not record all intermediate gradients or initial-cache autograd connectivity. The independent tiny review already covers those observations; no new full-size observer is required here. Neither CPU test grants native speed, compile-cache benefit, startup allowance, BPB, cold proof or submission readiness. Native remains separately assigned only after the actual CPU results.

## Independent checks

`audit_packet.py` ran under Python 3.14 in normal and optimized mode: **50/50 checks passed in each**. It verifies every manifest entry, Git-object candidate/tool/proposal hashes, original raw reference and profile, four rank/update/reload evidence, explicit OFF/ON bindings, ROOT-only drivers, shared lock, comparison targets and lack of self-authorization. It uses explicit exceptions rather than optimizable assertions and never imports the candidate or runs a host driver.

Commands:

```text
python3 -B research/v2/verification/20260926-architecture-cache-cpu-packet-review/audit_packet.py
python3 -B -O research/v2/verification/20260926-architecture-cache-cpu-packet-review/audit_packet.py
```

Both printed `PASS_CONDITIONAL_CPU_PACKET_ONLY`, 50 checks, and the exact plan/reference/candidate hashes above.
