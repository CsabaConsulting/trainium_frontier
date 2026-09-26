# Compact K2 native failure — no timing qualification

Exact f8047d07 run `fs-20260926-k2-runtimegroups-native201-trn22-r1`
failed naturally at **10:37:58.079048 UTC**. It was discovered at the first
requested lightweight startup-window check, **10:42:11.243474 UTC**:
253.164426s after the actual terminal. No automatic alert receipt was available
before that read. The original job was never restarted, stopped or retried.

Actual harness: returncode1, wall423.7343175020069s, zero completed step
boundaries. No fifth-boundary startup, t9..t44 rate, loss0..200, net tokens,
checkpoint or public score exists. Missing metrics do not pass either branch.
The conditional1300s cancellation script was NOT executed.

First actual rank0 error (preserved verbatim in complete training.log):

```
RuntimeError: Compilation error occurred on Neuron for operation=torch_compile;
COMPILATION FAILED: Block1_LoopBody_1_Resume_2_14 -> Block1_LoopBody_1_10, Block1_LoopBody_1_LoopBody_2_12 -> Block1_LoopBody_1_LoopBody_2_12. - Please open a support ticket at https://github.com/aws-neuron/aws-neuron-sdk/issues/new.
neff cache key=a5642c52ed7a4fcc786285d5c7dd2bb2745515c42e29749a581cf2fc780cd6e9:cd47b8171ac56a23cc896e6ceb62fa0bfd9e8cf0bf9b08bd2a965094c3965433
```

Traceback reaches unchanged prepare.py516 `inputs.copy_`; the SDK explicitly
warns this is asynchronous detection, not necessarily the originating operation.
Do not infer a data-loader defect from that frame. Torchrun terminated ranks1–3
with SIGTERM after rank0 exit1. Guardian/flock exited naturally10:37:58.218155.

Complete raw retrieval at10:42:38.368619: all103 actually observed identities
absent, no related/busy census rows, Neuron idle, fuser rc1, original
device66305/inode832589 free (competitor rc0). No checkpoint. Free167693754368B.
29 captured file hashes authenticated; source/prepare/harness identities stable.

Bundle `../20260926-final-sprint-successors/trn22-runtimegroups-native201-completed/results-bundle.json`
SHA256 `025a588c82059b363a3f8aefeb1d4188d2fefbcf23e7106235408be9c03a0ba5`.
No K2 retry, full or table successor is authorized. The live trn21 c2e stack
is unaffected; a separate Supervisor-selected bracket may use the now-idle host.
