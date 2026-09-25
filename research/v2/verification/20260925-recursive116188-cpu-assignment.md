# Clean Recursive port: bounded CPU-only assignment

Supervisor reviewed the exact two requests and reused94-line CPU drivers in
`research/v2/operator/20260925-trn21-recursive116188-g2-r1/` (source commit
da4b94b7, canonical import9be08a23), plus independent source review64e7bc04.
This records the earlier direct assignment; it does not authorize native or
cold training, a leaderboard upload, cleanup, or an AWS action.

The distinct Operator `trn22_queue_operator` may execute this one serial pair
on trn21 after the active scalar training AND fresh20M evaluation have exited,
the owned queue pause has taken effect, and the single FP8 diagnostic has
finished and released all workers/devices. Hold the existing common lock and
perform the fresh STOP/breaker, idle, disk and memory checks. Preserve failures;
do not reset a breaker or retry silently. No second launcher may be active.

## Exact immutable binding

- Port4083, bare science argv: source
  `4083da774ca0f8e26b69309203b07daceca1b559a1418eacc8371f2ff5021409`;
  request `a3a060af08d03dc8160c2edaf4794feaab18171cfdda85b278c9ab6105e97ec5`;
  driver `0bb39b62c4c3074236136cf93561bdd2f24cd988ddcf8d24f2c1ec9f299e917e`.
- Adopted6ef7, exact Recursive-ON science argv: source
  `6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3`;
  request `c5d0deb67bcd8b57b932e59bdaca584a1910272f44c4809bbcfd7ba1d5330feb`;
  driver `a27da04c463f0fa37f9be75e73902309aec62a3dfcf0ac2dcee55a082605c228`.
- Both use unchanged G2 tool
  `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`
  and real trusted prepare.py
  `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
- Unique labels, absolute commands and paths are exactly those in the staged
  CPU-only proposal; the host verified their hashes at18:13:24UTC. This was
  staging, not a completed test. No local test-support imports go to the host.

Each probe uses the full6x1024 parameter shape, four CPU ranks, three optimizer
steps and synthetic2x256 inputs, with the existing G2 CPU/fake-clock/no-compile
shims disclosed. This is not a native full-microbatch or budget test.
The clean port intentionally has no OFF switch: compare its baked ON behavior
to exact6ef7 Recursive ON. Require the unchanged comparator to match all rank
losses, optimizer ownership and final state digests, not merely PASS tokens.
Retrieve both complete raw receipts, logs, strict-load results and comparator
output for independent review before using the result.

After quiescence, start the prepared genuine6ef7 new-base control before the
five preserved old-base scalar fallbacks. If a CPU gate fails, archive the
actual error and keep native port validation on HOLD; the separately checked
control is still allowed after normal machine checks. This pair does not
authorize source edits, a new launcher or a cold-port run.
