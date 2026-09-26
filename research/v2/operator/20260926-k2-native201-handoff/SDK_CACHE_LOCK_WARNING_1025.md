# Actual K2 cache warning at the 10:25 UTC observation

Current source662c/native201 remains running and unmodified. No manual retry,
lock reset, signal, cache cleanup or successor launch was performed.

`trn21-boundary-1025-live.json` records the original device0.lock identity
66305:596823 **held**, with the same four train workers3280911–3280914. This is
separate from the SDK's internal local-cache locking below.

Training log warnings at 10:21:01–02 UTC, ranks1/2/3:

> Lock acquisition timed out, breaking stale lock path= .../cold-training-caches/local/locks/3fb4539f12a14621c2cff4a12c9ad36f.lock elapsed_seconds=1200 timeout_seconds=1200

Rank0 at 10:24:16:

> CCOM WARN Timeout waiting for RX (waited 120 sec) - retrying, [rx_from_child//1]

These are actual SDK warnings, not an Operator action or proof of terminal failure.
There was still no completed step0, fifth boundary, terminal or final charged-time
receipt at 10:25:09. Three new walrus compiler identities are captured. One bounded
sample of PID3282560 at10:25:40 showed50 CPUticks/0.250267s and a log advancing
through DMA spill/reload optimization. No fatal error appears in that captured
prefix/tail. Do not infer a native speed or net-token pass.

The simultaneous trn22 observation10:25:08 showed evaluator3397606 live (stateR),
harness3397592 and guardian3373965 with original832589 held. No BPB or terminal was
yet emitted. Its attached evaluation was preserved.
