# K2 FP32-first diagnostic: failed, no native qualification

The exact assigned one-shot diagnostic began 2026-09-26 00:33:35.568703 UTC and exited 1 at 00:34:04.113223 UTC after 28.54430987604428 seconds. All four raw rank receipts are retrieved. No Operator kill, retry, evaluation, integration, or adoption was performed.

All four ranks report exact direct-consumer and eager-custom gradients. The compiled-custom output is exact, but its gradient is all zero (0 nonzeros versus 32,709 expected; maximum absolute error 8.0). The ordinary compiled embedding output and gradient are exact. This narrows the observed failure to the compiled custom-autograd route in this specific synthetic discriminator; it is not a general compiler diagnosis or production-model proof. Every recorded route reports unchanged input hashes.

The raw terminal is DIAGNOSTIC_CHILD_FAILED. Torchrun reports ChildFailedError: ranks 0 and 3 exited 1 and terminates ranks 1 and 2 with SIGTERM after their receipts were written. Full errors are preserved in results-bundle.json.

Explicit postflight at 00:35:02.263547 UTC found all nine observed dispatch/flock/driver/timeout/torchrun/worker PIDs absent, no related current process or compiler, no Neuron device process, and the original device 66305 / inode 832589 lifetime lock available. All unfiltered current process identity rows are archived. The postflight raw scope prose has an inherited statement saying the timeout was unobserved; the actual PID list and first-native receipt correctly include observed timeout PID 3217195. No compiler PID was observed in the bounded live snapshots; none is invented.

The preceding W1 evidence remains separate and unchanged. Root owns scientific interpretation and next-slot state. The next reviewed formal control is not launched by this archive.
