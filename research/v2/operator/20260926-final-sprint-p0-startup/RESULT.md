# Actual P0 startup observation — 07:19:12 UTC / 12:49:12 IST

Both existing jobs remain live, with the same four workers and guardian start ticks,
original lock inode held, exact selected source and actual rank/cache environment.
No first training step or terminal is yet reported. Both have live compiler descendants,
including runnable-state descendants. trn21 has printed two Muon specialization lines;
trn22 has not printed those lines yet. No fatal traceback was observed in captured logs.
No cold-start PASS, throughput, quality result or warm-cache hit is inferred.

- trn21: guardian3203817, workers3203864–3203867, original66305:596823 held;
  free14,643,949,568 bytes. Training phase started07:05:49.019276UTC.
- trn22: guardian3315643, workers3315691–3315694, original66305:832589 held;
  free17,028,300,800 bytes. Training phase started07:05:49.285254UTC.

The observation occurred about803 seconds after launch. The exact uncapped fifth-step
boundary must come from the existing harness receipts when available, not this timestamp.
There was one bounded read-only call per host and no remote writes, launch, stop or cleanup.
All newly observed compiler identities are retained for eventual explicit postflight.

For the requested window **12:35:49 to12:49:12 IST only**, host occupation by the existing
training phases is99.99760% ontrn21 and99.96448% ontrn22 (combined99.98104%). This includes
cold compilation and is not accelerator utilization or useful training-step throughput.
Whole-day utilization is unknown here. `AUTHENTICATED.json` contains exact arithmetic,
raw receipt hashes and the PID/source/environment checks.

K2 short timing alternatives were prepared in5c81763f, and exact-e1e2 prospective
control fallbacks inb261bba1. Neither is staged. Before choosing the K2 host, admission
must cover both retained timing checkpoints/cache output and ON's unchanged cold reserve;
unscored short-run artifacts cannot be presumed disposable. Current P0/evals stay intact.
