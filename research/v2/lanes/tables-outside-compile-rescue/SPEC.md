# Conditional tables on the outside-graph K2 rescue

Local preparation only; not a released host job or submission candidate.
Human authority: if the current K2 rescue qualifies, tables timing is next.
Source ba5add518bfb93206313476b2cd929cd678b0ed63b0045d65ad547970a9b7ee7.

Composition: frozen FP32 table parent4b74 plus the949f graph-boundary strategy,
extended to gather both b7/t7 tables before the compiled transformer. Projection
layers and their training remain inside the transformer graph. Both kernel
bodies, metadata, collision accumulation, RMS state, clipping, hashes, checkpoint
and eval loader are unchanged. No BF16 collision accumulation is introduced.
The only other additions are949f's positive batch/LR parser overrides, allowing
the current batch262144/LR.021/.75/momentum300/seed42 control configuration.

One research factor versus949f: last-layer bigram+trigram tables, FP32 masters,
gradients and RMS accumulators; gathered activations BF16. Both registered
backwards stay outside captured AOT graphs. No other architecture/schedule gain.

The current native201 K2 must first qualify with measured cold net tokens>=+3%
and acceptable loss/correctness. Old table/kernel native evidence is component
evidence only; this integration has no new native PASS, cold startup or BPB.
The new table timing must measure cold startup/charge/steady rate itself.
No queue, launcher, budget, evaluator, data or submission changes here.
