# Startup-adjusted schedule full-parameter CPU qualification

The four serial arms reference3e79/default36a862/explicitOFF/ON completed exit0. Overall terminal22:52:10.769836UTC; original outer driver also exited. All four arms report standardG2 PASS at113256724 parameters/65 tensors/four CPU ranks/three synthetic2x256 updates.

Reference/default/OFF:12 losses,65 final parameter digests and ownership match; actual retained65-tensor checkpoint model states and B1x256 logits match exactly under the disclosed standard CPU BF16 shim. ON intentionally changes the schedule and passed its own finite/update/reload checks, not ON-vs-OFF equality.

This does not reach the fifth boundary, simulate long compile delay, establish native correctness or quality, or authorize a cold/full follow-on. No source/data/evaluator change or checkpoint download occurred.

`results-bundle.json` preserves91 exact small remote files,226544bytes, including16 original rankJSON/four torchrun logs/all requests/guards/lifecycle/checkpoint comparisons. Additional explicit observed-PID postflight is archived separately. Independent acceptance is pending.
