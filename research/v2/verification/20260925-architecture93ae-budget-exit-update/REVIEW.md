# Independent architecture93ae budget exit update

**PASS for this warm research run's through-exit budget accounting.** The
pending timing conclusion in independent receipt52ce7a72 is now resolved by
the complete training archive in `c39844eb`; quality/evaluation and cold
submission evidence remain separate and are not passed by this receipt.

Under `research/v2/operator/20260925-trn22-l2-93ae-cpu-r1/native/completed-training/`,
all9 retrieved files independently match recorded size/SHA256, including
complete training log SHA256
`1cf3d0a6343d45a9fd2deb0c1b504efd91da91f8f8dee939d7f7f6e3b0b8c27b`.
The launch file is byte-identical to the earlier reviewed exact93ae launch.

The unchanged driver writes training START before subprocess execution and
END after torchrun/timeout returns, including worker cleanup and exit-time
export. Raw timestamps are17:41:38.131774 and18:26:01.637384 UTC on25September:

```
driver UTC interval = 2663.505610 seconds
startup allowance = 900 seconds (saturation independently proven)
charged through exit <= 2663.505610 - 900 = 1763.505610 seconds
remaining headroom >= 1800 - 1763.505610 = 36.494390 seconds
```

This uses the consistent driver UTC enclosure, not a newly invented monotonic
measurement. The final log does not reveal a clock discontinuity; bracketing
the subprocess is conservative. The exact fifth-boundary wall timestamp
remains unknown, but is unnecessary for this calculation: the same first-five
duration lower bound1544.745 already exceeds900. We do not infer a precise
startup timestamp from polling or call startup900 just because the source
printed a capped estimate.

All workers returned through torchrun with exit0. Native summary:2246steps,
1784.6 source training seconds,1785.0 source charged estimate,
startup allowance estimate900.0, `over_budget: False`. Source1785.0 and external
1763.505610 are distinct clocks; the source includes its30-second margin and
does not include all exit work. Their difference is not a budget refund.

Checkpoint recorded after exit:453,048,594bytes, SHA256
`fb41241f812b0a357a36c18e0cbfbf81adc1df26ababf86d90fa40f51ded7c6b`.
Fresh public-20M evaluation starts separately afterward; its wall is not charged
training time. No final evaluation score is claimed here. Step count is not a
failure gate. This is still a reused-cache research run, not cold feasibility
or submission certification. No host action, model/source edit, threshold
change, queue/ledger mutation or replay was performed.
