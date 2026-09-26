# P r2 bounded decoder refusal

Assignment6458daec with state reconciliation94e0fc0e was consumed once remotely.
Actual detached start00:05:29.772457UTC; original common lock authenticated,
fresh ps/neuron-ls passed. Prior idle gap739.101360s (12m19s), below15minutes.
The two preceding local-only staging refusals are retained; neither called SSH
staging or created the r2 root. The prior r1 refusal remains unchanged.

Readiness repair worked on the installed host: shared guard at16GiB passed and
restored parent512MiB. Exact version check passed. Driver independently copied
the86 preserved profile inputs/all80NEFFs into the new private r2 root. No
original profile bytes were transferred to the laptop or modified by staging.

At00:05:37.094050UTC the decoder started. Its observed process3101016 exceeded
the sampled generated-scratch cap: `RuntimeError: GENERATED_SCRATCH_CAP`.
Sampled peak271389612bytes, overshoot2954156bytes over268435456 (256MiB).
Tracked decoder received TERM and exited−15 after1.44574183598seconds. It
produced zero stdout/stderr bytes and no usable summary. No all-model mapping,
device costs, dtype, MFU, throughput or quality conclusions are available.

Driver ended00:05:38.607968UTC after8.791105207s with
`RuntimeError: DECODER_FAILED_OR_LIMITED`, exit1. Parent/reaped-child CPU8.781054s;
postflight idle was true. Outer driver exited00:05:38.615010UTC. The source/copy
posthash stage was not reached, so it is not reported as passed. No third
attempt, larger cap, alternative decoder, cleanup or breakpoint reset occurred.

Completed raw bundle contains22 files/104923bytes, exact base64 and hashes checked.
The separate explicit postflight records all observed dispatch/flock/driver,
version and decoder PIDs plus full unfiltered process identity rows and related
PPID/PGID/SID/compiler/explorer rows, fresh device status and original lock.
All observed work is absent, device idle, original inode596823/device66305 free.

Disk after preserved4.31GB input copies and generated scratch:19.7616GB. This is
below the NIGHT25GB preference but above job floor; no deletion is assigned.
Next idle interval starts00:05:38.615010UTC;15minute threshold00:20:38.615010UTC.
Current trn22 full36 and attached evaluation remain untouched. No next job is
automatically launched by this refusal or by this report.
