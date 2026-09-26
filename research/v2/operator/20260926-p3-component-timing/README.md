# Disabled direct P3 diagnostic alternatives

One local alternative per host, for one selected run behind K2 and the protected
P0 train/eval jobs. Exact source3637166b and existing single independent source
review PASS are unchanged. No native call, source import, staging or launch.
Supervisor still supplies the completed proposal and selects an eligible slot.

Direct argv matches SPEC: four LNC2 ranks, `--p3-component-timing --device-type neuron
--seed 42 --p3-max-seconds 900 --p3-blocks 4 --p3-repeats 3`, unique output path.
It runs from the private package directory with a byte-identical pinned prepare.py
copy solely for the ordinary import; prepare.py/evaluator/harness are not invoked.
There is no submission_rehearsal step observer, checkpoint, scoring, eval or NTFF.

Containment is the existing P0 dispatcher, idle guard and lifetime Guardian, all
byte-identical. The full P0 driver is mechanically reduced to one direct diagnostic
phase. External timeout930s permits30s before the source's900s diagnostic watchdog;
timeout TERM has15s kill grace. Existing Guardian uses its supported15s TERM and10s
KILL/reap parameters, retaining original host inode and explicit descendant/device
proof. Normal timeout+cleanup ceiling is approximately970s; unknown surviving
descendants/device state retains the lock rather than pretending idle. No new
containment framework was built. Failures keep literal logs and nonzero exit.

Each alternative names six new empty private cache directories in one namespace.
Cold provenance requires actual absent→empty and worker-environment receipts later.
Disk admission currently retains conservative13.5GiB (10.5 runtime+3 projected) and
64GiB available RAM from the proven machinery; actual diagnostic native memory and
compile feasibility are unproved. There is no authority to lower those floors here.

Output is only the source's per-rank component JSON snapshots and raw first/steady
timings. Zero outer exit remains pending raw interpretation, not automatic native
correctness/performance PASS. Partial/timeout/error remains incomplete. No whole-step
saving or BPB inference follows. Mechanical checks do not repeat the source review.
