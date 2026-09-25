# Same lifetime lock, in-lock process and device evidence

Prepared only. No active process or remote file was changed by this patch.

Both pending one-shot drivers use the same `idle_guard.py`, staged by exact hash
into their respective immutable packet root. The outer command stays exactly
`/usr/bin/flock -n /home/ubuntu/.cache/trainium_freshstart/device0.lock /usr/bin/python3 -I -B <root>/host_driver.py`.

The guard authenticates its parent command and the inherited descriptor's inode,
owner, regular-file type and existing `FLOCK WRITE` fdinfo. Reasserting the same
exclusive lock on the inherited open-file description does not release it. The
outer process keeps its lock until the synchronous driver exits, including all
attached phases. These two diagnostics have no eval or automatic next job.

Fresh ps and neuron-ls checks run only after lock authentication and immediately
before spawn. The only process exclusions are the exact current driver PID and
its authenticated outer flock parent. Known training/eval/compiler/launcher names
and native-process comm prefixes refuse launch. All ps rows must parse; the
device JSON must match the actually observed one-device Trn2 LNC2/four-core shape.
Any device process refuses launch. Missing/unknown schema, failed command or
unexpected stderr fails closed. No broad process-ignore regex is used.

Successful and refused guard attempts write `under-lock-idle.json` before return
or exception. The receipt contains command hashes and matched process identities,
not unrelated process command lines or environments. An outer lock failure means
the driver never starts; the Operator must retrieve the outer exit and driver log
and archive a launch-refused receipt. That case is not silently retried.

Eleven local tests passed: clean, busy torchrun/compiler/eval/device, failed ps,
failed neuron-ls, malformed ps, unknown device schema, refused lock receipt and
real inherited util-linux flock with a contending exclusive acquisition refused.
These are local control-plane tests, not new model/native correctness evidence.

Historical queue7fe160cf, run_job.py, contained.py and parked seed/full bindings
must not be reactivated: they lack this new guard. No new resolution publication
to those paths is authorized. Future native/eval bindings must reuse this guard
under the same lifetime lock and be independently bound before launch.
