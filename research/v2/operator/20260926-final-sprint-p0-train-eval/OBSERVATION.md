# Protected P0 near-end observation

One bounded, read-only parallel observation was made after the Supervisor's
07:48 request to gather the planned train-to-eval milestone. No remote file,
job, lock, cache, queue or source was changed. Both were still training; neither
host was declared free. No second observation or launch was made.

| Host | Actual UTC | Last printed step | Loss | Tokens/s | Source-clock remaining | Free bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| trn21 | 07:48:46.163960 | 3550 | 2.7321 | 267531 | 55 s | 14638268416 |
| trn22 | 07:48:45.975022 | 1840 | 2.7020 | 282546 | 88 s | 17014910976 |

Both original four-rank process sets, their individual start ticks, world=4,
LNC=2 and original guardian processes are unchanged. The original locks
66305:596823 and 66305:832589 remain held; guardian and immediate timeout/harness
processes retain FD3. Neuron device process lists exactly match the four known
workers on each host. No fatal traceback, runtime error or OOM was observed.

The terminal training rehearsal, run exit, checkpoint identity and attached
public-20M launch files are still absent in this snapshot. Exact uncapped
fifth-boundary startup, capped allowance, final charge and actual training exit
therefore remain pending. Current phase elapsed times of 2577.144684 s and
2576.689768 s are **not** startup times or official charged durations.

The printed remaining clocks suggest the next useful handoff observation at
approximately 07:51 UTC. Previous 8-layer fresh-20M durations suggest a score
around 08:01–08:06 UTC if the handoff succeeds; this is an estimate, not a result.
All protected training and attached evaluation phases must finish naturally.

## Immutable captured receipts

- `trn21-live.json`: SHA256
  `b99bbb80d2c98e96fb3909470f9423b915a9f08fb16b9c59b5ba9f2ed45ee308`.
- `trn22-live.json`: SHA256
  `aaab04e3fc3cfc9ff617173bee7a2c1ec2ddde2a170fc801f48a90ffb06cdf23`.

Local authentication passed 100 checks: all nine present files per host have
matching captured SHA/size and no tail truncation; exact train source pins,
lock identities, guardian and worker identities, ranks/LNC, device PID sets,
finite last losses and absence of observed fatal log signatures are checked.
The collector preserves complete named terminal receipts up to 2 MiB per file
and a 16 MiB aggregate bound; it never reads or copies checkpoint/cache contents.

## Next-slot constraints

The separately committed two-checkpoint cleanup note and canonical narrow
assignment apply only after each P0 plus attached evaluation completes and
explicit descendant/device/lock quiescence is proven. No cleanup occurred.
K2 requires admission for the whole OFF/ON pair while retaining both unscored
short-run checkpoints. Current free bytes are not a next-job reservation or a
claim that a future resource gate has already passed.
