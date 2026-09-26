# Actual f804 native201 launch — 26 September 2026

One released diagnostic on trn22, no attached evaluation or automatic successor.
Assignment `a77b5904c`, proposalSHA `e2c199370c1a0ed8e9a9b9d840c4b2d810bcd5522db38adcd3ef3641c5534947`.

- Label: `fs-20260926-k2-runtimegroups-native201-trn22-r1`.
- Dispatch:10:30:51.644273 UTC, PID3399165; phase10:30:51.769621.
- Flock3399166 and guardian3399167, start103271087; original66305:832589 held.
- Torchrun3399200/start103271352.
- Rank0 worker3399215/start103271555; ranks1–3 workers3399216–18/start103271556.
- First live observation10:31:23.209590; exact sourcef8047d07...96,861B, native201 argv,
  world4/LNC2/seed42/MAX1800 and actual six-cache environment matched.
- Six private cold roots absent then created empty, inodes28312836–28312841.
- Initial free168618967040B, MemAvailable129104683008B; no gate lowering.
- All16 staged member hashes and9 first-live captured hashes rechecked locally.
- Prior batch.021 terminal10:25:10.115361; all147 observed identities gone and original
  lock/device idle at10:30:24 before fresh guarded launch. Turnover341.528912s.

Exact launch records are in `../20260926-final-sprint-successors/` with prefix
`trn22-runtimegroups-native201-`. Dispatch receiptSHA
`284cd5ec95e2126aba0c612c3113992b14e9a50327bc58dc7c9e74bfcdb7c83a`;
first-liveSHA `7db70311fbc10a795f56844434618531ef22083f6d498677441786fafdd58da0`.

The existing all-outcome retriever is mechanically rebound in
`../20260926-final-sprint-successors/retrieve_runtimegroups_native201.py`.
Use only after actual terminal/outer exit; the next natural observation supplies
the additional process-identity receipt. Observation timeouts never cause restart.

Startup, timing, net tokens and losses are not yet known. The required continuation
uses measured cold startup and net tokens, not speed alone. Full/table packets
remain disabled. Source662c on trn21 was not changed or interrupted.
