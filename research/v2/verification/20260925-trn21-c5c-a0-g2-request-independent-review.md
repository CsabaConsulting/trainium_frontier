# Independent Verifier — trn21 c5c A0 stage-r2 CPU G2 request

**Verdict: PASS for the exact disabled, offline binding at Implementer commit
`44eceac395eff36105752f72cd78934254ce8a1e`; HOLD for SSH, CPU G2,
native training, evaluation, adoption, and submission.** This is a static
review, not a live host preflight, final enabled-request review, Supervisor
assignment, or CPU numeric result. I did not contact a host or run a probe.

The reviewed diff adds only these five files under
`research/v2/authorizations/trn21-c5c-a0-stage-r2-cpu-g2-request/`; it changes
no existing source, evaluator, tokenizer, or scoring file:

| File | SHA-256 |
| --- | --- |
| `CPU_G2_REQUEST.HOLD.json` | `20bb8c7349e7d98a2ff6dd05ff7570326c8e25da9b44871600ec321434bf1a78` |
| `INVOCATION_BINDING.HOLD.json` | `62c79238920222cbb35237815b73d7d1aaef499acf155d133ec23675988066fb` |
| `RUNBOOK.md` | `2508f1bbf0be6f08d70d4ad10aa0a3d6116edc297b60b5ad2b63eaa0eedc3a17` |
| `build_cpu_g2_request.py` | `7ebeb249bc82021fc117711b61e4b779610ba54a632da0121ccc607ffb860dbb` |
| `test_cpu_g2_request.py` | `5016dc67de307b5d3b01ddd317aad895ea15cc488c689b2c07db3bafd75a768d` |

I recomputed the three hardened-G2 pins (`743d75e4…` contract,
`f308730a…` builder, `ad19dc6a…` independent static PASS), the accepted
checkout Operator/result-review pins (`1e11e59f…`, `b3c82bd5…`), and the raw
stage/unconditional-postflight pins (`09725656…`, `4dea161f…`). Their complete
digests match the request. The hardened builder in turn rechecked its frozen
draft, independently accepted transfer, final checkout-request PASS, and
checkout-only Supervisor assignment; I independently rehashed those eight
ancestor files. The accepted result review is expressly
`PASS_EXACT_SOURCE_CHECKOUT_RESULT_ONLY`, with CPU/native/evaluation authority
false. The raw stage recorded a clean, one-commit, shallow, no-remote checkout
on trn21 `ip-172-31-34-85`, device 66305/inode 3168319/mode 0700/UID 1000.

The exact local Git object `e65a3097a5b00f6178188a935bb2acb9a5042c6c`
rehashes to `train.py` `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
and `prepare.py` `5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
The archived checkout receipt reports those same bytes, with source-file
inodes 3174933 and 3168417. The two ext4 CPU helpers rehash locally to
`6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`
and `0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d`;
the raw stage and postflight agree on their host paths, hashes, device,
inodes 294946/296647, UID 1000, single links, and mode 0600. All four
compact-JSON argv digests independently match the frozen plan: launch
`1810cdc…`, wait `3a416123…`, failure stop `9216b16f…`, quiesce `c959acbe…`.
The 64-element launch array uses the exact three-step, four-rank synthetic
CPU helper with `--timeout 900 --keep`, contains `--no-eval-public`, and has
no `--real-data` or SSH command.

`git diff 44eceac^ 44eceac --check` passed. The supplied local suite passed
8/8 tests. I separately mutated 18 in-memory review, checkout, tool,
source, scratch-label, disk, command-option, and authority fields; all 18
were rejected. Direct execution of the builder exits `DISPATCH HOLD`. Its
imports and `build_plan()` read local evidence and return argv data; they do
not invoke SSH, a CPU probe, native training, or public evaluation.

**Mandatory next gates.** This package has `one_attempt_per_label=true` but
does not define or consume a fixed, exclusive/fsynced CPU attempt marker or
contain an SSH-capable Operator wrapper. That is acceptable only for this
non-dispatchable artifact. A separately named enabled request must pin and
enforce a fixed marker before any SSH, reject replay regardless of journal
path, authenticate the actual separate final-review receipt, and receive its
own independent review and narrow Supervisor CPU-only one-shot assignment.
The old checkout postflight checked an older G2 root (`…trn21-r2`), not the
proposed `…trn21-stage-r2` root. The Operator therefore needs a *fresh* full
live preflight proving that exact root, process directory, label, step log,
science outputs, and STOP/breakers are absent, while rechecking source and
tool identity, protected assets, host/device state, and the 10-GiB floor.
On any attempted result, retrieve raw evidence and perform a separate full
read-only postflight even on failure; a different Verifier must compare CPU
numerics. The e65 checkout lacks the later paired disk/STOP monitor source,
so this CPU-only PASS conveys no native-launch authority.
