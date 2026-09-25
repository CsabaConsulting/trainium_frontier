# trn22 c5c A0 r6: disabled optimization-safety HOLD packet

**HOLD. This packet authorizes no host contact, source transfer or checkout,
monitor arm, native dispatch, training, or evaluation.** The separate r5
package and its independent HOLD remain unchanged. The exact frozen source
commit `446b657695f086f9b484fa601c9a3b61c2495912` is retained here only
as an evidence/recipe pin, **not as a dispatch-eligible launcher**. Do not
execute its `research/fresh_start_launch.py`, directly or through a wrapper,
for r6. A distinct, reviewed source successor is necessary before an enabled
request can exist.

The planned science label is
`fs-20260925-v2-c5c-a0-control-trn22-r6`. The four JSON artifacts are new
DISABLED/static bytes, not flipped r5 files. They keep the prospective r4
checkout path only as a historical static binding, and use r6-specific
output, run-log, observer, and external-monitor paths. If the launcher
source is repaired, these bytes must be regenerated and rebound to its new
commit, source hashes, capsule, checkout, monitor bindings and final one-shot
review; do not turn `dispatch_authorized` true in this draft.

| Draft artifact | SHA-256 |
| --- | --- |
| `NATIVE_REQUEST.DISABLED.json` | `08198052a3311117b5f08b77e30d88e69012db421451d194b5d44070734dee37` |
| `BASE_BINDING.STATIC_ONLY.json` | `9fde6d69b2ba64df941e54aec66d9fe986cdc9d8050843698a0a84b8e10be730` |
| `PAIR_BINDING.STATIC_ONLY.json` | `dd1ed782e903c76a1065228f5d269ce9aafb7838d16441c1c1f3ce3c4be023da` |
| `PUBLIC20M_EVAL.DISABLED.json` | `ec86c02f74a19d207e5eb3d3435f0564ab4714ce4a06754177fc8da75c6e79db` |
| `no_dispatch_gate.py` | `c11506f0d882a9d0940164f1ec76e7239caa70fc6303522a7886102fe77d6291` |

The request → base-binding → pair-binding hashes and all eleven frozen
production-source hashes are pinned by `GATE_CONTRACT.json`. `train.py` is
`c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`,
unchanged `prepare.py` is
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`,
and the unsafe frozen launcher is
`9488efa918119439a33cbe68cb5638bd1af3b15f1f8f61ec790a216e2bc1e3f4`.
The old r4 capsule pin is
`767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`;
it is not a successor-source capsule.

## Decisive launcher defect and scope of the diagnostic gate

The independent r5 review at `919b6f99688d5bd8a23ec35e8cd133faa717b0b6`
is HOLD. In exact frozen 446b, local `dispatch_authorized` and request
validation are Python `assert` statements. Under `-O` or a nonzero
`PYTHONOPTIMIZE`, they disappear before the local dispatch receipt and SSH
call. The remote `python3 -c` script also uses assertions for checkout,
operational free space, clean tree, source commit and label absence, with no
remote `dispatch_authorized` test. Its interpreter environment is distinct
from the training child's `env -i`; a local-only wrapper cannot prove those
remote checks run. Therefore the disabled JSON flag is **not** an
unconditional no-contact gate in the frozen launcher.

`no_dispatch_gate.py` is deliberately a refusal-only diagnostic. It has an
explicit, non-assert gate for both `local` and `remote` interpreter contexts:
nonzero `sys.flags.optimize` or any `PYTHONOPTIMIZE` variable refuses, and
normal Python also refuses because the frozen source is unsafe. It imports
no SSH/host machinery and has no dispatch path. It does **not** repair or
wrap the 446b launcher, and its successful negative tests confer no launch
authority. An Operator must not use it as a substitute for source repair.

## Narrow source-successor work required before r6 can progress

1. In a separate Implementer source commit, change only the actual
   `research/fresh_start_launch.py` local entry and embedded remote script,
   plus focused tests. Keep `train.py`, `prepare.py`, evaluator, tokenizer,
   data, model, and the accepted control recipe unchanged. Place an explicit
   non-assert optimization/environment refusal at the beginning of **both**
   interpreter paths, before any local receipt/SSH or remote checkout,
   lock, receipt, manifest, or guard side effect. Reject nonzero
   `sys.flags.optimize`, any inherited `PYTHONOPTIMIZE`, and a requested
   child environment that would re-enable optimization. A remote refusal
   must propagate as failure; it cannot be mistaken for a completed preflight.
2. Replace each launcher safety `assert` with an explicit conditional
   exception. This includes the local authorization, host, label, commit,
   remote path, bounded command, Neuron configuration and floor checks, and
   the remote checkout, 6.5 GiB floor, clean-source, exact-commit and label
   checks. Add an explicit remote `dispatch_authorized is True` check before
   remote side effects. Do not rely on a wrapper around the old module or
   on `env -i`, which affects the child training command, not both launcher
   interpreters.
3. Add local/fake-remote negative tests under normal Python, `-O`, and
   `PYTHONOPTIMIZE` (including inherited remote optimization). They must
   show a disabled request and malformed/unsafe enabled requests cannot
   create a local receipt, reach fake SSH, create remote state, or reach fake
   guard spawn. Keep all fake contacts in temporary directories; no real
   SSH or host operation is part of this review. An independent Verifier
   must inspect the source diff and run these tests. New source hashes,
   commit, exact-source capsule, staged checkout and G2 identity disposition
   require separate review. The old 446b hash cannot be silently carried
   forward as the repaired launcher hash.

## Science and later gates, still disabled

The planned A0 command is the unchanged dense control: seed 42, full-chip
LNC2 with four ranks, 1,800-second training budget, matrix LR 0.015, and
the six warm training-cache strings. The command retains a bounded timeout
and exact `NEURON_CC_FLAGS`. G1 supports a fresh-process public-20M
context-1024 decision with 20,971,520 tokens and 83,462,186 public bytes;
the eval plan is disabled and has no checkpoint hash. The r2 proposal
ratification is label-specific and does not cover r6. G2 r3-to-r4
carry-forward was numeric-only and must be reassessed for the eventual
source identity and exact command; it is not native safety proof.

Before any *separate* future request, the r4 transfer and a clean exact
checkout need accepted independent results; the repaired successor then
needs its own exact-byte transfer/checkout and review. Complete an r6
proposal, Supervisor selection and one-shot assignment, final request and
binding hashes, and independent review. Immediately before a launch,
perform **every** `research/experiment-protocol.md` pre-run check on both
boxes: idle process/Neuron/lock, source sync and clean tree, seed/environment
capture, host and protected-asset identity, absent label/output, and at least
10,737,418,240 root-free bytes. Maintain the immutable 6,979,321,856-byte
floor with independently live paired monitors, stop at 9,126,805,504 bytes,
poll no slower than one second, and an authenticated read-only boundary
observer before guard spawn. One run per host; no second active launcher.
No transfer or checkout is authorized by this packet.

A valid later training terminal must yield the complete retrieved log,
finite optimizer boundaries, checkpoint hash, monitor/observer records and
postflight; a failed attempt consumes its label. Only a separately reviewed
and assigned fresh-process public-20M evaluation on the actual checkpoint
could support comparison. Public-2M is a screen, not the decision. No
quality win, adoption or submission is claimed here.

Offline Implementer checks, not independent verification:

```text
PYTHONDONTWRITEBYTECODE=1 python3 research/v2/authorizations/trn22-c5c-a0-r6-optimization-hold/test_disabled_package.py
PYTHONDONTWRITEBYTECODE=1 python3 -O research/v2/authorizations/trn22-c5c-a0-r6-optimization-hold/test_disabled_package.py
PYTHONOPTIMIZE=1 PYTHONDONTWRITEBYTECODE=1 python3 research/v2/authorizations/trn22-c5c-a0-r6-optimization-hold/test_disabled_package.py
```
