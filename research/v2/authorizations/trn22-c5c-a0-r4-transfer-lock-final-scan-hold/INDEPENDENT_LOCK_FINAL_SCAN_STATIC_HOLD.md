# Independent trn22 r4 lock/final-label review — STATIC HOLD, DISPATCH HOLD

Reviewed 2026-09-25 UTC by a Verifier separate from the Implementer. The
subject is the **disabled** package at Implementer commit
`37ef4f7375753b1010e0a1c1a567565b5ff6e600` on
`codex/20260925-trn22-r4-final-label-lock-hold` (reported canonical
cherry-pick `55ea58f8`). I read the repository contract in its required
order and treated old chats and receipts only as evidence. This review used
local source, local synthetic tests, and the previously archived Operator
**read-only** lock receipt. It made no connection to trn22.

**Verdict: STATIC HOLD for these exact bytes; DISPATCH HOLD.** This is not a
static PASS, enabled transfer request, final enablement review, Supervisor
one-shot assignment, or host-readiness finding. The earlier r3 final PASS was
superseded by its independent late-label HOLD addendum and remains historical
only. Do not use this package for marker/journal creation, SSH, transfer,
checkout, G2, native training, evaluation, or submission.

## Exact reviewed artifacts

| Artifact | SHA-256 |
| --- | --- |
| `GATE_CONTRACT.HOLD.json` | `fde40c4bc944af70e57de9159b5239c286a925d4ca8ec266f929fe99c194520e` |
| `RUNBOOK.md` | `09463781701ad6058b02d96ba62d56b132fc494173f3d6a4de8019840fecdc72` |
| `stage_r4_remote_lock_hold.py` | `26ca03e94a005513a648ce968326c6d2dfbd160cc099872f5ebed3706dfeeafd` |
| `stage_r4_transport_lock_hold.py` | `61169006f0fb22d1ff4de23dba6835f14284af3c91bdcac335d7053df755a47d` |
| Supplied `test_lock_final_scan_hold.py` | `db03bec8f73bbccbb0edaeaec1108b0cf7d769183ae9a1c5f135ed8ab59590a5` |
| New independent negative tests | `76ffedc5df51ea08212f20d0c1080e3b16d895c5b3553491425622f4b8c8896d` |
| Prior controlling r3 late-label HOLD addendum | `94de8bc3d12a4c36cdad6d2160c1db081d4ea0c0035b6024bf7471e389ffaa0b` |
| Operator read-only lock raw receipt at `58c37349d11c5d3190179d4abd2e5b5b4a2a321a` | `2355c6814dc3e847fd9882738065c049f0a39bf078ce99992bb4ac1526f88305` |

The package contract pins the **actual** remote, transport, test and runbook
hashes. The frozen launcher files at source commit
`446b657695f086f9b484fa601c9a3b61c2495912` independently matched
the contract's three SHA-256 pins. It uses the exact existing
`/home/ubuntu/.cache/trainium_freshstart/device0.lock`, obtains a nonblocking
exclusive `flock`, and passes the descriptor to its guard/child.

## Blocking findings

1. **The lock ancestry gate rejects the observed host.** The Operator's
   2026-09-25 05:11 UTC read-only receipt records `.cache` as UID/GID
   `1000:1000`, mode `0700`; `trainium_freshstart` as UID/GID `1000:1000`,
   mode `0775`; and the existing regular, single-link lock file as UID/GID
   `1000:1000`, mode `0664`. Named and opened no-follow metadata matched;
   `O_RDONLY`/nonblocking exclusive flock succeeded at that instant. In the
   candidate, `stage_r4_remote_lock_hold.py:255-260` rejects any lock
   directory with `mode & 0o022`; `0775 & 0o022` is nonzero. `probe()` calls
   that check at line 583, so it cannot produce a valid fresh preflight on
   the observed path. The transport independently rejects the same mode at
   `stage_r4_transport_lock_hold.py:243-257`. Synthetic replicas of the
   observed modes reproduced both rejections in the independent suite;
   the same synthetic read-only flock succeeded. This is a hard mismatch
   between the package policy and the last actual host evidence, not a
   claim that the lock itself is unusable or that host permissions may be
   changed. The private `0700` parent may affect the security analysis, but
   that is for a separate design/review, not an assumption of PASS here.
2. **The runbook has a wrong exact-byte transport pin.** Its SHA table says
   `50865a842b5568b21bdbd08b18a1fdf7cc02bffbbb43b0f6f7e2beab57806494`
   for `stage_r4_transport_lock_hold.py`; the actual file and contract both
   say `61169006f0fb22d1ff4de23dba6835f14284af3c91bdcac335d7053df755a47d`.
   This breaks the human-readable exact-artifact manifest and must be
   corrected in a separately reviewed package. The supplied contract-hash
   test checks the JSON pins but does not detect this runbook contradiction.

## Other static checks and limits

The specific r3 **late-label timing gap is repaired in isolation**: the new
writer acquires the retained cooperative lock at lines 712–720, reruns full
STOP/process/checkpoint/Neuron/label checks while holding it at lines
725–726, checks STOP/process and lock identity again at lines 727–730, and
makes a final empty broad label scan at line 734. Only monotonic timestamp
and one-second staleness checks intervene before the sole descriptor-relative
`O_CREAT|O_EXCL|O_NOFOLLOW` at lines 735–740; the lock is retained through
write, fsync, readback and receipt construction, then closed on success or
failure at lines 793–797. The owned-lock live check records the held inode
instead of `fuser`-checking its own descriptor, avoiding a self-fuser false
HOLD. My injected label appearing **after** the held full checks was seen by
the final scan and blocked target creation. The receipt comparator requires
the lock identity, final empty scan and ordered acquisition/open timestamps
across preflight/write/read-only-verify processes. These checks do not prove
atomic exclusion of noncooperating manual science, which the runbook itself
acknowledges.

The inherited protected checkpoint/cache, old r1/r2/r3 paths including
file-backed `.git`, host/boot/mount/source identity, STOP/process/Neuron,
one-shot target metadata and readback/postflight comparators remain present.
Only the five new Implementer files were added by the subject commit; old
PASS and HOLD evidence was not rewritten. The disabled remote `main()` and
transport `build_ssh_argv()` unconditionally raise `DISPATCH HOLD` before
marker, journal or SSH. This package has no enabled wrapper: a future
one-attempt marker/journal, strict-host-key sessions, and unconditional
separate full postflight still require a new exact request and review.

Local-only results: supplied suite **9/9**, new independent negatives
**3/3**, archived r3 independent final suite **7/7**, archived r3 static
suite **4/4**; `git diff --check` passed. The independent tests cover the
observed `0775` rejection in both remote and transport, successful Linux
read-only flock on a matching synthetic file, and the old r3 late-label
counterexample at the new final scan. They used disposable local fixtures,
not trn22, SSH, AWS or production source artifacts.

**Next boundary:** retain DISPATCH HOLD. A separate Implementer must resolve
the exact-byte runbook pin and the host-mode/security-policy contradiction;
read-only evidence of the private-parent traversal and relevant group
memberships/ACLs can inform that design, but this receipt requests no host
action and grants none. Then a different Verifier must review the new exact
bytes. An enabled transfer-only request, independent final review, fresh
host preflight and distinct Supervisor one-shot assignment would still be
required before any Operator action. No host permission change, lock
replacement, cleanup, same-path retry or authority escalation follows from
this HOLD.
