# Independent trn21 final-r4 private-lock transfer review — HOLD

**Verdict: HOLD.** The exact final-r4 request remains disabled and its wrapper
rejects before a marker, journal, SSH process or host write. The private-lock
repair passed the local adversarial checks below, but an inherited full-probe
validator defect can falsely accept a preflight and postflight that finish
below the request's minimum free-inode floor. This is not transfer, checkout,
CPU G2, native-launch or evaluation authority. I did not implement or edit the
candidate, contact trn21, create a transfer marker, or operate AWS.

## Exact subject and earlier evidence

Implementer commit `bea7124c390b35a2edb0b231c9620ae8054d1849` and
canonical cherry-pick `bca55b2cbc10a4d22184b7308e2c8b9c2624dc5c` have
identical Git blobs for the six new final-r4 files (`git diff --exit-code`
returned 0). The canonical commit adds only these six package files; it does
not modify any frozen r3 or older asset.

| New final-r4 file | SHA-256 |
| --- | --- |
| `FINAL_R4_DISABLED_RUNBOOK.md` | `b1ff4dff41fd4fbb158ec3af49b4f30e51c072d7dd051ee3f7b9f6cc96950bc1` |
| `TRANSFER_REQUEST.FINAL_R4_HOLD.json` | `58390b53ae09c666b705d153355baa12dd6392d8e3ea550bd44451fa41ab09e6` |
| `operator_transfer_final_r4_once.py` | `e36742d05bc517e4da0357b47144c329ec666cf197f8673f1b5f9570928bbf04` |
| `stage_r4_transport_final_r4.py` | `8907b6fe1e9f775cb1b14bbfe0be867be54fbe071098bee507762c54146e405f` |
| `stage_r4_remote_final_r4.py` | `ad30d93b464175c6df1550654076bc868a8bd8cd652fedd08bd5bfe91f055e2b` |
| `test_operator_transfer_final_r4_once.py` | `802481a6c303a4c1ce3aadd9ef92656a80b584e066d8cca5bbae5074ae4e0563` |

The request and wrapper rehash the frozen r3 request
`3763e3c6c469745a92547d27e46aa01b05874d9b7911cd7ce8f6dcdfb49dd21e`,
wrapper `6bf2aea104e9dbdefc69ee48a5bfe095e5b8f46c3d99bdd9ba9ba9aecf913a1a`,
transport `0b676d2105010a6d5e6770dc507133ca958dc677ae21079cf673713a9894a083`,
remote `3a738cff02b26824ea0be1fb50ee16b6cacce96941a9e2162506c7cf608ead32`,
tests `d404c81111c38a298f897af1026b08b22dd96163400df60fde0723175969225d`,
runbook `46f4b5c39e1ab3db200455f730bdb6bde9371ef4b2b8a186edf7cc40b0b9dcb2`,
and independent disabled-only r3 static PASS
`57e50f32bb13b29a4a9b7e41f88c09c09d26dc0eb29efe38a3e81538d53d6000`.
That r3 PASS did not approve an enabled writer; the live r3 lock probe later
HOLDed on mode `0775`/`0664`. The earlier r1 independent HOLD and r2 Supervisor
HOLD remain in force.

The new exception is grounded in the two pinned, read-only trn21 receipts:
mode/flock receipt `0557d5b940808fa3549123fe881c92f6c611d08f1a66a1401b22eeb344781f3a`
(raw `687b4d7b57c4de041adad37352a385e15ead7cfba0b83c4d3a3df5bd12058d14`)
and ACL/group receipt `065636a908c2b8b5aa8feb99e952a14a2dc63c2803140228b5d85cd496e515c4`
(raw `484ea4e76b6a59d5e5ec6757db6936b0bc3e7fc3ad89f69aa304a84da483443c`).
The raw ACL record confirms no POSIX ACL xattr on the opened `.cache`, child
or lock; named and opened UID/GID `1000:1000`, device `66305`, inodes
`295736/596822/596823`, modes `0700/0775/0664`, directory links `6/3`,
one lock link and zero lock bytes. Its 30-entry same-namespace mount scan found
no relevant explicit alias. These are point-in-time observations only. The
local 104,154,875-byte 446b capsule rehashed to
`767d1d0ce0526b822c680fccc5d79ed0423827d6f2ce3570b8d7d8b5c5d7c84a`;
frozen 446b launcher and host-lock module hashes are respectively
`9488efa918119439a33cbe68cb5638bd1af3b15f1f8f61ec790a216e2bc1e3f4`
and `b34c368c12175615514e63daebc296ec09332ef801e176e0d19cc3edc045fed3`.

## Blocking reproducer

The r4 `validate_probe()` delegates to frozen
`stage_r4_transport.py:247-258`. That base validator tests
`identity_before.root_free_inodes < MIN_INODES` but omits
`identity_after.root_free_inodes`; r4 adds no end-of-probe inode check. Both
preflight and postflight use this validator. The independent synthetic test
starts with otherwise clean pinned receipt fixtures and changes only
`pre.identity_after.root_free_inodes` and
`post.identity_after.root_free_inodes` to `9999`, below the required `10000`.
Both `validate_probe(..., before_write=True/False)` calls return successfully.
`compare_sessions()` still returns `CROSS_SESSION_PERSISTENCE_HASH_PASS`, and
`compare_postflight()` returns `READ_ONLY_POSTFLIGHT_PASS`. Thus the wrapper can
record `FRESH_PREFLIGHT_PASS` and a passing final full postflight despite a
terminal inode-floor failure. The writer's own `_prewrite_safety()` checks
both inode samples, but it cannot retroactively validate those separate full
preflight and postflight receipts. This defect is inherited from the frozen
r3/base validator; the r4 package still depends on it.

An Implementer should make the final-r4 validator reject an after-sample below
`MIN_INODES` for every full probe and add preflight/postflight negative tests.
The repaired exact bytes need a new independent review. This Verifier made no
candidate edit.

## Other static and offline findings

- The exact request has both dispatch flags `false`; the wrapper compares the
  frozen request and pinned dependency hashes, then explicitly raises
  `DISPATCH HOLD` before reading future authority files or reaching marker,
  journal or SSH. No production transfer module in this chain relies on a
  Python `assert` or `__debug__`; the same disabled-gate tests pass with `-O`.
- R4's group-writable exception is limited by the private `.cache` mode,
  owner/group/device/inode/link/size pins, descriptor-relative `O_NOFOLLOW`,
  `O_RDONLY|O_NONBLOCK` file open, repeated named/opened identity checks,
  absent opened-inode POSIX ACLs, and a complete same-namespace mountinfo
  check. Synthetic mode, inode, GID, link, device, symlink, FIFO, ACL and
  relevant mount-alias substitutions HOLD. Its read-only `LOCK_EX|LOCK_NB`
  flock blocks a competing launcher-style append-open descriptor on the same
  inode without changing lock bytes, mode or mtime.
- The remote writer repeats full protected checkpoint, e65 source, cache,
  old-path, STOP, process, lock, Neuron, target and disk checks. It holds the
  existing lock through the last activity gate, clean broad science-label
  scan, exclusive target creation, stream/fsync, same-descriptor readback and
  write receipt. A label introduced after the lock identity recheck prevents
  target creation. Distinct readback and unconditional postflight are wired
  into the wrapper; lock/label evidence is compared across sessions. A short
  stream retains its partial target and the same path is refused on retry.

`python3 -B -m unittest -v` in the package passed **51/51** Implementer tests.
The separate independent reproducer and negatives at
`research/v2/verification/test_trn21_r4_private_lock_independent.py`, SHA-256
`b9798cd4d805fa6a1103f61fbaf06fb1fe55e99bdb78c9e1a0735c1d5a8f63da`,
passed **8/8** normally and **8/8** under `python3 -O -B`. The passing
inode-gap test means the unsafe receipt was accepted as predicted, not that
the gate is correct. All tests used local synthetic data or `/dev/shm`
fixtures; none contacted a host or ran training.

Even after the inode fix, the private-parent inference remains bounded to
ordinary DAC access in the observed mount namespace and cooperation by the
frozen launcher. Another process with UID 1000, a privileged actor, an
already-open directory descriptor, or an alias in another mount namespace
remains outside that proof. Final STOP/process/Neuron/label observations are
not atomic against an uncooperative actor. Fresh live host checks, a new
enabled request, separate final review and Supervisor one-attempt assignment
would still be required for transfer only.
