# trn22 A2 exact tool-path feasibility — disabled read-only package

**HOLD: no host contact or launch authority.** This Implementer package answers
the single unknown in the [independent A2 CPU-G2 r4 HOLD](../../verification/20260925-a2-c5c-m64d128-trn22-r4-cpu-g2-final-independent-HOLD.md):
are both frozen tool files actually present inside the clean e65 checkout?
The e65 Git tree has no `workflow/tools` paths. That is evidence of a likely
absence, **not** a measurement of the host's untracked/ignored files.

The request pins these exact paths and hashes:

| Tool | Host path suffix under the fixed e65 checkout | SHA-256 required |
|---|---|---|
| CPU G2 | `workflow/tools/preflight_dryrun.py` | `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8` |
| Containment | `workflow/tools/contained.py` | `0b5aae142d2c0b0af4c8c2676521426a6e9ca6cb0c91c97fda2bf8f310c4e46d` |

The new `remote_tool_paths.py` traverses each path using directory file
descriptors and `O_NOFOLLOW`, records regular-file type, owner, device,
inode, mode, link count, size, SHA-256 and stable before/after metadata, and
reports absence without guessing. It also records `MemAvailable` and exact
checkout metadata. The collector reuses, **byte-for-byte and SHA-pinned**, the
earlier read-only dual-host snapshot for STOP/process/Neuron/lock/disk/clean
e65 commit+source identity, plus its linked-Git-pointer ACL probe. Three
strict-host-key SSH reads are joined only when host, UID, boot and mount
namespace agree. Its planning decision requires the A2 r4's 32-GiB root,
10,000-inode and 96-GiB memory thresholds; it cannot lower them.

The collector archives raw stdout/stderr as bounded base64 plus SHA-256 in an
exclusive **local** `OPERATOR_RAW_TRN22_A2_TOOLPATH_READONLY_<UTC>.json` file.
Credential-looking or oversized streams are suppressed, never parsed or
printed. No remote file, one-shot marker, scratch directory, checkpoint,
training or evaluation is created. The receipt is a dated planning snapshot,
not fresh A2 preflight and not CPU/native dispatch authority.

## Present offline state

`REQUEST.HOLD.json` has every authority flag false and `fresh_host_values`
null. `collect_readonly.py --check` verifies pinned local bytes without SSH.
`collect_readonly.py --capture` currently refuses **before any SSH** because
the three fixed future files are absent:

1. `READONLY_ENABLEMENT.FINAL.json` — exact read-only request and source hashes;
2. `INDEPENDENT_READONLY_ENABLEMENT.PASS.json` — independent Verifier's fixed-byte PASS;
3. `SUPERVISOR_READONLY_ASSIGNMENT.json` — a separate one-capture assignment.

Only after an independent review of this disabled package may another role
prepare those exact files and ask an Operator to run the capture. This
Implementer did not create any of them, contact trn22, import a writer's
dispatch entry point, touch AWS, reset a breaker, or consume the r4 CPU label.
The old trn22 A2 r4 HOLD remains controlling throughout.

Offline checks, both normal and optimized Python:

```text
python -B -m unittest discover -s research/v2/authorizations/trn22-a2-toolpath-readonly-hold -p test_readonly.py -v
python -OB -m unittest discover -s research/v2/authorizations/trn22-a2-toolpath-readonly-hold -p test_readonly.py -v
```

Each ran 8/8 PASS. Tests include a no-SSH disabled capture, no-follow
regular/missing/symlink/oversize cases, clean-vs-dirty/occupied/low-memory
planning decisions, and sensitive-output suppression. These are **offline
source tests only**, not host observations.

## Decision after a future independent Operator receipt

- If either exact path is missing, symlinked, has the wrong owner/device/hash,
  or the linked checkout is dirty, **HOLD**. Stage a distinct reviewed tool
  source at a new target, regenerate the CPU-G2 request/command hashes with a
  new label, and review it independently. Do not mutate e65 in place, fall
  back to `/dev/shm`, or spend the current CPU-G2 one-shot marker.
- If both paths match, the result proves only point-in-time feasibility. A
  different Verifier must check the raw receipt. A2 r4 still needs its own
  corrected final review, Supervisor CPU-only assignment, fresh live
  preflight, contained synthetic run, unconditional postflight and separate
  numeric attribution before any native experiment.

The historical similarly named Git branch
`codex/20260925-trn22-a2-tool-feasibility-hold` belongs to an older review
worktree; this new isolated branch does not modify or overwrite it.
