# A2 trn22 CPU G2 prelaunch HOLD before one-use marker

At `2026-09-25T14:50–14:51Z`, the Operator's final SSH launch seam exited
with status 1 **before** creating `ATTEMPT_ONCE.json` or starting G2. It
emitted no stdout/stderr because a quiet shell assertion failed. Subsequent
read-only inspection identified the exact failed assertion:

```bash
[[ -z $(find "$root/tmp" -mindepth 1 -print -quit) ]]
```

The preceding approved clean-environment TMPDIR/Torch import check at
`14:49:10Z` had created the empty private directory
`$root/tmp/torchinductor_ubuntu` (mtime `14:49:09Z`). The packet requires
private TMPDIR resolution, **not** an empty TMPDIR after Torch import. This
extra Operator guard was therefore overstrict and is not evidence of a
science, resource, source, or numeric G2 failure. No cleanup or attempt to
repeat the launch was made.

The unique root was staged once at `14:47:53Z`, with private `source/`,
`tool/`, and `tmp/`. The source and G2 tool were copied from the clean
Operator worktree at HEAD `6034746a90b3155d1d239bf5e3325740016fc487`
and verified locally before/after transfer and on host against the pinned
SHA-256s `c5c410afef8269850aaea01f5b986b522dc0bf5f76f9a64953f8daf1288db586`
and `6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`.
The trusted host `prepare.py` hash matched
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.
The clean-environment probe resolved `tempfile.gettempdir()` exactly to the
private `tmp/` and imported Torch `2.11.0+cpu` from the expected venv.

Read-only host checks at `14:51:38Z` confirmed `ATTEMPT_ONCE.json`,
`g2.json`, `launcher.log`, and `driver.log` all absent, with no G2 worker,
train/eval process or Neuron holder. The staged root and all prior work are
preserved. The Operator has HOLD pending Supervisor direction on whether
continuation of the same staged, unmarked one-use assignment is permitted
with only this extra TMPDIR-emptiness assertion removed. No alternate label,
new root, changed source/tool, changed flags, or host cleanup is proposed.
