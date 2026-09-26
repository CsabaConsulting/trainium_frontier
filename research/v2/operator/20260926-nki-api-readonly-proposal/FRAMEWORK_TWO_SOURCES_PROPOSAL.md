# Disabled follow-up: two observed NKI framework files

Preparation only, no host execution. Supervisor named these two exact paths from archived94fb52c4 traceback and the actual installed nki_kernel.py import; hashes and sizes are unknown until read, so none is invented:

- `/home/ubuntu/workspace/native_venv/lib/python3.12/site-packages/nki/framework/kernel.py`
- `/home/ubuntu/workspace/native_venv/lib/python3.12/site-packages/nki/framework/compiled.py`

Proposed pinned trn21 stdin command: `/usr/bin/timeout --signal=TERM --kill-after=2s 15s /usr/bin/nice -n 15 /usr/bin/python3 -I -B -S -`, payload `framework-two-sources.py`.

Exactly two regular-file reads, <=64KiB each, <=128KiB combined and <=192KiB JSON output;15s wall/2CPU/128MiB. No links allowed in any path component; O_NOFOLLOW and stable before/open/after/path identity checks. Record full SHA256, bytes, inode/device/mtime/ctime and source text from this first read. Missing, nonregular, symlink, oversized, changed or truncated reads refuse without fallback or traversal.

No SDK imports/execution, tensors, device or lock access, cache/model/data/checkpoint/env access, subcommands, new host files or native job. No following imports beyond these two names. Sensitive-output scan precedes local archive/display. Current training transition and attached20M take priority. This source receipt does not establish numerical correctness, authorize a candidate change, or qualify a native launch.
