# Disabled, strictly read-only installed NKI wrapper inventory

Preparation only. No host execution, native import, launch, stage, cache access, or device lock manipulation is authorized by this packet. Supervisor must separately select a host and release this exact command. Active training/evaluation keeps priority.

Proposed one invocation: `/usr/bin/timeout --signal=TERM --kill-after=2s 15s /usr/bin/nice -n 15 /usr/bin/python3 -I -B -S -`, receiving the exact `inventory.py` bytes over the existing pinned SSH transport. No remote file is written. `-S` prevents site initialization, and no third-party package is imported. Only stdlib `PathFinder.find_spec` against one explicit search directory locates top-level package metadata; it does not import the package. No nested `importlib.util.find_spec` is used.

Exactly six selected paths under the source checkout already identified in existing native tracebacks:

- `/home/ubuntu/workspace/torch_neuron_eager/torch_neuronx/__init__.py`
- `/home/ubuntu/workspace/torch_neuron_eager/torch_neuronx/nki_hop.py`
- `/home/ubuntu/workspace/torch_neuron_eager/torch_neuronx/nki_kernel.py`
- `/home/ubuntu/workspace/torch_neuron_eager/torch_neuronx/nki_op.py`
- `/home/ubuntu/workspace/torch_neuron_eager/torch_neuronx/nki.py`
- `/home/ubuntu/workspace/torch_neuron_eager/torch_neuronx/experimental/__init__.py`

Optional files may be absent. Report missing rather than scanning, following another import, or broadening scope. Source presence/export evidence is not proof that importing or executing that API works on the device.

Bounds: 15 seconds total / 2 CPU seconds / 128 MiB address space, six file reads, 512 KiB per file / 2 MiB aggregate read, 192 KiB total source excerpts / 256 KiB total JSON output. Streams use 64 KiB chunks. Every ancestor and selected file must be non-symlink, regular files only, opened read-only with `O_NOFOLLOW`, before/open/after/path inode-size-mtime-ctime identities equal. No recursive traversal, source reconstruction, SDK import, tensor, compile, network, subprocess, environment dump, auth files, cache, data, model or checkpoint access inside the remote script.

Output: package origin metadata, exact selected-file identities and full-file SHA-256, line-numbered `nki_op` / `wrap_nki` source snippets, AST-normalized function signatures and definition SHA-256 where statically present, and explicit absent/unknown/refused limitations. Imports/reexports are recorded without following them. Exceptions or bound violations remain visible; no retry or fallback. Source excerpts are from the installed SDK only. The transport must scan output for sensitive-pattern matches before display or archive.

This read is deliberately tiny and low-priority, but not claimed zero-cost. It is safe to postpone to avoid a tight native timing boundary. It acquires no device lock because it neither inspects nor touches the device; the existing job retains its original exclusive lifetime lock without interruption.
