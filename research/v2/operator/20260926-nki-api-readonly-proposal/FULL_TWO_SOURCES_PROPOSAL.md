# Follow-up: exact two already-observed installed SDK source files

Preparation only; no host invocation until Supervisor release. Read only:

1. `/home/ubuntu/workspace/torch_neuron_eager/torch_neuronx/nki_hop.py`,22473 bytes, SHA256 `a8b7bffbf461378b54e6f5b4ad3406b80e069a3d89122712c40df79ebef08914`.
2. `/home/ubuntu/workspace/torch_neuron_eager/torch_neuronx/nki_kernel.py`,8598 bytes, SHA256 `37b9c6f56388695fd5dd44a4638576461d8d075f9508cb80b0df9b04dfbd10a2`.

Both pins come from actual completed source inventory0770f22a, not guessed paths. Full source enables the Researcher to inspect NKIHOPCaller indexing and wrap_nki's expected argument without importing/executing SDK code or broadening to a presumed jit/indexing module.

Exact proposed command over the existing pinned trn21 SSH stdin route: `/usr/bin/timeout --signal=TERM --kill-after=2s 15s /usr/bin/nice -n 15 /usr/bin/python3 -I -B -S -` with `full-two-sources.py` as stdin. Native training/attached20M remains untouched; no device lock access, host staging, imports, tensors, compiles, environment, cache, data or checkpoint read. This command is never a launcher.

Limits:15s wall/2CPU/128MiB, exactly two opens/31071 expected aggregate bytes and hard64KiB aggregate read/output, no links/regular-file only/O_NOFOLLOW and stable before/open/after/path identities. File size or full SHA drift refuses; absent files refuse; no fallbacks, searches or retry. Scan output for sensitive-pattern matches before local display/archive. Record exact payload/command/hash/time/exit and raw bytes locally. Source evidence is not device correctness or runtime acceptance.
