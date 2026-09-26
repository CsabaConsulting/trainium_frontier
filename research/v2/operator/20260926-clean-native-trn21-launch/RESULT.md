# One-use clean native diagnostic: failed mechanics gate, lock retained

Scope: exact Supervisor release e0202f57, assignment SHA256
2c95c557e4396213b42c0eada88bc6b6f5010d30de380bb96f560d884299f082.
No scientific source, loader, checkpoint, tolerance, environment, guard or
transport was changed. No retry or successor was launched.

- Staging finished 2026-09-26 06:26:14.106024 UTC. Original lock, all17
  retained inputs, eight runtime pins, disk/RAM, STOP, ps/neuron-ls/fuser and
  clean e65a3097 project checks passed. Exact9 small packet files were staged;
  checkpoint contents were not copied.
- Detached dispatch 06:26:19.059069 UTC, PID3202396.
- Native child launch 06:26:19.841757 UTC. Timeout3202416, torchrun3202417;
  four worker identities3202421–3202424 are present in the guardian's actual
  lifecycle records. The first live observer arrived after cancellation, so
  no unobserved per-rank live environment mapping is claimed.
- Terminal 06:26:56.535471 UTC: UNRESOLVED, not numerical acceptance.

Literal error:

`RuntimeError: ARTIFACT_NONREGULAR:/home/ubuntu/trainium_confirmed36_clean_native_consistency_20260926_trn21_r1/caches/debug/neuronx-cc/fe9ba2da9e05617e76c02c31e3816f51/neuronxcc-wxrrwtkv/nc00/sg00/default_control_fast_table.bin`

The watchdog caused the unchanged guardian to TERM/reap its actual descendants.
The same cache census error then occurred in postflight. The guardian therefore
recorded waitpid_ECHILD and FAILED_LOCK_RETAINED rather than releasing the host.
The Operator sent no signal, removed nothing and did not clear a breaker.

At 06:28:30.284703 UTC explicit postflight checked43 observed PID/start-time
identities against a267-row process census: only dispatch3202396,
flock3202397 and parked guardian3202398 remained. All40 observed owned job and
compiler identities were gone. The guardian's task children list was empty;
neuron-ls showed no device processes and fuser returned1 with empty output.
Original device66305/inode596823 was still held. Host reuse remains prohibited.

Metadata-only observation at06:29:04.774414 UTC identifies the exact failing
entry as a121-byte symlink to the installed SDK's regular4096-byte file
`/home/ubuntu/workspace/native_venv/lib/python3.12/site-packages/neuronxcc/dve/dve_bin_gen3/default_control_fast_table.bin`.
The target bytes were not read. No recursive link resolution or cache edit was
performed. This explains the observed nonregular-file predicate; it does not
grant permission to weaken the guard or retry.

The exact accepted repaired retrieval transport was attempted once. It emitted
and authenticated27 noncache files totaling230411 bytes, then returned
`RuntimeError('RETRIEVAL_NONREGULAR')` when its scan reached the compiler-cache
symlink. The local receiver retained all partial files and its refusal record.
Completeness is explicitly false. All27 retained file hashes/lengths were
independently recomputed locally against the receiver's completed-file records.
Full diagnostic log, terminal, original release, runtime/input pins, lifecycle,
pre/post idle records and post-input hashes are among the authenticated partials.
No rank result is promoted to PASS; no source or submission qualification follows.

The historical RELEASE.json in this evidence directory is consumed evidence,
not reusable authority. Parent was immediately asked to close the one-use
controller launch bit. There is no automatic retry, new label or guardian-stop
permission in this record.
