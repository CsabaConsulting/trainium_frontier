# Implementer self-checks (not independent approval)

Local Python runtime: `/home/poojith-devan/venv/bin/python`, Python3.14.4,
Torch2.13.0+cpu. No package installed. One bounded tensor job at a time after
explicit slot handoff; OMP/MKL/OpenBLAS1, only2x128,4 ranks,synthetic1x8. The
four-worker OFF job exited before ON began; all workers exited before slot release.

The existing G2 driver SHA is
`6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`.
The test-only bridge SHA is
`7f55d6598cbfd386308b4432022ebbd7e4ba599275b35b76154877efd8a0cbf8`.
The production wrapper never imports the bridge or any G2 shim.

Exact successful OFF command from the packet worktree (ON changes only
`P_CPU_CAPTURE=1` and receipt to `tiny-on.json`):

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
 PYTHONPATH=/home/poojith-devan/Desktop/Trainium-recursive-schedule-20260925/research/v2/lanes/recursive-schedule/test_support
 P_CPU_CAPTURE=0 /home/poojith-devan/venv/bin/python
 research/v2/lanes/p-profile/cpu_g2_hooks.py
 --train-py /home/poojith-devan/Desktop/Trainium-k1-muon-audit-20260926/workflow/proposals/research-v2/train.py
 --stub-prepare --steps 3 --nproc 4 --micro-rows 1 --micro-tokens 8 --timeout 180
 --out research/v2/lanes/p-profile/tiny-off-r3.json --
 --depth 2 --n-embd 128 --no-ngram-ve --no-qk-shift --compile-sdpa-direct
 --nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce
 --seq-len 1024 --no-eval-public --matrix-lr 0.015 --seed 42
 --max-train-seconds 1800 --recursive-schedule
```

The import-only PyArrow and Gloo AVG→SUM/world compatibility support is the existing
`research/v2/lanes/recursive-schedule/test_support` from the Recursive packet.
G2 also substitutes BF16 CPU compute, disables NKI CE installation, uses a fixed
clock, synthetic tokenizer/data, slices microbatches and appends CPU/no-compile/
3-step/temp-output arguments. These are test-only, explicitly not a production
131072-token update. Its normal main block installs the actual training adapters.

OFF PASS4.3s and ON PASS5.3s are process-test wall durations, not performance
measurements. Four ranks' three exact losses, all22 final parameter digests and
ownership plan `190ff1c2328be1226db47a5cad55e262745dd4e401524cfb26aedac7128f5935`
match. ON CPU profiler captures precisely step1→2 with one forward/backward/
optimizer/data-fetch and two original sync annotations. G2 cross-rank state,
finite loss and strict deterministic checkpoint reload checks pass. No native
API, full-size model, Neuron compiler or hardware MFU was exercised.

```text
python3 workflow/tools/preflight_dryrun.py --compare research/v2/lanes/p-profile/tiny-off-r3.json research/v2/lanes/p-profile/tiny-on.json
```

Returned `identical: true`, `n_params_compared: 22`, no differences.
Normal and `python3 -O` unittest commands in RUNBOOK each pass8 tests; tensor
training itself was not rerun under `-O`.

Preserved failed attempts (no optimizer work): `tiny-off.json` failed missing
PyArrow because its declared PYTHONPATH pointed to nonexistent
`Trainium-recursive-initial-cache-refresh-20260925/research/v2/lanes/initial-cache-refresh/test_support`;
`tiny-off-r2.json` used the correct test path but `--prepare-dir
/home/poojith-devan/Desktop/Trainium` exposed missing CPU `requests`. Their receipts
contain exact argv/tracebacks. The r3 correction was switching only the test route
to existing `--stub-prepare`, not editing candidate, environment packages or scorer.
