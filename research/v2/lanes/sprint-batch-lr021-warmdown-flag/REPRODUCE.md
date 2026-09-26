# Executed bounded command and historical documentation pin

From `/home/poojith-devan/Desktop/Trainium`, one process at a time:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 180 /home/poojith-devan/venv/bin/python -B research/v2/lanes/sprint-batch-lr021-warmdown-flag/author_cpu.py
```

This actual execution exited0 and produced CPU_RECEIPT.json. Its script pins
the tested historical Researcher spec61ea9e1d, archived byte-for-byte in this
lane as IMPLEMENTATION_SPEC.tested.md. The current Researcher document was
subsequently changed only from `source-size` to `model-size` (cf1e9240).

Literal reproduction requires the historical61ea bytes at the spec path in
an isolated checkout; do not overwrite the live corrected document or weaken
the old hash assertion. Recover those bytes with `git show d08ce1130:` followed
by the spec's repository path, or use the archived copy. The raw receipt's
old document pin is intentional and must not be relabelled as currentcf1e.
No rerun was needed or performed for the documentation-only correction.

The fixture source is itself pinned. Only its named/dump/load/cli_prefix/route/
digest/run functions execute; its separate Recursive experiment does not run.
Data imports are removed in the in-memory test module only, with explicit
constants supplied. Actual scientific model/optimizer definitions are unchanged.
Memory4GiB, Torch/BLAS threads1, small8-layer width128 model and synthetic data.
All limits and per-step bit hashes are recorded in CPU_RECEIPT.json.
