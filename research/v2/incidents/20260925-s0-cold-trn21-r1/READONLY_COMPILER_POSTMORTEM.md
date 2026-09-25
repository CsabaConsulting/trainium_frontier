# S0 trn21 r1: read-only zero-step compiler postmortem

Observed 2026-09-25 UTC. This is an investigator's read-only receipt, not an
independent verification, a root-cause claim, or authority to retry the label.

## Bound identities

- Failed label: `fs-20260925-s0-cold-replay-trn21-r1`.
- Host: trn21; run root:
  `/home/ubuntu/trainium_s0_cold_20260925_trn21_manual_r1`.
- `train.py` SHA-256:
  `ad223cdc9aa339d820b8f66e32ad35e30d3e93437a79d263a1418f7e53e125bb`.
- Launch: `NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py`;
  launch JSON says `NEURON_CC_FLAGS` was unset before importing `train.py`.
- Started `13:17:22Z`; exited `13:17:44Z`; zero training boundaries and no
  checkpoint or public BPB.

## First recorded error, then reported failure

The first preceding error found in the host journal is exact text:

```text
Sep 25 13:17:35 ip-172-31-34-85 kernel: python3[2863535]: segfault at 7176fd3f7990 ip 0000718f4609e77d sp 00007176ffbfae40 error 4 in libc.so.6[9e77d,718f46028000+189000] likely on CPU 11 (core 5, socket 0)
```

The first exception in the run's `training.log`, at `13:17:43Z`, was rank 3
during `dist.init_process_group('neuron')` / `_nrt_barrier`, before model
construction or a training step:

```text
RuntimeError: Compilation error occurred on Neuron for operation=nki_kernel_global;
error message="COMPILATION FAILED: Failed to read file: /home/ubuntu/trainium_s0_cold_20260925_trn21_manual_r1/cache/debug/neuronx-cc/47773f9a3f7185347fc76735876b8080/module.neff"
neff cache key=nki_24_set_rng_state_gpsimd_0cce7292df05aa59_()_[1]_((torch.Size([1, 6]), torch.int32),):cd47b8171ac56a23cc896e6ceb62fa0bfd9e8cf0bf9b08bd2a965094c3965433
```

Torchrun then SIGTERMed ranks 0-2 after rank 3 failed. This points to an NKI
RNG/runtime initialization kernel, not to a model forward pass. The kernel
journal showed no OOM or permission-denied message in the incident interval.

The `47773f9a...` debug directory did not exist on inspection. The entire
S0 debug tree contained just
`neuronx-cc/480d891635e27ba6ab95a68eb3b81e21/module.mlir` (866 bytes,
timestamp `13:17:43Z`), also a `set_rng_state_gpsimd` graph. No S0
`log-neuron-cc.txt` or similarly named compiler log was found in the S0 run
root, `/tmp`, `/var/tmp`, `/home/ubuntu/.cache`, or `/home/ubuntu/workspace`
for the incident time. A later log at `13:33:19Z` belonged to a *different*
trn21 A0 run and must not be attributed to S0. `coredumpctl` provided no
record/argv/parent for PID 2863535. Its identity is therefore only `python3`;
calling it a `neuronx-cc` child is **not proven**. The missing `.neff` can be
downstream of the earlier segfault, but the causal link is likewise unproved.

## Cache-state caveat

The S0 launch JSON set empty job-owned `NEURON_COMPILE_CACHE_DIR` and
`TORCH_NEURONX_*` cache directories, but did **not** set the documented
`NEURON_COMPILE_CACHE_URL` or `NEURON_CC_FLAGS --cache_dir`. The named
`NEURON_COMPILE_CACHE_DIR` remained empty after failure. AWS's published
PyTorch NeuronX cache contract names `NEURON_COMPILE_CACHE_URL`/`--cache_dir`
and defaults the persistent cache to `/var/tmp/neuron-compile-cache`:

- [AWS Neuron environment variables](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.30.0/frameworks/torch/torch-neuronx/api-reference-guide/training/torch-neuron-envvars.html)
- [AWS Neuron persistent cache](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.1/about-neuron/arch/neuron-features/neuron-caching.html)

On trn21 the default directory held about 273 MiB of older (September 14)
NEFFs. The archived September 24 research `TORCH_NEURONX_NEFF_CACHE_DIR`
held about 193 MiB of NEFFs; its `NEURON_COMPILE_CACHE_DIR` was empty. These
facts do **not** prove which cache implementation/path this competition's
native-beta runtime actually used. In particular, the r1 label says
"cold", but its persistent compiler cache was not demonstrably isolated.
Do not call code-versus-cache settled from this single failure.

## Next discriminators, not yet authorized launches

1. Same S0 SHA and bare argv on trn21 with an authenticated populated research
   cache, after its A0 control ends; see the separate warm-S0 proposal.
2. Known-good 116188 SHA on trn22 with fresh empty directories explicitly
   configured through documented `NEURON_COMPILE_CACHE_URL` and the native
   cache variables; see the separate cold-contrast proposal.
3. Regardless of these outcomes, S0 remains **not submittable** until a new
   unique-label, verified *true cold-cache* S0 run completes with checkpoint
   and fresh-process public20M evaluation.

No host data was changed or deleted for this investigation.
