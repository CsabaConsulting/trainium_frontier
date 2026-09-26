"""Per-rank environment receipt only, then run the unchanged tiny checker."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path("/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-k2-fp32-first-trn22-d1-packet")
CHECKER_SHA = 'bd2bca7c29615df599094d3c618f5cdf0e4f4381e6d19213dc022691214ce788'
KEYS = ('HOME', 'PATH', 'NKI_VENV_PATH', 'PYTHONUNBUFFERED', 'PYTHONDONTWRITEBYTECODE',
        'NEURON_LOGICAL_NC_CONFIG', 'NEURON_CC_FLAGS', 'NEURON_COMPETITION_R1_SEED',
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TRN_EXP_LABEL',
        'TORCH_NEURONX_ENABLED_METRIC_TABLES', 'TORCH_NEURONX_METRICS_DIR',
        'TORCH_NEURONX_NEFF_CACHE_DIR', 'TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR',
        'TORCH_NEURONX_HLO_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR',
        'NEURON_COMPILE_CACHE_DIR', 'NEURON_COMPILE_CACHE_URL', 'TORCH_NEURONX_DEBUG_DIR')


def record(name, value):
    with (ROOT / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('WRONG_PACKET_ROOT')
    rank = int(os.environ.get('RANK', '-1'))
    if rank not in range(4) or os.environ.get('LOCAL_RANK') != str(rank) or os.environ.get('WORLD_SIZE') != '4':
        raise RuntimeError('BAD_RANK_ENVIRONMENT')
    checker = ROOT / 'check_native.py'
    if hashlib.sha256(checker.read_bytes()).hexdigest() != CHECKER_SHA:
        raise RuntimeError('CHECKER_HASH_MISMATCH')
    record('rank-environment-r' + str(rank) + '.json',
           dict(rank=rank, world_size=4, local_rank=rank, pid=os.getpid(),
                utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                capture_phase='before Torch/NKI import; SDK honoring of variables requires compiler/cache evidence',
                environment={key: os.environ.get(key) for key in KEYS},
                checker_sha256=CHECKER_SHA, checker_argv=sys.argv[1:]))
    rc, error = 0, None
    sys.argv = [str(checker), *sys.argv[1:]]
    try:
        runpy.run_path(str(checker), run_name='__main__')
    except SystemExit as exc:
        rc = 0 if exc.code is None else exc.code if type(exc.code) is int else 1
        raise
    except BaseException as exc:
        rc, error = 1, type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        record('rank-wrapper-exit-r' + str(rank) + '.json',
               dict(rank=rank, returncode=rc, error=error,
                    utc=dt.datetime.now(dt.timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
