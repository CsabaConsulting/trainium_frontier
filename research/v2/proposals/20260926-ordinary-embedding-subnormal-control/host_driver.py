"""One disabled-packet ordinary-only diagnostic binding; derived from guarded P driver.

Only a future separately assigned sole Operator may stage/run this packet.
"""
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path("/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-ordinary-fp32-subnormal-trn21-d1-packet")
PROJECT = Path('/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2')
LABEL = "fs-20260926-ordinary-fp32-subnormal-trn21-d1"
EXPECTED_HOSTNAME = 'ip-172-31-34-85'
LOCK = '/home/ubuntu/.cache/trainium_freshstart/device0.lock'
PINS = {"ordinary_subnormal_probe.py":"750ba91a5c22745bec65c2e603fa21186b6a88859ed757b7c442b825cec38679","validate_receipts.py":"55013b6aec3c8e64fc3044440b062030a3ca03371244dafab934f1d35b59a422","rank_entry.py":"e0ffe0416dc910e844cfaa13a1ab44e0a699921893bbd9e6579dd187168eeae8","dispatch_driver.py":"c71474bc920f1bdba8f577d072497c4b94ca2c0318c29d39a37555da57ea041a","idle_guard.py":"adafb54a98527570a76af19ec2785e68496bf1027c51c4127664435c8c963737"}
COMMAND = ["/usr/bin/timeout","--foreground","--signal=TERM","--kill-after=60s","600s","/home/ubuntu/workspace/native_venv/bin/torchrun","--standalone","--nproc_per_node=4","/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-ordinary-fp32-subnormal-trn21-d1-packet/rank_entry.py","--enable-control-only","--label","fs-20260926-ordinary-fp32-subnormal-trn21-d1","--out-dir","/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-ordinary-fp32-subnormal-trn21-d1-packet/receipts"]
CHECKER_ARGS = ["--enable-control-only","--label","fs-20260926-ordinary-fp32-subnormal-trn21-d1","--out-dir","/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-ordinary-fp32-subnormal-trn21-d1-packet/receipts"]
ENV = {"HOME":"/home/ubuntu","PATH":"/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin","NKI_VENV_PATH":"/home/ubuntu/workspace/native_venv","PYTHONUNBUFFERED":"1","PYTHONDONTWRITEBYTECODE":"1","NEURON_LOGICAL_NC_CONFIG":"2","NEURON_CC_FLAGS":"--optlevel=1 --auto-cast matmult --auto-cast-type bf16","NEURON_COMPETITION_R1_SEED":"42","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","TRN_EXP_LABEL":"fs-20260926-ordinary-fp32-subnormal-trn21-d1","TORCH_NEURONX_ENABLED_METRIC_TABLES":"graph_stats","TORCH_NEURONX_METRICS_DIR":"/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-ordinary-fp32-subnormal-trn21-d1-packet/metrics","TORCH_NEURONX_NEFF_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-ordinary-fp32-subnormal-trn21-d1-packet/caches/neff","TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-ordinary-fp32-subnormal-trn21-d1-packet/caches/local","TORCH_NEURONX_HLO_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-ordinary-fp32-subnormal-trn21-d1-packet/caches/hlo","TORCHINDUCTOR_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-ordinary-fp32-subnormal-trn21-d1-packet/caches/inductor","NEURON_COMPILE_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-ordinary-fp32-subnormal-trn21-d1-packet/caches/compiler","TORCH_NEURONX_DEBUG_DIR":"/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-ordinary-fp32-subnormal-trn21-d1-packet/caches/debug","NEURON_COMPILE_CACHE_URL":"/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/out/fs-20260926-ordinary-fp32-subnormal-trn21-d1-packet/caches/compiler"}
CACHE_KEYS = ["TORCH_NEURONX_NEFF_CACHE_DIR","TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR","TORCH_NEURONX_HLO_CACHE_DIR","TORCHINDUCTOR_CACHE_DIR","NEURON_COMPILE_CACHE_DIR","TORCH_NEURONX_DEBUG_DIR"]


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def record(name, value):
    with (ROOT / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    started = time.monotonic()
    child_returncode = None
    try:
        require(Path(__file__).resolve().parent == ROOT, 'WRONG_PACKET_ROOT')
        require(socket.gethostname() == EXPECTED_HOSTNAME, 'WRONG_HOST')
        for base in (Path('/home/ubuntu'), PROJECT, ROOT,
                     *[p for p in Path('/home/ubuntu').glob('trainium*') if p.is_dir()]):
            for rel in ('STOP', 'research/agentic/STOP', 'research/v2/STOP'):
                require(not os.path.lexists(base / rel), 'STOP: ' + str(base / rel))
        v = os.statvfs('/')
        require(v.f_bavail * v.f_frsize >= 10.5 * 1024**3, 'DISK_HEADROOM')
        for name, expected in PINS.items():
            require(sha(ROOT / name) == expected, 'SUPPORT_SHA:' + name)
        for name in ('receipts', 'caches', 'metrics'):
            require(not os.path.lexists(ROOT / name), 'ALREADY_EXISTS:' + name)
        (ROOT / 'caches').mkdir(exist_ok=False)
        cold = {}
        for key in CACHE_KEYS:
            cache = Path(ENV[key])
            cache.mkdir(exist_ok=False)
            require(not any(cache.iterdir()), 'CACHE_NOT_EMPTY:' + key)
            cold[key] = dict(path=str(cache), initially_empty=True,
                             device=cache.stat().st_dev, inode=cache.stat().st_ino)
        require(ENV['NEURON_COMPILE_CACHE_URL'] == ENV['NEURON_COMPILE_CACHE_DIR'], 'COMPILER_URL_DIR')
        (ROOT / 'metrics').mkdir(exist_ok=False)
        record('fresh-caches.json', dict(utc=utc(), directories=cold,
                                       compile_cache_url=ENV['NEURON_COMPILE_CACHE_URL'],
                                       scope='Created absent and empty before child; runtime use requires worker/compiler evidence'))
        guard = load_module('bound_idle_guard', ROOT / 'idle_guard.py')
        lock_fd = guard.require_idle_under_inherited_lock(LOCK, ROOT / 'host_driver.py', record)
        record('diagnostic-launch.json', dict(utc=utc(), label=LABEL, command=COMMAND,
                                             environment=ENV, support_sha256=PINS,
                                             quality_evidence=False, no_eval=True))
        with (ROOT / 'diagnostic.log').open('x') as output:
            output.write('DIAGNOSTIC_START_UTC=' + utc() + '\n')
            output.flush()
            child_started = time.monotonic()
            child_returncode = subprocess.call(COMMAND, cwd=ROOT, env=ENV,
                                               stdin=subprocess.DEVNULL, stdout=output,
                                               stderr=subprocess.STDOUT, pass_fds=(lock_fd,))
            child_wall = time.monotonic() - child_started
            output.write('DIAGNOSTIC_END_UTC=' + utc() + '\n')
            output.flush()
        record('child-exit.json', dict(utc=utc(), returncode=child_returncode,
                                      timeout_wrapped_child_wall_seconds=child_wall,
                                      diagnostic_only=True, official_training_budget='NOT_A_QUALITY_RUN'))
        if child_returncode != 0:
            record('terminal.json', dict(utc=utc(), label=LABEL, status='DIAGNOSTIC_CHILD_FAILED',
                                         returncode=child_returncode, complete=False,
                                         no_eval=True, no_automatic_next_job=True))
            return child_returncode
        # Exit zero / four files alone is insufficient. No revised checker math.
        validator = load_module('bound_receipt_validator', ROOT / 'validate_receipts.py')
        completion = validator.validate(ROOT, LABEL, ENV, PINS, CHECKER_ARGS)
        record('completion-check.json', completion)
        record('terminal.json', dict(utc=utc(), label=LABEL, status=completion['status'],
                                     returncode=0, complete=True, no_eval=True,
                                     no_automatic_next_job=True, accepted_speedup=False,
                                     driver_elapsed_to_terminal_seconds=time.monotonic() - started))
        return 0
    except Exception as exc:
        record('terminal.json', dict(utc=utc(), label=LABEL, status='DIAGNOSTIC_DRIVER_REFUSED_OR_INCOMPLETE',
                                     error=type(exc).__name__ + ': ' + str(exc),
                                     child_returncode=child_returncode, complete=False,
                                     no_eval=True, no_automatic_next_job=True))
        raise


if __name__ == '__main__':
    sys.exit(main())
