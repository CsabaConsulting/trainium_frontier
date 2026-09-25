"""One disabled-packet checkpoint-only behavioural diagnostic binding; derived from guarded P driver.

Only a future separately assigned sole Operator may stage/run this packet.
"""
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import time

ROOT = Path("/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-clean-checkpoint-cpu-trn22-r1-packet")
PROJECT = Path('/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1')
LABEL = "fs-20260926-clean-checkpoint-cpu-trn22-r1"
EXPECTED_HOSTNAME = 'ip-172-31-32-253'
LOCK = '/home/ubuntu/.cache/trainium_freshstart/device0.lock'
PINS = {"probe.py":"d21d35c68440e6a2b62abf5338603f0c8a4891a454bb61d3df0fdc420d7682a1","validate_receipts.py":"9565ad654d75a897bc2d8c3f39872c4af54763996af827beca87b0b8e34a5c19","candidate.py":"3f5035da293d984e15df915e2348258aea7f399bbf4d0e47aeb5c9896c35a47f","reference.py":"3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c","idle_guard.py":"adafb54a98527570a76af19ec2785e68496bf1027c51c4127664435c8c963737","dispatch_driver.py":"c71474bc920f1bdba8f577d072497c4b94ca2c0318c29d39a37555da57ea041a"}
COMMAND = ["/usr/bin/timeout","--foreground","--signal=TERM","--kill-after=15s","600s","/home/ubuntu/workspace/native_venv/bin/python","-B","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-clean-checkpoint-cpu-trn22-r1-packet/probe.py","--enable-diagnostic","--backend","cpu","--label","fs-20260926-clean-checkpoint-cpu-trn22-r1","--candidate","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-clean-checkpoint-cpu-trn22-r1-packet/candidate.py","--reference","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-clean-checkpoint-cpu-trn22-r1-packet/reference.py","--checkpoint","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-primed-trn22-r2-full-packet/train/out/final.pt","--prepare","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/prepare.py","--out-dir","/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-clean-checkpoint-cpu-trn22-r1-packet/receipts"]
BACKEND = "cpu"
ENV = {"HOME":"/home/ubuntu","PATH":"/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin","NKI_VENV_PATH":"/home/ubuntu/workspace/native_venv","PYTHONUNBUFFERED":"1","PYTHONDONTWRITEBYTECODE":"1","NEURON_LOGICAL_NC_CONFIG":"2","NEURON_CC_FLAGS":"--optlevel=1","NEURON_COMPETITION_R1_SEED":"42","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","TRN_EXP_LABEL":"fs-20260926-clean-checkpoint-cpu-trn22-r1","TORCH_NEURONX_ENABLED_METRIC_TABLES":"graph_stats","TORCH_NEURONX_METRICS_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-clean-checkpoint-cpu-trn22-r1-packet/metrics","TORCH_NEURONX_NEFF_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-clean-checkpoint-cpu-trn22-r1-packet/caches/neff","TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-clean-checkpoint-cpu-trn22-r1-packet/caches/local","TORCH_NEURONX_HLO_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-clean-checkpoint-cpu-trn22-r1-packet/caches/hlo","TORCHINDUCTOR_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-clean-checkpoint-cpu-trn22-r1-packet/caches/inductor","NEURON_COMPILE_CACHE_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-clean-checkpoint-cpu-trn22-r1-packet/caches/compiler","TORCH_NEURONX_DEBUG_DIR":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-clean-checkpoint-cpu-trn22-r1-packet/caches/debug","NEURON_COMPILE_CACHE_URL":"/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-clean-checkpoint-cpu-trn22-r1-packet/caches/compiler","NEURON_COMPETITION_R1_CACHE_DIR":"/home/ubuntu/.cache/neuron_competition_r1"}
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
        require(v.f_bavail * v.f_frsize >= 6.5 * 1024**3, 'DISK_HEADROOM')
        mem = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        require(int(mem['MemAvailable'].split()[0])*1024 >= 64*1024**3,'RAM_BELOW_64_GIB')
        require(sha(PROJECT/'prepare.py')=='5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb','PREPARE_SHA')
        checkpoint=Path("/home/ubuntu/trainium_freshstart_20260924_v2_35c9_lr015_control_trn22_r1/out/fs-20260926-arch3e79-primed-trn22-r2-full-packet/train/out/final.pt")
        with checkpoint.open('rb') as stream: checkpoint_before=hashlib.file_digest(stream,'sha256').hexdigest()
        require(checkpoint.stat().st_size==453048594 and checkpoint_before=='068749c2559f7c68f6acc76d558caff9d3806d85273fa8562483d87bbad50d25','CHECKPOINT_SHA')
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
            child=subprocess.Popen(COMMAND,cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,pass_fds=(lock_fd,),start_new_session=True)
            artifact_cap_hit=False
            while child.poll() is None:
                total=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and 'caches' not in p.relative_to(ROOT).parts)
                if total>256*1024**2:
                    artifact_cap_hit=True
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                    break
                time.sleep(0.25)
            child_returncode=child.wait()
            require(not artifact_cap_hit,'ARTIFACT_CAP_STOPPED_DIAGNOSTIC')
            child_wall = time.monotonic() - child_started
            output.write('DIAGNOSTIC_END_UTC=' + utc() + '\n')
            output.flush()
        record('child-exit.json', dict(utc=utc(), returncode=child_returncode,
                                      timeout_wrapped_child_wall_seconds=child_wall,
                                      diagnostic_only=True, official_training_budget='NOT_A_QUALITY_RUN'))
        with checkpoint.open('rb') as stream: checkpoint_after=hashlib.file_digest(stream,'sha256').hexdigest()
        require(checkpoint_after==checkpoint_before,'CHECKPOINT_CHANGED')
        record('checkpoint-readonly.json',dict(before=checkpoint_before,after=checkpoint_after,bytes=checkpoint.stat().st_size))
        if child_returncode != 0:
            record('terminal.json', dict(utc=utc(), label=LABEL, status='DIAGNOSTIC_CHILD_FAILED',
                                         returncode=child_returncode, complete=False,
                                         no_eval=True, no_automatic_next_job=True))
            return child_returncode
        # Exit zero alone is insufficient; all fixed cases and ranks required.
        validator = load_module('bound_receipt_validator', ROOT / 'validate_receipts.py')
        completion = validator.validate(ROOT, LABEL, BACKEND)
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
