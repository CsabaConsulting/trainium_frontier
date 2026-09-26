"""Fixed F0/F2 packet supervisor under the original outer flock; disabled binding."""
import json
import os
from pathlib import Path
import selectors
import signal
import socket
import subprocess
import sys
import time

# Required because the authenticated original outer invocation uses python -I.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
from binding import BINDING
from monitor import Processes, subreaper, sample, guards, free_bytes, memavailable, tree_bytes
import idle_guard


def bounded_run(command, *, capture_output=True, text=True, timeout=15, env=None):
    """Two donor idle checks: 15s each, <=2MiB combined retained pipe output."""
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, env=env)
    selector, data = selectors.DefaultSelector(), [bytearray(), bytearray()]
    for index, pipe in enumerate((proc.stdout, proc.stderr)):
        selector.register(pipe, selectors.EVENT_READ, index)
    until = time.monotonic() + timeout
    try:
        while selector.get_map():
            remaining(until)
            for key, _ in selector.select(min(0.1, remaining(until))):
                block = os.read(key.fileobj.fileno(), 65536)
                if not block:
                    selector.unregister(key.fileobj)
                    continue
                require(sum(map(len, data)) + len(block) <= 2*MIB, 'IDLE_CHECK_OUTPUT_CAP')
                data[key.data].extend(block)
        code = proc.wait(timeout=remaining(until))
        return subprocess.CompletedProcess(command, code, data[0].decode(), data[1].decode())
    finally:
        selector.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=1)
        proc.stdout.close()
        proc.stderr.close()


def environment(root, lock_fd):
    native = '/home/ubuntu/workspace/native_venv'
    env = dict(HOME=str(root/'tmp'/'home'), TMPDIR=str(root/'tmp'), XDG_CACHE_HOME=str(root/'cache'/'xdg'),
        PATH=native+'/bin:/usr/bin:/bin:/opt/aws/neuron/bin', NKI_VENV_PATH=native,
        PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', NEURON_LOGICAL_NC_CONFIG='2',
        NEURON_CC_FLAGS='--optlevel=1 --auto-cast matmult --auto-cast-type bf16',
        NEURON_COMPETITION_R1_SEED='42', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', TRN_EXP_LABEL=LABEL,
        TORCH_NEURONX_ENABLED_METRIC_TABLES='graph_stats', TORCH_NEURONX_METRICS_DIR=str(root/'logs'/'metrics'),
        K2_LOCK_FD=str(lock_fd))
    for key, name in (('TORCH_NEURONX_NEFF_CACHE_DIR','neff'), ('TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR','local'),
                      ('TORCH_NEURONX_HLO_CACHE_DIR','hlo'), ('TORCHINDUCTOR_CACHE_DIR','inductor'),
                      ('NEURON_COMPILE_CACHE_DIR','compiler'), ('TORCH_NEURONX_DEBUG_DIR','debug')):
        env[key] = str(root/'cache'/name)
    env['NEURON_COMPILE_CACHE_URL'] = env['NEURON_COMPILE_CACHE_DIR']
    return env


def main():
    started = time.monotonic()
    # No host checks, native imports, directories or children while unbound.
    require(BINDING is not None, 'DISABLED_NO_SEPARATE_HOST_RELEASE')
    require(callable(getattr(os,'pidfd_open',None)) and callable(getattr(signal,'pidfd_send_signal',None)),
            'PIDFD_PLATFORM_REQUIRED_BEFORE_CHILD')
    probe_fd = os.pidfd_open(os.getpid())
    os.close(probe_fd)
    pins = source_pins()
    root = Path(BINDING['output_root'])
    require(BINDING['label'] == LABEL and BINDING['hostname'] == socket.gethostname(), 'HOST_LABEL_BINDING')
    require(BINDING['independent_review_sha256'] and BINDING['supervisor_release_sha256'], 'MISSING_RELEASE')
    require(root.is_absolute() and '..' not in root.parts and not root.exists(), 'OUTPUT_ROOT_MUST_BE_ABSENT')
    for ancestor in root.parents:
        require(not ancestor.is_symlink(), 'OUTPUT_ANCESTOR_SYMLINK')
    terminal = dict(status='FAILED_OR_INCOMPLETE', started=started, no_automatic_next_job=True,
                    unknown_precision=UNKNOWN, native_qualification=False)
    lock_fd, child, tracker, selector, monitor_log = None, None, None, None, None
    retained = 0
    def record(name, value):
        nonlocal retained
        raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
        require(retained + len(raw) <= 120*MIB, 'DRIVER_LOG_CAP')
        path = root/'logs'/name
        with path.open('xb') as f:
            f.write(raw)
        retained += len(raw)
    try:
        lock = idle_guard.inherited_lock('/home/ubuntu/.cache/trainium_freshstart/device0.lock', HERE/'host_driver.py')
        lock_fd = lock['fd']
        for base in BINDING['stop_roots']:
            for rel in ('STOP','research/agentic/STOP','research/v2/STOP'):
                require(not os.path.lexists(Path(base)/rel), 'STOP_PRESENT')
        source = Path(BINDING['scientific_source'])
        require(digest(source) == '36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb', 'SCIENTIFIC_SOURCE_PIN')
        clean = bounded_run(['/usr/bin/git','-C',BINDING['project_root'],'status','--porcelain'], timeout=5)
        require(clean.returncode == 0 and not clean.stdout and not clean.stderr, 'UNCLEAN_PROJECT')
        revision = bounded_run(['/usr/bin/git','-C',BINDING['project_root'],'rev-parse','HEAD'], timeout=5)
        require(revision.returncode == 0 and revision.stdout.strip() == BINDING['project_commit'], 'PROJECT_COMMIT')
        for path in (root.parent, HERE, source.parent):
            require(free_bytes(path) >= 16*GIB, 'INITIAL_DISK_16_GIB')
        require(memavailable() >= 24*GIB, 'INITIAL_MEMAVAILABLE_24_GIB')
        root.mkdir(mode=0o700)
        for name in ('artifacts','cache','tmp','logs'):
            (root/name).mkdir()
        pre = {'lock':lock}
        idle_guard.check_idle(bounded_run, os.getpid(), os.getppid(), pre)
        record('preflight.json', pre)
        env = environment(root, lock_fd)
        for name in ('neff','local','hlo','inductor','compiler','debug','xdg'):
            (root/'cache'/name).mkdir()
        (root/'tmp'/'home').mkdir()
        (root/'logs'/'metrics').mkdir()
        require(time.monotonic()-started <= 80, 'PREFLIGHT_RESERVE')
        subreaper()
        tracker = Processes()
        require(len([r for r in tracker.observe()[0] if r['pid'] != os.getpid()]) == 0, 'PREEXISTING_CHILD')
        command = ['/home/ubuntu/workspace/native_venv/bin/python','-B',str(HERE/'coordinator.py'),str(root),str(started)]
        record('launch.json', dict(argv=command, environment=env, source_sha256=pins,
                                  label=LABEL, binding=BINDING, total_seconds=1200))
        child = subprocess.Popen(command, cwd=HERE, env=env, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
                                 pass_fds=(lock_fd,))
        tracker.observe()
        selector = selectors.DefaultSelector()
        pipes = [(root/'logs'/name).open('xb') for name in ('job.stdout','job.stderr')]
        for pipe, out in zip((child.stdout,child.stderr),pipes):
            selector.register(pipe, selectors.EVENT_READ, out)
        monitor_log = (root/'logs'/'resource-samples.jsonl').open('xb')
        previous, last_started = None, time.monotonic()-1
        while selector.get_map() or child.poll() is None:
            now = time.monotonic()
            require(now < started+1100, 'WORK_STOP_1100')
            if now-last_started >= 0.5:
                observation = sample(root, tracker)
                raw = (json.dumps(observation, sort_keys=True)+'\n').encode()
                require(retained+len(raw) <= 120*MIB, 'MONITOR_OUTPUT_CAP')
                monitor_log.write(raw)
                monitor_log.flush()
                retained += len(raw)
                guards(observation, previous)
                require(observation['duration'] <= 1, 'SAMPLER_TOO_SLOW')
                previous = last_started = observation['at']
            for key, _ in selector.select(0.05):
                chunk = os.read(key.fileobj.fileno(),65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                require(retained+len(chunk) <= 120*MIB, 'PIPE_OUTPUT_CAP_INCOMPLETE')
                key.data.write(chunk)
                key.data.flush()
                retained += len(chunk)
        require(child.wait(timeout=1) == 0, 'COORDINATOR_FAILED')
        observation = sample(root, tracker)
        guards(observation, previous)
        result = json.loads((root/'artifacts'/'coordinator-terminal.json').read_text())
        require(result['status'] == 'PARTIAL_DIAGNOSTIC_F0_F2_EXACT_OBSERVED_PENDING_POSTFLIGHT', 'INCOMPLETE_EVIDENCE')
        terminal['component_result'] = result
        terminal['status'] = 'PENDING_POSTFLIGHT'
    except BaseException as exc:
        terminal['error'] = type(exc).__name__+': '+str(exc)
    finally:
        if selector:
            selector.close()
        if monitor_log:
            monitor_log.close()
        containment_ok = True
        if tracker is not None:
            try:
                now = time.monotonic()
                terminal['termination'] = tracker.stop(min(now+60, started+1160), min(now+65, started+1165))
                containment_ok = not terminal['termination']['survivors']
            except BaseException as exc:
                containment_ok = False
                terminal['containment_error'] = type(exc).__name__+': '+str(exc)
        post = {}
        try:
            require(containment_ok and lock_fd is not None and time.monotonic() < started+1167, 'POSTFLIGHT_TIME_OR_CONTAINMENT')
            post['lock'] = idle_guard.inherited_lock('/home/ubuntu/.cache/trainium_freshstart/device0.lock', HERE/'host_driver.py')
            require(post['lock']['fd'] == lock_fd, 'LOCK_CHANGED')
            idle_guard.check_idle(bounded_run,os.getpid(),os.getppid(),post)
            require(time.monotonic() <= started+1198, 'POSTFLIGHT_DEADLINE')
            post['status'] = 'IDLE_OBSERVED'
            source_pins()
            if terminal['status'] == 'PENDING_POSTFLIGHT':
                terminal['status'] = 'PARTIAL_DIAGNOSTIC_F0_F2_EXACT_OBSERVED'
        except BaseException as exc:
            post['error'] = type(exc).__name__+': '+str(exc)
            containment_ok = False
            terminal['status'] = 'CONTAINMENT_FAILED'
        terminal.update(postflight=post, elapsed=time.monotonic()-started)
        if root.exists() and (root/'logs').exists():
            record('terminal.json', terminal)
        if not containment_ok and lock_fd is not None:
            # Deliberately preserve the original lock on failed containment.
            # No computation or successor; human intervention required. The
            # 1200s bound is attempted work/termination, not guaranteed OS kill.
            while True:
                time.sleep(1)
    return 0 if terminal['status'] == 'PARTIAL_DIAGNOSTIC_F0_F2_EXACT_OBSERVED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
