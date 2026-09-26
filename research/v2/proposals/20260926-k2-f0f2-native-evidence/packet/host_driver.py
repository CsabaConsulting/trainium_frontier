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


def bounded_run(command, *, capture_output=True, text=True, timeout=15, env=None,
                tracker=None, events=None, deadline=None):
    """Whole helper lifetime, including descendants/cleanup, within timeout."""
    subreaper()  # Before even the first git/ps helper, also for standalone tests.
    tracker = tracker if tracker is not None else Processes()
    tracker.quiescent()
    until = min(time.monotonic()+timeout, deadline if deadline is not None else float('inf'))
    cleanup_reserve = min(1.0, timeout/4)
    work_until = until-cleanup_reserve
    proc = None
    selector, data = selectors.DefaultSelector(), [bytearray(), bytearray()]
    receipt = dict(command=command, started=time.monotonic(), status='INCOMPLETE')
    try:
        remaining(work_until)
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                stdin=subprocess.DEVNULL, env=env, start_new_session=True)
        tracker.observe()
        for index, pipe in enumerate((proc.stdout, proc.stderr)):
            selector.register(pipe, selectors.EVENT_READ, index)
        while selector.get_map():
            remaining(work_until)
            tracker.observe()
            for key, _ in selector.select(min(0.05, remaining(work_until))):
                block = os.read(key.fileobj.fileno(), 65536)
                if not block:
                    selector.unregister(key.fileobj)
                    continue
                require(sum(map(len, data)) + len(block) <= 2*MIB, 'IDLE_CHECK_OUTPUT_CAP')
                data[key.data].extend(block)
        code = proc.wait(timeout=remaining(work_until))
        # A zero-exit helper that leaves a detached child is a failed helper.
        require(not tracker.alive(), 'HELPER_LEFT_DESCENDANTS')
        receipt.update(status='EXITED_AND_QUIESCENT', returncode=code)
        return subprocess.CompletedProcess(command, code, data[0].decode(), data[1].decode())
    finally:
        # Evidence/close failures must never skip descendant containment.
        close_errors = []
        try:
            now = time.monotonic()
            receipt['cleanup'] = tracker.stop(min(now+cleanup_reserve/4, until), until)
            require(not receipt['cleanup']['survivors'], 'HELPER_CONTAINMENT_FAILED')
            receipt['final_quiescence'] = tracker.quiescent()
        finally:
            for stream in (selector, *( (proc.stdout,proc.stderr) if proc is not None else ())):
                try:
                    stream.close()
                except BaseException as exc:
                    close_errors.append(type(exc).__name__+': '+str(exc))
            receipt.update(ended=time.monotonic(), close_errors=close_errors)
            if events is not None:
                events.append(receipt)
        require(not close_errors, 'HELPER_STREAM_CLOSE_FAILED')
        require(time.monotonic() <= until, 'HELPER_TOTAL_DEADLINE')


def protection_hold():
    """No logging/filesystem prerequisite; retain inherited FD despite errors."""
    while True:
        try:
            time.sleep(1)
        except BaseException:
            continue


def require_original_lock(lock):
    require(lock.get('device') == BINDING['expected_lock_device'] and
            lock.get('inode') == BINDING['expected_lock_inode'], 'ORIGINAL_LOCK_IDENTITY_MISMATCH')


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
    require(BINDING is not None and BINDING.get('launch_authorized') is True,
            'DISABLED_NO_SEPARATE_HOST_RELEASE')
    require(HERE == Path(BINDING['packet_root']), 'WRONG_BOUND_PACKET_ROOT')
    require(BINDING.get('project_commit') and BINDING.get('stage_authorized') is True,
            'INCOMPLETE_BOUND_PROJECT_OR_STAGE')
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
    retained, helper_events = 0, []
    pipes = []
    def helper(command, **kwargs):
        return bounded_run(command, tracker=tracker, events=helper_events,
                           deadline=started+1197, **kwargs)
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
        require_original_lock(lock)
        # Whole-lifetime ownership starts before any external preflight helper.
        subreaper()
        tracker = Processes()
        tracker.quiescent()
        def interrupted(signum, frame):
            raise RuntimeError('DRIVER_SIGNAL:'+str(signum))
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
            signal.signal(sig,interrupted)
        for base in BINDING['stop_roots']:
            for rel in ('STOP','research/agentic/STOP','research/v2/STOP'):
                require(not os.path.lexists(Path(base)/rel), 'STOP_PRESENT')
        source = Path(BINDING['scientific_source'])
        require(digest(source) == 'e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43', 'SCIENTIFIC_SOURCE_PIN')
        clean = helper(['/usr/bin/git','-C',BINDING['project_root'],'status','--porcelain'], timeout=5)
        require(clean.returncode == 0 and not clean.stdout and not clean.stderr, 'UNCLEAN_PROJECT')
        revision = helper(['/usr/bin/git','-C',BINDING['project_root'],'rev-parse','HEAD'], timeout=5)
        require(revision.returncode == 0 and revision.stdout.strip() == BINDING['project_commit'], 'PROJECT_COMMIT')
        for path in (root.parent, HERE, source.parent):
            require(free_bytes(path) >= 16*GIB, 'INITIAL_DISK_16_GIB')
        require(memavailable() >= 24*GIB, 'INITIAL_MEMAVAILABLE_24_GIB')
        root.mkdir(mode=0o700)
        for name in ('artifacts','cache','tmp','logs'):
            (root/name).mkdir()
        pre = {'lock':lock}
        idle_guard.check_idle(helper, os.getpid(), os.getppid(), pre)
        record('preflight.json', pre)
        env = environment(root, lock_fd)
        for name in ('neff','local','hlo','inductor','compiler','debug','xdg'):
            (root/'cache'/name).mkdir()
        (root/'tmp'/'home').mkdir()
        (root/'logs'/'metrics').mkdir()
        require(time.monotonic()-started <= 80, 'PREFLIGHT_RESERVE')
        tracker.quiescent()
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
        release_safe = False
        try:
            close_errors = []
            # All close failures are recorded in memory, never allowed to skip stop.
            for stream in (selector,monitor_log,*pipes,
                           *((child.stdout,child.stderr) if child is not None else ())):
                if stream is not None:
                    try:
                        stream.close()
                    except BaseException as exc:
                        close_errors.append(type(exc).__name__+': '+str(exc))
            containment_ok = not close_errors
            terminal['stream_close_errors'] = close_errors
            if tracker is not None:
                try:
                    now = time.monotonic()
                    terminal['termination'] = tracker.stop(min(now+60, started+1160), min(now+65, started+1165))
                    containment_ok = containment_ok and not terminal['termination']['survivors']
                except BaseException as exc:
                    containment_ok = False
                    terminal['containment_error'] = type(exc).__name__+': '+str(exc)
            post = {}
            try:
                require(containment_ok and lock_fd is not None and tracker is not None and
                        time.monotonic() < started+1167, 'POSTFLIGHT_TIME_OR_CONTAINMENT')
                post['lock'] = idle_guard.inherited_lock('/home/ubuntu/.cache/trainium_freshstart/device0.lock', HERE/'host_driver.py')
                require(post['lock']['fd'] == lock_fd, 'LOCK_CHANGED')
                require_original_lock(post['lock'])
                idle_guard.check_idle(helper,os.getpid(),os.getppid(),post)
                source_pins()
                post['status'] = 'IDLE_OBSERVED'
            except BaseException as exc:
                post['error'] = type(exc).__name__+': '+str(exc)
                containment_ok = False
            finally:
                # Postflight commands are children too: recensus/terminate/reap
                # AFTER the final helper even if that helper/check raised.
                try:
                    require(tracker is not None, 'NO_LIFETIME_TRACKER')
                    now = time.monotonic()
                    post['final_termination'] = tracker.stop(min(now+0.1,started+1198),started+1198)
                    require(not post['final_termination']['survivors'], 'POSTFLIGHT_SURVIVORS')
                    post['final_quiescence'] = tracker.quiescent()
                    require(time.monotonic() <= started+1198, 'FINAL_REAP_DEADLINE')
                except BaseException as exc:
                    containment_ok = False
                    post['final_containment_error'] = type(exc).__name__+': '+str(exc)
            if not containment_ok:
                terminal['status'] = 'CONTAINMENT_FAILED'
            elif terminal['status'] == 'PENDING_POSTFLIGHT':
                terminal['status'] = 'PARTIAL_DIAGNOSTIC_F0_F2_EXACT_OBSERVED'
            terminal.update(postflight=post, helpers=helper_events, elapsed=time.monotonic()-started)
            require(root.exists() and (root/'logs').exists(), 'TERMINAL_LOCATION_MISSING')
            record('terminal.json', terminal)
            require(time.monotonic() < started+1200, 'TERMINAL_TOTAL_DEADLINE')
            release_safe = containment_ok
        except BaseException:
            # ENOSPC, cap, JSON/write/flush/close and asynchronous signal errors
            # cannot bypass the hold. Do not depend on an emergency log write.
            release_safe = False
            # Unexpected finalization exceptions still get a containment attempt
            # before parking; this path itself has no evidence-write dependency.
            try:
                if tracker is not None:
                    now=time.monotonic()
                    tracker.stop(min(now+0.1,started+1198),started+1198)
                    tracker.quiescent()
            except BaseException:
                pass
        finally:
            if lock_fd is not None and not release_safe:
                protection_hold()
    return 0 if terminal['status'] == 'PARTIAL_DIAGNOSTIC_F0_F2_EXACT_OBSERVED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
