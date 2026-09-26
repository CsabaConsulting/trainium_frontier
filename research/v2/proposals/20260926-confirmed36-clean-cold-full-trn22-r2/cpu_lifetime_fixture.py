"""Harmless bounded CPU faults only. Never imports training/prepare/evaluation."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def leaf(directory, mode):
    directory = Path(directory)
    if mode == 'grandchild':
        os.setsid()
        if os.fork() == 0:
            os.setsid()
            return leaf(directory, 'sleep')
    if mode in ('grandchild', 'sleep'):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    lock = (directory/'device0.lock').stat()
    same = []
    for entry in Path('/proc/self/fd').iterdir():
        try:
            st = os.fstat(int(entry.name))
            if (st.st_dev, st.st_ino) == (lock.st_dev, lock.st_ino):
                same.append(int(entry.name))
        except OSError:
            pass
    ident = module('guard_leaf', HERE/'lifetime_guard.py').identity(os.getpid())
    marker=directory/('leaf-' + str(os.getpid()) + '.tmp')
    marker.write_text(json.dumps(dict(ident, lock_fds=same)))
    marker.rename(marker.with_suffix('.json'))
    # Every fixture child is bounded even if the tested old wrapper loses it.
    time.sleep(1.8 if mode in ('grandchild', 'sleep') else 0.15)
    return 0


def harness(directory, mode):
    directory = Path(directory)
    # Execute the exact Popen statement in the immutable protected harness.
    raw = subprocess.check_output(['git', 'show', '00f2ca7f79696e65ce52458f074dd13b871da90c:research/scratch/submission_rehearsal.py'], cwd=HERE)
    node = next(n for n in ast.walk(ast.parse(raw)) if isinstance(n, ast.Assign)
                and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                and n.value.func.attr == 'Popen')
    child_mode = 'grandchild' if mode in ('timeout', 'harness-crash', 'signal', 'interrupt', 'driver-exception', 'audit-fail') else 'short'
    command = ['/usr/bin/python3', '-I', '-B', str(Path(__file__).resolve()), '--leaf', str(directory), child_mode]
    ns = dict(subprocess=subprocess, command=command, sandbox=directory, env={'PATH':'/usr/bin:/bin'})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<exact-harness-Popen>', 'exec', optimize=0), ns)
    if mode == 'harness-crash':
        # Leave enough time for the setsid/double-fork fixture to start.
        time.sleep(0.2)
        os.kill(os.getpid(), signal.SIGKILL)
    rc = ns['process'].wait()
    return 7 if mode == 'train-fail' else 9 if mode == 'eval-fail' else rc


def driver(directory, mode):
    directory = Path(directory)
    idle = module('idle', HERE.parents[1]/'operator/20260926-common-host-idle-guard/idle_guard.py')
    lock_path = directory/'device0.lock'
    fd = idle.inherited_lock(lock_path, Path(__file__).resolve())['fd']
    # The fixture's exact outer command has no argv after its path, like production.
    life = module('lifetime', HERE/'lifetime_guard.py')
    st = lock_path.stat()
    expected = (st.st_dev, st.st_ino + int(mode == 'wrong-inode'))
    command = ['/usr/bin/timeout', '--foreground', '--signal=TERM', '--kill-after=0.1s',
               '0.4s' if mode in ('old-timeout', 'timeout') else '4s',
               '/usr/bin/python3', '-I', '-B', str(Path(__file__).resolve()), '--harness', str(directory),
               'timeout' if mode == 'old-timeout' else 'normal' if mode == 'eval-fail' else mode]
    if mode == 'old-timeout':
        return subprocess.call(command, pass_fds=(fd,), env={'PATH':'/usr/bin:/bin'})
    guardian = life.Guardian(fd, lock_path, expected, directory/'events.jsonl', grace=0.2, kill_wait=0.5)
    (directory/'guardian.json').write_text(json.dumps(life.identity(os.getpid())))
    result = 1
    try:
        if mode == 'preflight-refuse':
            raise RuntimeError('INITIAL_PREFLIGHT_REFUSED_NO_JOB')
        if mode in ('driver-exception', 'audit-fail'):
            event = guardian.event
            def injected(name, **fields):
                event(name, **fields)
                if name == 'phase_spawned':
                    time.sleep(0.2)
                    if mode == 'driver-exception':
                        raise RuntimeError('INJECTED_DRIVER_CRASH')
                    guardian.receipt.close()
                    guardian.receipt = open('/dev/full','w',buffering=1)  # Real ENOSPC on event writes.
            guardian.event = injected
        with (directory/'phase.log').open('x') as log:
            result = guardian.run(command, cwd=directory, env={'PATH':'/usr/bin:/bin'}, out=log)
            if result == 0:
                guardian.check_cancelled()
                (directory/'between-phases').touch()
                time.sleep(0.1)
                (directory/'eval-started').touch()
                command[-1] = 'eval-fail' if mode == 'eval-fail' else 'normal'
                result = guardian.run(command, cwd=directory, env={'PATH':'/usr/bin:/bin'}, out=log)
    except BaseException as exc:
        (directory/'error.txt').write_text(type(exc).__name__ + ': ' + str(exc))
        result = 1
    finally:
        (directory/'underlying-result.txt').write_text(str(result))
        def postflight():
            (directory/'postflight-entered').touch()
            time.sleep(0.15)
            if mode == 'postflight-fail':
                # Contain a helper which escaped its failed postflight caller too.
                subprocess.Popen(['/usr/bin/python3','-I','-B',str(Path(__file__).resolve()),
                                  '--leaf',str(directory),'sleep'])
                time.sleep(0.1)
                raise RuntimeError('INJECTED_POSTFLIGHT_FAILURE')
        if mode == 'quiescence-unknown':
            guardian.reap = lambda: False
        guardian.finish(postflight)
    return result if result >= 0 else 128 - result


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--leaf':
        sys.exit(leaf(*sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == '--harness':
        sys.exit(harness(*sys.argv[2:]))
    directory = Path.cwd()
    sys.exit(driver(directory, (directory/'case.txt').read_text()))
