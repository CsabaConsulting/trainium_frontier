"""One fixed, disabled-until-assigned offline decoder under the original lifetime lock."""
import datetime as dt
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import sys
import time

ROOT = Path('/home/ubuntu/trainium_p_offline_summary_20260926_trn21_r2')
PROJECT = Path('/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2')
LOCK = '/home/ubuntu/.cache/trainium_freshstart/device0.lock'
PINS = {"EXPERIMENT_PROPOSAL.json":"efaeef0f914bf01023b417f9632ca88b47f5e136ee95eece5bacd4e5d646ec2a","decoder_support.py":"96aebc1e9a43843f57691e67053c75ecbe904f2cec6899688a9e3036c863efd9","idle_guard.py":"adafb54a98527570a76af19ec2785e68496bf1027c51c4127664435c8c963737","dispatch_driver.py":"c71474bc920f1bdba8f577d072497c4b94ca2c0318c29d39a37555da57ea041a"}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def record(name, value):
    with (ROOT / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def alarm_handler(signum, frame):
    raise RuntimeError('OVERALL_WALL_OR_PARENT_CPU_LIMIT')


def readiness_guard(guard, limits, receipt):
    """Only shared readiness subprocesses inherit the already-approved AS ceiling."""
    before = resource.getrlimit(resource.RLIMIT_AS)
    expected = (limits['parent_address_space_bytes'], limits['decoder_address_space_bytes'])
    if before != expected:
        raise RuntimeError('READINESS_PARENT_AS_UNEXPECTED')
    guard_error = None
    try:
        resource.setrlimit(resource.RLIMIT_AS, (expected[1], expected[1]))
        return guard.require_idle_under_inherited_lock(LOCK, ROOT / 'host_driver.py', receipt)
    except BaseException as exc:
        guard_error = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        resource.setrlimit(resource.RLIMIT_AS, before)
        restored = resource.getrlimit(resource.RLIMIT_AS)
        if restored != before:
            raise RuntimeError('READINESS_PARENT_AS_NOT_RESTORED')
        receipt('readiness-address-space.json',
                dict(before=list(before), readiness_soft_bytes=expected[1],
                     restored=list(restored), guard_error=guard_error,
                     go_16gib_compatibility='UNPROVEN; no automatic larger retry'))


def main():
    started = time.monotonic()
    deadline = started + 580
    terminal = dict(status='REFUSED_OR_INCOMPLETE', started_utc=utc(), quality_evidence=False,
                    coverage_unknown=True, no_automatic_retry=True, no_automatic_next_job=True)
    code = 1
    try:
        # No source/SDK/model import; authenticate these small staged files first.
        import hashlib
        if Path(__file__).absolute().parent != ROOT or socket.gethostname() != 'ip-172-31-34-85':
            raise RuntimeError('WRONG_PACKET_ROOT_OR_HOST')
        for name, expected in PINS.items():
            path = ROOT / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
                raise RuntimeError('SUPPORT_PATH_OR_SIZE:' + name)
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError('SUPPORT_SHA:' + name)
        helper = load('fixed_decoder_support', ROOT / 'decoder_support.py')
        helper.no_symlink(ROOT)
        plan = json.loads((ROOT / 'EXPERIMENT_PROPOSAL.json').read_text())
        source = Path(plan['source_root'])
        for base in (Path('/home/ubuntu'), PROJECT, source.parents[2], ROOT):
            for relative in ('STOP', 'research/agentic/STOP', 'research/v2/STOP'):
                helper.require(not os.path.lexists(base / relative), 'STOP:' + str(base / relative))
        limits = plan['limits']
        fs = os.statvfs(ROOT)
        helper.require(fs.f_bavail * fs.f_frsize >= limits['minimum_free_disk_bytes'], 'DISK_BELOW_12_GIB')
        mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        helper.require(int(mem['MemAvailable'].split()[0]) * 1024 >= limits['minimum_available_memory_bytes'], 'RAM_BELOW_24_GIB')
        for name in ('input', 'home', 'tmp', 'decoder.stdout', 'decoder.stderr', 'version.stdout', 'version.stderr'):
            helper.require(not os.path.lexists(ROOT / name), 'OUTPUT_ALREADY_EXISTS:' + name)
        cpu = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cpu})
        # Retain hard headroom until the child sets its own larger AS/CPU bounds.
        resource.setrlimit(resource.RLIMIT_AS, (limits['parent_address_space_bytes'], limits['decoder_address_space_bytes']))
        cpu_hard = resource.getrlimit(resource.RLIMIT_CPU)[1]
        helper.require(cpu_hard == resource.RLIM_INFINITY or cpu_hard >= 235, 'INHERITED_CPU_LIMIT_UNSUPPORTED')
        resource.setrlimit(resource.RLIMIT_CPU, (55, cpu_hard))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.signal(signal.SIGXCPU, alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.monotonic()))
        guard = load('bound_idle_guard', ROOT / 'idle_guard.py')
        lock_fd = readiness_guard(guard, limits, record)
        # The same authenticated lock remains held during every copy/hash/decode/postflight.
        binary = Path(plan['binary'])
        binary_before = helper.stream_hash(binary, deadline, 256 * 1024**2)
        (ROOT / 'home').mkdir()
        (ROOT / 'tmp').mkdir()
        env = {'HOME': str(ROOT / 'home'), 'TMPDIR': str(ROOT / 'tmp'),
               'XDG_CACHE_HOME': str(ROOT / 'tmp/cache'), 'XDG_DATA_HOME': str(ROOT / 'tmp/data'),
               'PATH': '/usr/bin:/bin:/opt/aws/neuron/bin', 'LC_ALL': 'C',
               'GOMAXPROCS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
               'TRN_EXP_LABEL': plan['label']}
        run_limits = dict(address_space=limits['decoder_address_space_bytes'], cpu_seconds=5,
                          file_size=limits['decoder_file_size_bytes'], wall_seconds=5, grace=limits['term_grace_seconds'],
                          retained_bytes=65536, scratch_bytes=limits['generated_scratch_stop_bytes'],
                          interval=limits['monitor_interval_seconds'], entry_cap=limits['max_scratch_entries'])
        version = helper.run_bounded([str(binary), '--version'], env, ROOT, set(), lock_fd,
                                     run_limits, deadline, ROOT / 'version.stdout', ROOT / 'version.stderr')
        record('version-process.json', version)
        helper.require(version['status'] == 'PROCESS_EXIT_ZERO_COVERAGE_UNKNOWN', 'VERSION_PROCESS_FAILED')
        helper.require((ROOT / 'version.stdout').read_text().strip() == plan['expected_version'] and
                       (ROOT / 'version.stderr').stat().st_size == 0, 'BINARY_VERSION_CHANGED')
        copies = helper.copy_inputs(source, ROOT / 'input/native', plan['inputs'], deadline)
        record('input-identities.json', dict(source_root=str(source), copied_root=str(ROOT / 'input/native'),
                                            original_content_hashes_previously_known=False, files=copies))
        excluded = {ROOT / 'input/native' / row['path']: row['bytes'] for row in plan['inputs']}
        previous_children = resource.getrusage(resource.RUSAGE_CHILDREN)
        # Reserve the parent's hard60 and32 CPU seconds for the two final15s guard calls.
        decoder_cpu_limit = min(235, math.floor(300 - 60 - 32 - previous_children.ru_utime - previous_children.ru_stime))
        helper.require(decoder_cpu_limit > 0, 'NO_REMAINING_CPU_BUDGET')
        run_limits.update(cpu_seconds=decoder_cpu_limit, wall_seconds=limits['decoder_wall_seconds'],
                          retained_bytes=limits['combined_retained_stdout_stderr_bytes'] - version['retained_bytes'],
                          parent_final_cpu_hard=limits['parent_cpu_hard_seconds'])
        record('decoder-launch.json', dict(utc=utc(), command=plan['core_argv'], environment=env,
                                          binary_identity=binary_before, selected_cpu=cpu, limits=run_limits,
                                          overall_work_deadline_seconds=580, overall_budget_seconds=600,
                                          cpu_subbudgets='parent<=60; decoder gets floor(300-60-32-actual preceding child CPU), at most235; final guard reserve32',
                                          input_files=86, supplied_neffs=80, coverage_unknown=True))
        result = helper.run_bounded(plan['core_argv'], env, ROOT, excluded, lock_fd, run_limits,
                                    deadline, ROOT / 'decoder.stdout', ROOT / 'decoder.stderr')
        record('decoder-process.json', result)
        helper.require(result['status'] == 'PROCESS_EXIT_ZERO_COVERAGE_UNKNOWN', 'DECODER_FAILED_OR_LIMITED')
        helper.post_inputs(source, ROOT / 'input/native', copies, deadline)
        helper.exact_layout(source, plan['inputs'])
        helper.exact_layout(ROOT / 'input/native', plan['inputs'])
        helper.require(helper.stream_hash(binary, deadline, 256 * 1024**2) == binary_before, 'BINARY_CHANGED')
        record('readonly-postflight.json', dict(source_and_copy_all86_hashes_unchanged=True,
                                                binary_unchanged=True, no_original_files_written=True))
        terminal['status'] = 'OFFLINE_PROCESS_EXITED_PENDING_INDEPENDENT_MAPPING_REVIEW'
        code = 0
    except BaseException as exc:
        terminal['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        # Always preserve bounded exit evidence. No decoder success or all-model coverage inferred.
        if 'guard' in locals() and 'lock_fd' in locals() and time.monotonic() - started < 565:
            try:
                signal.setitimer(signal.ITIMER_REAL, max(0.001, started + 598 - time.monotonic()))
                post_fd = readiness_guard(guard, limits,
                                  lambda name, value: record('post-' + name, value))
                if post_fd != lock_fd:
                    raise RuntimeError('LOCK_FD_CHANGED')
                terminal['postflight_idle'] = True
            except BaseException as exc:
                terminal['postflight_idle'] = False
                terminal['postflight_error'] = type(exc).__name__ + ': ' + str(exc)
                terminal['status'] = 'REFUSED_OR_INCOMPLETE'
                code = 1
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
        else:
            terminal['postflight_idle'] = False
            terminal['postflight_error'] = 'Not enough remaining bounded time or initial lock gate incomplete; Operator must establish quiescence before any future work.'
            terminal['status'] = 'REFUSED_OR_INCOMPLETE'
            code = 1
        usage = resource.getrusage(resource.RUSAGE_SELF)
        children_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu_total = usage.ru_utime + usage.ru_stime + children_usage.ru_utime + children_usage.ru_stime
        if time.monotonic() - started > 600 or cpu_total > 300:
            terminal['status'] = 'REFUSED_OR_INCOMPLETE'
            terminal['measured_budget_limit_exceeded'] = True
            code = 1
        terminal.update(ended_utc=utc(), elapsed_seconds=time.monotonic() - started,
                        parent_cpu_seconds=usage.ru_utime + usage.ru_stime, parent_max_rss_kib=usage.ru_maxrss,
                        observed_parent_plus_reaped_children_cpu_seconds=cpu_total,
                        exit_code=code, source_or_training_changes=False)
        record('terminal.json', terminal)
    return code


if __name__ == '__main__':
    sys.exit(main())
