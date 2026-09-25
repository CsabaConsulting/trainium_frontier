"""Independent frozen-packet audit; tiny stdlib fixtures, no SDK/host/trace use."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import time
import types

PACKET = 'research/v2/proposals/20260926-p-offline-decoder-trn21/'
DEFAULT = '8429c1a2033e46936a40a7be4195419724971e48'
p = argparse.ArgumentParser()
p.add_argument('--repo', default='/home/poojith-devan/Desktop/Trainium')
p.add_argument('--commit', default=DEFAULT)
p.add_argument('--expect-fixed', action='store_true')
a = p.parse_args()
checks = []


def need(ok, name):
    if not ok:
        raise RuntimeError(name)
    checks.append(name)


def raw(path):
    return subprocess.check_output(['git', 'show', a.commit + ':' + path], cwd=a.repo)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def refusal(call, fragment):
    try:
        call()
    except Exception as exc:
        need(fragment in str(exc), 'negative:' + fragment)
    else:
        raise RuntimeError('negative accepted:' + fragment)


manifest = json.loads(raw(PACKET + 'MANIFEST.sha256.json'))
for name, expected in manifest['files'].items():
    need(digest(raw(PACKET + name)) == expected, 'manifest:' + name)
plan = json.loads(raw(PACKET + 'EXPERIMENT_PROPOSAL.json'))
binding = json.loads(raw(PACKET + 'OFFLINE_BINDING.disabled.json'))
for key in ('stage_authorized', 'launch_authorized', 'host_reserved'):
    need(plan[key] is False and binding[key] is False, 'disabled:' + key)
for key in ('host', 'hostname', 'label', 'root', 'source_root', 'core_argv'):
    need(plan[key] == binding[key], 'binding:' + key)
need(binding['host'] == 'trn21' and binding['hostname'] == 'ip-172-31-34-85', 'host')
need(binding['core_argv'] == [plan['binary'], 'view', '--session-dir', plan['root'] + '/input/native',
                            '--output-format', 'summary-json', '--ignore-nc-buf-usage',
                            '--ignore-instruction-hierarchy'], 'fixed argv')
need(binding['outer_command'] == ['/usr/bin/flock', '-n', '/home/ubuntu/.cache/trainium_freshstart/device0.lock',
                                 '/usr/bin/python3', '-I', '-B', plan['root'] + '/host_driver.py'], 'same flock')
for name, expected in binding['runtime_sha256'].items():
    path = PACKET + name if name not in ('idle_guard.py', 'dispatch_driver.py') else 'research/v2/operator/20260926-common-host-idle-guard/' + name
    need(digest(raw(path)) == expected, 'runtime:' + name)
inventory = subprocess.check_output(['git', 'show', plan['source_inventory_commit'] + ':' + plan['source_inventory_path']], cwd=a.repo)
need(digest(inventory) == plan['source_inventory_sha256'], 'inventory pin')
prefix = plan['source_root'] + '/'
expected_inputs = sorted([dict(path=x['path'][len(prefix):], bytes=x['bytes'])
                         for x in json.loads(inventory)['profile_companions'] if x['path'].startswith(prefix)], key=lambda x:x['path'])
need(expected_inputs == sorted(plan['inputs'], key=lambda x:x['path']), 'entire archived layout')
need(len(expected_inputs) == 86 and len({x['path'] for x in expected_inputs}) == 86, '86 unique inputs')
need(sum(x['path'].endswith('.neff') for x in expected_inputs) == 80, '80 NEFFs')
need(sum(x['bytes'] for x in expected_inputs) == 4310546731, 'input total bytes')
limits_expected = dict(overall_wall_seconds=600, work_deadline_seconds=580, cleanup_reserve_seconds=20,
                       decoder_wall_seconds=300, term_grace_seconds=10, cpu_seconds_total_ceiling=300,
                       parent_cpu_hard_seconds=60, decoder_cpu_hard_seconds=240, one_cpu_affinity=True,
                       GOMAXPROCS='1', decoder_address_space_bytes=16*1024**3,
                       parent_address_space_bytes=512*1024**2, decoder_file_size_bytes=64*1024**2,
                       combined_retained_stdout_stderr_bytes=16*1024**2,
                       generated_scratch_stop_bytes=256*1024**2, monitor_interval_seconds=.1,
                       max_scratch_entries=4096, minimum_available_memory_bytes=24*1024**3,
                       minimum_free_disk_bytes=12*1024**3)
need(plan['limits'] == binding['bounds'] == limits_expected, 'exact ceilings')
driver = raw(PACKET + 'host_driver.py').decode()
need(driver.count('require_idle_under_inherited_lock(') == 2, 'initial and final shared guard')
need("pass_fds=(lock_fd,)" in raw(PACKET + 'decoder_support.py').decode(), 'decoder inherits live fd')
need("decoder_cpu_limit = min(235, math.floor(300 - 60 - 32" in driver, 'stricter cumulative CPU reserve')
h = types.ModuleType('independent_exact_helper')
exec(compile(raw(PACKET + 'decoder_support.py'), 'frozen:' + a.commit + '/decoder_support.py', 'exec'), h.__dict__)
observations = {}


def fixture(script, **overrides):
    with tempfile.TemporaryDirectory(prefix='independent-p-boundary-') as directory:
        root = Path(directory)
        with (root / 'fixture.lock').open('w') as lock:
            limits = dict(address_space=128*1024**2, cpu_seconds=2, file_size=65536,
                          wall_seconds=2, grace=.1, retained_bytes=8192, scratch_bytes=262144,
                          interval=.1, entry_cap=128)
            limits.update(overrides)
            value = h.run_bounded(['/usr/bin/python3', '-I', '-B', '-c', script],
                                  {'PATH':'/usr/bin:/bin', 'GOMAXPROCS':'1'}, root, set(),
                                  lock.fileno(), limits, time.monotonic()+5,
                                  root/'stdout', root/'stderr')
            value['actual_retained_file_bytes'] = (root/'stdout').stat().st_size + (root/'stderr').stat().st_size
            value['stdout_text'] = (root/'stdout').read_text()
            value['stderr_text'] = (root/'stderr').read_text()
            need(not h.owned_descendants(h.proc_rows(), -1, value['observed_process_identities'], os.getpid()),
                 'all fixture descendants quiescent')
            return value


with tempfile.TemporaryDirectory(prefix='independent-p-input-layout-') as directory:
    root = Path(directory)
    source, target = root/'source', root/'assembled'
    source.mkdir()
    entries = []
    for i, item in enumerate(expected_inputs):
        path = source/item['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        content = ('fixture-' + str(i)).encode()
        path.write_bytes(content)
        entries.append(dict(path=item['path'], bytes=len(content)))
    before = {x['path']:h.signature(source/x['path']) for x in entries}
    copies = h.copy_inputs(source, target, entries, time.monotonic()+5)
    h.post_inputs(source, target, copies, time.monotonic()+5)
    need(len(copies) == 86, 'tiny complete 86-file copy')
    for row in copies:
        need(row['source']['sha256'] == row['copy']['sha256'], 'copy digest:' + row['path'])
        need(row['copy']['st_nlink'] == 1 and row['copy']['st_ino'] != row['source']['st_ino'], 'independent inode:' + row['path'])
        need(h.signature(source/row['path']) == before[row['path']], 'original untouched:' + row['path'])
    need(all(not p.stat().st_mode & 0o222 for p in target.rglob('*')), 'assembled readonly modes')
    (source/'extra').write_bytes(b'x')
    refusal(lambda:h.exact_layout(source, entries), 'UNEXPECTED_INPUT_ENTRY')
    (source/'extra').unlink()
    refusal(lambda:h.exact_layout(source, [dict(path='../outside', bytes=1)]), 'UNSAFE_RELATIVE_PATH')
    victim = source/entries[0]['path']
    victim.unlink()
    victim.symlink_to(source/entries[1]['path'])
    refusal(lambda:h.exact_layout(source, entries), 'SYMLINK')
    for path in target.rglob('*'):
        path.chmod(0o700 if path.is_dir() else 0o600)
    target.chmod(0o700)

observations['resource_probe'] = fixture('import json,os,resource; print(json.dumps({"affinity":list(os.sched_getaffinity(0)),"as":resource.getrlimit(resource.RLIMIT_AS),"cpu":resource.getrlimit(resource.RLIMIT_CPU),"file":resource.getrlimit(resource.RLIMIT_FSIZE),"gomax":os.environ["GOMAXPROCS"]}))')
r = observations['resource_probe']
need(r['status'] == 'PROCESS_EXIT_ZERO_COVERAGE_UNKNOWN', 'tiny ordinary child accepted')
v = json.loads(r['stdout_text'])
need(len(v['affinity']) == 1 and v['as'] == [128*1024**2]*2 and v['cpu'] == [2,2] and v['file'] == [65536,65536] and v['gomax'] == '1', 'actual child resource reference')
observations['combined_pipe_negative'] = fixture('import os,time; os.write(1,b"a"*1700); os.write(2,b"b"*1700); time.sleep(30)', retained_bytes=2048)
r = observations['combined_pipe_negative']
need(r['status'] == 'REFUSED_OR_INCOMPLETE' and 'COMBINED_OUTPUT_CAP' in r['error'], 'both streams combined refusal')
need(r['retained_bytes'] == r['actual_retained_file_bytes'] == 2048 and r['observed_pipe_bytes'] > 2048, 'exact retained cap')
observations['late_detached_child'] = []
for _ in range(3):
    r = fixture('import os,time\nos.close(1); os.close(2); time.sleep(.04)\npid=os.fork()\nif pid: os._exit(0)\nos.setsid(); time.sleep(30)')
    observations['late_detached_child'].append(r)
late = observations['late_detached_child']
need(all(len(x['observed_process_identities']) >= 2 for x in late), 'late descendant genuinely observed')
false_success = [x for x in late if x['status'] == 'PROCESS_EXIT_ZERO_COVERAGE_UNKNOWN']
if a.expect_fixed:
    need(not false_success and all(x['status'] == 'REFUSED_OR_INCOMPLETE' for x in late), 'late descendant fail closed')
else:
    need(bool(false_success), 'original final-census classification defect reproduced')
print(json.dumps(dict(schema='independent-p-decoder-review-v1', commit=a.commit,
                      disposition='PASS_MECHANICS_ONLY' if a.expect_fixed else 'HOLD_FINAL_DESCENDANT_CLASSIFICATION',
                      checks=len(checks), check_names=checks,
                      source_hashes={name:digest(raw(PACKET+name)) for name in ('host_driver.py','decoder_support.py','OFFLINE_BINDING.disabled.json','MANIFEST.sha256.json')},
                      observations=observations, no_host_sdk_trace_tensors=True,
                      no_stage_or_launch_authority=True), indent=2, sort_keys=True))
