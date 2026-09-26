"""Read-only raw-result authentication; not fresh review of the author's r2 repair."""
import base64
import datetime as dt
import hashlib
import json
from pathlib import Path
import stat
import subprocess

REPO = Path('/home/poojith-devan/Desktop/Trainium')
REF = '72087392'
ROOT = 'research/v2/operator/20260926-trn21-p-offline-decoder-r2/completed/'
PACKET = 'research/v2/proposals/20260926-p-offline-decoder-readiness-r2-trn21/'
checks = []

def require(ok, name):
    if not ok:
        raise RuntimeError(name)
    checks.append(name)

def read(path, ref=REF):
    return subprocess.check_output(['git', 'show', ref + ':' + path], cwd=REPO)

def sha(data):
    return hashlib.sha256(data).hexdigest()

bundle_data = read(ROOT + 'results-bundle.json')
post_data = read(ROOT + 'explicit-process-postflight.json')
require(sha(bundle_data) == '44d6f1abe2dee31f041f0bb529ce43d2d44eb05d76db185109cba693e2fe2038', 'frozen raw bundle')
require(sha(post_data) == '293679fdfbaee597b87706a01259431bd3a0d24ca4a32d779e9af3d319d1ecb7', 'frozen external postflight')
bundle, post = json.loads(bundle_data), json.loads(post_data)
files = {}
for name, record in bundle['files'].items():
    data = base64.b64decode(record['base64'], validate=True)
    require(len(data) == record['bytes'] and sha(data) == record['sha256'], 'raw size/SHA:' + name)
    files[name] = data
require(len(files) == 22 and sum(map(len, files.values())) == bundle['raw_bytes'] == 104923, 'all22/104923 raw bytes')
j = lambda name: json.loads(files[name])
proposal, binding = j('EXPERIMENT_PROPOSAL.json'), j('OFFLINE_BINDING.disabled.json')
for name in ('EXPERIMENT_PROPOSAL.json', 'OFFLINE_BINDING.disabled.json', 'MANIFEST.sha256.json'):
    require(files[name] == read(PACKET + name, '99d7c261'), 'original assigned packet identity:' + name)
preflight = j('preflight.json')
for name, pin in binding['runtime_sha256'].items():
    require(preflight['staged_hashes'][name] == pin, 'actual staged runtime pin:' + name)
require(preflight['host'] == binding['host'] == 'trn21', 'actual assigned host')
require(preflight['prior_r1_terminal']['error'] == 'RuntimeError: NEURON_LS_COMMAND_FAILED', 'r1 failure preserved')

for prefix in ('', 'post-'):
    address, idle = j(prefix + 'readiness-address-space.json'), j(prefix + 'under-lock-idle.json')
    require(address['before'] == address['restored'] == [512*1024**2,16*1024**3], prefix + 'parent AS restored')
    require(address['readiness_soft_bytes'] == 16*1024**3 and address['guard_error'] is None, prefix + '16GiB guard success')
    require(idle['status'] == 'PASS' and idle['ps']['returncode'] == 0 and idle['ps']['busy'] == [], prefix + 'fresh ps idle')
    require(idle['neuron_ls']['returncode'] == 0 and idle['neuron_ls']['process_pids'] == [] and idle['neuron_ls']['lnc'] == 2, prefix + 'installed neuron-ls success idle')
    lock = idle['lock']
    require(lock['verified_existing_exclusive_lock'] is True and (lock['device'],lock['inode'],lock['fd'],lock['driver_pid'],lock['parent_pid']) == (66305,596823,3,3101006,3101005), prefix + 'same original live lock')
version = j('version-process.json')
require(version['returncode'] == 0 and version['status'] == 'PROCESS_EXIT_ZERO_COVERAGE_UNKNOWN', 'actual version child success')
require(files['version.stdout'].decode().strip() == proposal['expected_version'] == 'neuron-explorer 2.32.0.498-b1f3998 built on 2026-08-10T05:09:49Z (commit b1f3998a)' and files['version.stderr'] == b'', 'exact version stdout/no stderr')

identities = j('input-identities.json')
expected = {r['path']: r['bytes'] for r in proposal['inputs']}
observed = identities['files']
require(len(expected) == len(observed) == 86 and {r['path'] for r in observed} == set(expected), 'full86 relative layout')
require(sum(p.endswith('.neff') for p in expected) == 80 and sum(expected.values()) == 4310546731, '80NEFF/4310546731 input bytes')
require(identities['source_root'] == proposal['source_root'] and identities['copied_root'] == binding['root'] + '/input/native', 'source/copy roots exact')
for row in observed:
    src, dst = row['source'], row['copy']
    require(src['st_size'] == dst['st_size'] == expected[row['path']] and src['sha256'] == dst['sha256'], 'observed copy/source hash+size:' + row['path'])
    require(stat.S_ISREG(dst['st_mode']) and stat.S_IMODE(dst['st_mode']) == 0o444 and dst['st_nlink'] == 1 and (src['st_dev'],src['st_ino']) != (dst['st_dev'],dst['st_ino']), 'independent readonly copy:' + row['path'])
require(identities['original_content_hashes_previously_known'] is False, 'first content hashes not fabricated historical pins')

launch, decoder, terminal, outer = j('decoder-launch.json'), j('decoder-process.json'), j('terminal.json'), j('outer-flock-exit.json')
require(launch['command'] == decoder['command'] == proposal['core_argv'], 'fixed decoder command')
require(launch['input_files'] == 86 and launch['supplied_neffs'] == 80, 'all86 supplied/not proof used')
require(launch['environment']['GOMAXPROCS'] == '1' and launch['selected_cpu'] == decoder['selected_cpu'] == 0, 'recorded single CPU/Go parallelism')
limits = launch['limits']
require(limits['address_space'] == 16*1024**3 and limits['file_size'] == 64*1024**2 and limits['scratch_bytes'] == 256*1024**2 and limits['wall_seconds'] == 300 and limits['grace'] == 10, 'unchanged decoder ceilings')
require(limits['retained_bytes'] + version['retained_bytes'] == 16*1024**2 and limits['cpu_seconds'] == 207, 'stricter residual output/CPU limits')
require(decoder['status'] == 'REFUSED_OR_INCOMPLETE' and decoder['error'] == 'RuntimeError: GENERATED_SCRATCH_CAP', 'precise scratch refusal')
require(decoder['scratch_peak_bytes'] == 271389612 and decoder['scratch_peak_bytes']-limits['scratch_bytes'] == decoder['scratch_peak_overshoot_bytes'] == 2954156, 'sampled overshoot arithmetic')
require(decoder['scratch_sample_interval_seconds'] == 0.1, 'sampled stop not hard quota')
require(decoder['observed_process_identities'] == {'3101016':92152355} and decoder['returncode'] == -15, 'exact decoder identity/TERM exit')
term = decoder['termination'][0]
require(term['signal'] == 15 and term['identities'] == [dict(pgid=3101016,pid=3101016,ppid=3101006,sid=3101016,start=92152355,state='R')], 'tracked TERM target exact')
require(decoder['termination'][1] == dict(identities=[],signal=9), 'no remaining SIGKILL target')
require(decoder['retained_bytes'] == decoder['observed_pipe_bytes'] == 0 and files['decoder.stdout'] == files['decoder.stderr'] == b'', 'no stdout/stderr or summary')
require('readonly-postflight.json' not in files, 'aborted source/copy/binary posthash NOT passed')
require(terminal['status'] == 'REFUSED_OR_INCOMPLETE' and terminal['error'] == 'RuntimeError: DECODER_FAILED_OR_LIMITED' and terminal['exit_code'] == outer['returncode'] == 1, 'failed terminal/outer preserved')
require(terminal['elapsed_seconds'] < 600 and terminal['observed_parent_plus_reaped_children_cpu_seconds'] < 300 and decoder['wall_seconds'] < 300, 'recorded wall/CPU bounds except disclosed scratch overshoot')
require(terminal['no_automatic_retry'] and terminal['no_automatic_next_job'] and not terminal['quality_evidence'], 'no retry/next/quality')
require(outer['command'] == binding['outer_command'], 'actual common flock command')

observed_pids = {3101004,3101005,3101006,3101012,3101016}
require(set(post['observed_pids']) == observed_pids and all(not row['exists'] for row in post['pid_rows']), 'all5 observed processes absent')
rows = post['all_process_identity_rows']
require(len(rows) == 258 and len({r['pid'] for r in rows}) == 258, 'full258 process identity census')
require(not any(r[k] in observed_pids for r in rows for k in ('pid','ppid','pgid','sid')), 'no matching PID/PPID/PGID/SID lineage')
require(not any(any(s in r['comm'].lower() for s in ('neuron','explor','walrus','torchrun')) for r in rows), 'no listed runtime/compiler/explorer names')
require(post['ps_rc'] == 0 and not post['ps_stderr'] and not post['related_rows'], 'external ps clean')
require(post['neuron_device_idle'] is True and len(post['neuron_ls']) == 1 and post['neuron_ls'][0]['neuron_processes'] == [] and not post['neuron_ls_stderr'], 'external device idle')
require(post['lock_identity'] == dict(device=66305,inode=596823) and post['lock_competitor_rc'] == 0, 'original lock free')
require(dt.datetime.fromisoformat(post['utc']) > dt.datetime.fromisoformat(outer['ended_utc']), 'external census after outer exit')
print(json.dumps(dict(verdict='AUTHENTICATED_REFUSED_SCRATCH_CAP_PARKED',checks=len(checks),checks_detail=checks,raw_files=22,raw_bytes=104923,input_identity_records=86,source_posthash_pass=False,decoded_summary=False,retry_authorized=False,role_disclosure='Outcome Analyst; also authored r2 repair, not independent source approval',raw_bundle_sha256=sha(bundle_data),external_postflight_sha256=sha(post_data)),indent=2,sort_keys=True))
