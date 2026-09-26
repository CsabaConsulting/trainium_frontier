"""Stdlib/Git-only mechanical checks; no tensors, host calls or scientific execution."""
import ast
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=HERE, text=True).strip())
COUNT = 0


def check(value, message):
    global COUNT
    COUNT += 1
    if not value:
        raise RuntimeError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=REPO)


def main():
    plan = json.loads((HERE / 'CPU_PLAN.disabled.json').read_text())
    proposal_raw = (HERE / 'EXPERIMENT_PROPOSAL.json').read_bytes()
    proposal = json.loads(proposal_raw)
    check(proposal_raw == blob('e9abc159', str((HERE / 'EXPERIMENT_PROPOSAL.json').relative_to(REPO))), 'PROPOSAL_EXACT')
    check(sha(proposal_raw) == plan['proposal_sha256'], 'PROPOSAL_SHA')
    for key in ('stage_authorized', 'launch_authorized', 'native_launch_authorized', 'source_adoption', 'submission_authorized', 'host_reserved'):
        check(plan[key] is False, 'DISABLED:' + key)
    check(plan['local_tensor_execution_prohibited'] is True and plan['cpu_only'] is True, 'CPU_ONLY')
    check(plan['proposed_host'] == 'trn21' and plan['expected_hostname'] == 'ip-172-31-34-85', 'HOST')
    check(plan['original_lock'] == {'device': 66305, 'inode': 596823}, 'ORIGINAL_LOCK')
    check(plan['device_lock'] == '/home/ubuntu/.cache/trainium_freshstart/device0.lock', 'COMMON_LOCK')
    check(plan['ordering'] == ['reference', 'candidate', 'standard-compare'], 'EXACT_PAIR_ONLY')
    check(plan['minimum_available_ram_gib'] == 64 and plan['minimum_free_disk_gib'] == 10.5, 'FLOORS')
    check(plan['additional_projected_footprint_bytes'] == 2 * 1024**3, 'ADDITIONAL_FOOTPRINT')
    check([plan[k] for k in ('steps', 'nproc', 'parameters', 'parameter_tensors', 'seed', 'max_train_seconds', 'timeout_seconds_per_arm')] == [3, 4, 113256724, 65, 42, 1800, 3600], 'PROFILE')
    check(plan['micro_batch'] == [2, 256] and plan['keep_checkpoints'] is True and plan['real_data'] is False, 'SYNTHETIC_PROFILE')
    root = plan['stage_root']
    check(root == '/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1', 'UNIQUE_ROOT')
    check(plan['prepare_dir'] == '/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2', 'REAL_PREPARE_DIR')
    for name in ('reference', 'candidate', 'g2'):
        entry = plan['sources'][name]
        check(sha(blob(entry['commit'], entry['git_path'])) == entry['sha256'], 'EXACT_SOURCE:' + name)
    check(plan['sources']['reference']['sha256'] == proposal['reference']['sha256'], 'REFERENCE_BINDING')
    check(plan['sources']['candidate']['sha256'] == proposal['candidate']['sha256'], 'CANDIDATE_BINDING')
    check(plan['sources']['g2']['sha256'] == '6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8', 'G2_PIN')
    for name in ('idle_guard', 'dispatch'):
        entry = plan['sources'][name]
        check(sha((REPO / entry['git_path']).read_bytes()) == entry['sha256'], 'UNCHANGED_SUPPORT:' + name)
    check(plan['sources']['prepare']['sha256'] == '5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb', 'PREPARE_PIN')
    ref_spec = json.loads(blob('e3e5f149', proposal['reference']['argv_source']))
    launch_raw = (HERE / 'candidate/launch-command.original.txt').read_bytes()
    check(launch_raw == blob(proposal['candidate']['commit'], proposal['candidate']['launch_command_file']), 'CANDIDATE_LAUNCH_BYTES')
    check(sha(launch_raw) == proposal['candidate']['launch_command_sha256'], 'CANDIDATE_LAUNCH_SHA')
    launch = shlex.split(launch_raw.decode())
    check(launch[:5] == ['NEURON_LOGICAL_NC_CONFIG=2', 'torchrun', '--standalone', '--nproc_per_node=4', 'train.py'], 'CANDIDATE_LAUNCH_PREFIX')
    for probe in plan['probes']:
        name = probe['name']
        raw = (HERE / probe['request_path']).read_bytes()
        req = json.loads(raw)
        check(sha(raw) == probe['request_sha256'], 'REQUEST_SHA:' + name)
        check(req['argv'] == (ref_spec['argv'] if name == 'reference' else launch[5:]), 'EXACT_SCIENCE_ARGV:' + name)
        expected = ['/home/ubuntu/workspace/native_venv/bin/python', '-B', root + '/preflight_dryrun.py', '--train-py', root + '/' + name + '/train.py', '--prepare-dir', plan['prepare_dir'], '--steps', '3', '--nproc', '4', '--micro-rows', '2', '--micro-tokens', '256', '--timeout', '3600', '--keep', '--out', root + '/' + name + '/g2.json', '--'] + req['argv']
        check(req['command'] == probe['command'] == expected, 'EXACT_COMMAND:' + name)
        check(sha(json.dumps(expected, separators=(',', ':')).encode()) == probe['command_sha256'], 'COMMAND_SHA:' + name)
        check(req['cwd'] == root + '/' + name and req['temporary_root'] == req['cwd'] + '/tmp', 'ARM_PATHS:' + name)
        check(req['source_sha256'] == plan['sources'][name]['sha256'], 'REQUEST_SOURCE:' + name)
        check(req['g2_tool_sha256'] == plan['sources']['g2']['sha256'] and req['prepare_sha256'] == plan['sources']['prepare']['sha256'], 'REQUEST_SUPPORT:' + name)
        check([req[k] for k in ('steps', 'nproc', 'micro_rows', 'micro_tokens', 'parameters', 'parameter_tensors', 'timeout_seconds')] == [3, 4, 2, 256, 113256724, 65, 3600], 'REQUEST_PROFILE:' + name)
        check(req['keep'] is True and req['cpu_only'] is True and req['trace_rmsprop'] is False, 'REQUEST_SCOPE:' + name)
    expected_compare = ['/home/ubuntu/workspace/native_venv/bin/python', '-B', root + '/preflight_dryrun.py', '--compare', root + '/reference/g2.json', root + '/candidate/g2.json']
    check(plan['comparison_command'] == expected_compare, 'ONLY_STANDARD_COMPARE')
    check(plan['expected_outer_command'] == ['/usr/bin/flock', '-n', plan['device_lock'], '/usr/bin/python3', '-I', '-B', root + '/host_driver.py'], 'ORIGINAL_LIFETIME_COMMAND')
    derivation = json.loads((HERE / 'MECHANICAL_DERIVATION.json').read_text())
    parent = blob(derivation['parent_commit'], derivation['parent_path'])
    check(sha(parent) == derivation['parent_sha256'], 'PARENT_DRIVER_SHA')
    expected_driver = parent.decode()
    for pair in derivation['literal_replacements']:
        check(pair['from'] in expected_driver, 'EXACT_TRANSFORM_PRESENT')
        expected_driver = expected_driver.replace(pair['from'], pair['to'])
    driver = (HERE / 'host_driver.py').read_bytes()
    check(driver == expected_driver.encode(), 'ENTIRE_DRIVER_LITERAL_TRANSFORM')
    check(sha(driver) == plan['driver_sha256'], 'DRIVER_SHA')
    tree = ast.parse(driver)
    assignments = {n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    pins = ast.literal_eval(assignments['PINS'])
    check(set(pins) == {'reference/request.json', 'candidate/request.json'}, 'NO_NEW_VERIFIER_SUPPORT')
    for name, digest in pins.items():
        check(sha((HERE / name).read_bytes()) == digest, 'DRIVER_REQUEST_PIN')
    check('validate_receipts.py' not in driver.decode() and 'compare_checkpoints.py' not in driver.decode(), 'NO_CHECKPOINT_ALGORITHM_IN_PAIR')
    check(driver.decode().count('pass_fds=(lock_fd,)') == 2, 'INHERITED_FD_G2_AND_COMPARE')
    check("observed['identical'] is True and observed['n_params_compared'] == 65" in driver.decode(), 'EXISTING_STRICT_STANDARD_RESULT')
    check('timeout does not itself prove' in plan['retained_standard_timeout_limitation'], 'TIMEOUT_LIMITATION_RETAINED')
    manifest = HERE / 'MANIFEST.sha256.json'
    if manifest.exists():
        for name, digest in json.loads(manifest.read_text())['files'].items():
            check(sha((HERE / name).read_bytes()) == digest, 'MANIFEST:' + name)
    print(json.dumps({'status': 'LOCAL_MECHANICAL_CHECKS_PASS', 'checks': COUNT, 'tensor_execution': False, 'host_calls': False, 'independent_review': False, 'launch_authorized': False}, sort_keys=True))


if __name__ == '__main__':
    main()
