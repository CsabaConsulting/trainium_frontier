"""Stdlib/Git mechanical checks only; no source import, tensors, hosts or launch."""
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
    br = (HERE / 'NATIVE_BINDING.disabled.json').read_bytes()
    b = json.loads(br)
    pr = (HERE / 'EXPERIMENT_PROPOSAL.json').read_bytes()
    p = json.loads(pr)
    d = json.loads((HERE / 'MECHANICAL_DERIVATION.json').read_text())
    check(pr == blob(b['proposal_commit'], str((HERE / 'EXPERIMENT_PROPOSAL.json').relative_to(REPO))), 'EXACT_CORRECTED_PROPOSAL')
    check(b['proposal_commit'] == '68e3d5325c730b722ffd74f133fb6e6bd511c4a7', 'PROPOSAL_COMMIT')
    check(sha(pr) == b['proposal_sha256'], 'PROPOSAL_SHA')
    check(d == b['derivation'], 'DERIVATION_IDENTITY')
    parent_raw = blob(d['parent_commit'], d['parent_binding_path'])
    pb = json.loads(parent_raw)
    check(sha(parent_raw) == d['parent_binding_sha256'], 'PARENT_BINDING_SHA')
    parent_driver = blob(d['parent_commit'], d['parent_driver_path'])
    check(sha(parent_driver) == d['parent_driver_sha256'] == pb['driver_sha256'], 'PARENT_DRIVER_SHA')
    check(d['literal_replacements'] == [{'from': pb['label'], 'to': p['label']}], 'ONE_LABEL_SUBSTITUTION_ONLY')
    expected_driver = parent_driver.decode().replace(pb['label'], p['label']).encode()
    driver = (HERE / 'host_driver.py').read_bytes()
    check(driver == expected_driver, 'ENTIRE_DRIVER_BYTES')
    check(sha(driver) == b['driver_sha256'], 'DRIVER_SHA')
    ast.parse(driver)
    source = blob(b['source_commit'], b['source_git_path'])
    check(sha(source) == b['source_sha256'] == p['source_sha256'] == p['base_sha256'], 'EXACT_SOURCE')
    check(b['source_sha256'] == '36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb', 'ADOPTED36_ONLY')
    check(b['kind'] == b['arm_id'] == p['kind'] == p['arm_id'] == 'control', 'GENUINE_PROSPECTIVE_CONTROL')
    check(b['base_sha256'] == b['source_sha256'], 'NEW_BASE_LINEAGE')
    check(b['host'] == p['host'] == pb['host'], 'SAME_HOST')
    host = b['host']
    check(host in ('trn21', 'trn22'), 'HOST_SET')
    check(b['expected_hostname'] == {'trn21': 'ip-172-31-34-85', 'trn22': 'ip-172-31-32-253'}[host], 'HOSTNAME')
    check(b['original_lock'] == p['resource_gates']['original_lock'], 'PROPOSAL_ORIGINAL_LOCK')
    check(b['original_lock']['device'] == 66305 and b['original_lock']['inode'] == {'trn21': 596823, 'trn22': 832589}[host], 'LOCK_IDENTITY')
    check(b['original_lock']['path'] == b['device_lock'] == '/home/ubuntu/.cache/trainium_freshstart/device0.lock', 'COMMON_LOCK')
    for key in ('stage_authorized', 'launch_authorized', 'native_launch_authorized', 'operator_assigned', 'source_adoption', 'submission_authorized', 'host_reserved'):
        check(b[key] is False, 'DISABLED:' + key)
    check(b['no_auto_next_job'] is True and b['no_scientific_change'] is True and b['results'] is None, 'NO_AUTO_RESULT_OR_NEXT')
    check(b['historical_parent'] == p['historical_parent'], 'HISTORICAL_ROLE_PRESERVED')
    check(b['priority'] == p['priority'], 'PRIORITY_PRESERVED')
    ad = b['adopted_base']
    adr = blob(ad['commit'], ad['path'])
    adoption = json.loads(adr)
    check(sha(adr) == ad['sha256'], 'ADOPTION_SHA')
    check(adoption['status'] == 'ADOPTED_CONFIRMED_RESEARCH_BASE', 'EXISTING_ADOPTION')
    check(b['training_command'] == pb['training_command'], 'IDENTICAL_PARENT_SCIENCE')
    check(b['training_command'][4:] == p['argv'] == adoption['argv'], 'EXACT_ADOPTED_ARGV')
    check(b['training_command'][:4] == ['/home/ubuntu/workspace/native_venv/bin/torchrun', '--standalone', '--nproc_per_node=4', 'train.py'], 'LAUNCH_SHAPE')
    argv = b['training_command']
    for flag, value in [('--matrix-lr', '0.015'), ('--seed', '42'), ('--max-train-seconds', '1800'), ('--num-steps', '1000000'), ('--seq-len', '1024'), ('--pack-factor', '32')]:
        check(argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value, 'SCIENCE_FLAG:' + flag)
    check('--recursive-schedule' not in argv and '--startup-adjusted-schedule' in argv, 'NO_CHILD_SUBSTITUTION')
    launch = (HERE / 'package/launch-command.txt').read_bytes()
    check(launch.decode().strip() == shlex.join(argv), 'LAUNCH_TEXT_ARGV')
    check(launch == blob(d['parent_commit'], str(Path(d['parent_binding_path']).parent / 'package/launch-command.txt')), 'PARENT_LAUNCH_BYTES')
    for name, digest in b['package_support_sha256'].items():
        check(sha((HERE / 'package' / name).read_bytes()) == digest, 'PACKAGE_PIN:' + name)
    prov = json.loads((HERE / 'package/provenance.json').read_text())
    check(prov['kind'] == prov['arm_id'] == 'control' and prov['base_sha256'] == b['base_sha256'], 'PROVENANCE_CONTROL')
    check(prov['label'] == b['label'] and prov['train_sha256'] == b['source_sha256'], 'PROVENANCE_IDENTITY')
    for key in ('root', 'package', 'sandbox', 'eval_sandbox', 'driver_path', 'command', 'evaluation_command', 'outer_command', 'dispatch_command', 'training_environment', 'evaluation_environment'):
        expected = json.loads(json.dumps(pb[key]).replace(pb['label'], p['label']))
        check(b[key] == expected, 'ONLY_OPERATIONAL_LABEL_REBIND:' + key)
    check(b['root'] != pb['root'] and b['sandbox'] == b['root'] + '/train' and b['eval_sandbox'] == b['root'] + '/eval', 'UNIQUE_PATHS')
    cache_keys = ('TORCH_NEURONX_NEFF_CACHE_DIR', 'TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR', 'TORCH_NEURONX_HLO_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'NEURON_COMPILE_CACHE_DIR', 'NEURON_COMPILE_CACHE_URL', 'TORCH_NEURONX_DEBUG_DIR')
    for env_key in ('training_environment', 'evaluation_environment'):
        for key in cache_keys:
            check(b[env_key][key] == pb[env_key][key], 'UNCHANGED_HOST_CACHE:' + env_key + ':' + key)
        check(b[env_key]['NEURON_LOGICAL_NC_CONFIG'] == '2', 'LNC2')
        check(b[env_key]['NEURON_COMPETITION_R1_SEED'] == '42', 'SEED_ENV')
    check(b['training_environment']['NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS'] == '1800', 'BUDGET_ENV')
    check(b['budgets'] == pb['budgets'], 'UNCHANGED_ALL_BUDGETS')
    check(b['evaluation'] == pb['evaluation'], 'UNCHANGED_EVAL')
    check(b['evaluation']['tokens'] == 20971520 and b['evaluation']['seq_len'] == 1024 and b['evaluation']['batch_size'] == 1, 'FRESH20M_SHAPE')
    check(b['evaluation']['fresh_process'] and b['evaluation']['same_lifetime_lock'] and b['evaluation']['new_guard_before_eval'], 'EVAL_LOCK_AND_PROCESS')
    for key in ('prepare_path', 'prepare_sha256', 'harness_path', 'harness_sha256', 'source_gate', 'idle_guard_sha256', 'dispatch_driver_sha256'):
        check(b[key] == pb[key], 'UNCHANGED_SUPPORT:' + key)
    check(sha(blob('00f2ca7f79696e65ce52458f074dd13b871da90c', 'research/scratch/submission_rehearsal.py')) == b['harness_sha256'], 'PINNED_HARNESS_BLOB')
    for name, key in [('idle_guard.py', 'idle_guard_sha256'), ('dispatch_driver.py', 'dispatch_driver_sha256')]:
        check(sha((REPO / 'research/v2/operator/20260926-common-host-idle-guard' / name).read_bytes()) == b[key], 'SUPPORT_BLOB:' + name)
    check(b['parent_cold_gate_historical'] == pb['cold_gate'], 'OLD_GATE_METADATA_NOT_REWRITTEN')
    check(b['resource_gates']['disk_floor_gib'] == 10.5 and b['resource_gates']['projected_footprint_required'] is True, 'DISK_PLUS_FOOTPRINT')
    check(b['cache_policy'] == p['cache_policy'], 'NO_STABLE_PATH_SHORTCUT')
    manifest = HERE / 'MANIFEST.sha256.json'
    if manifest.exists():
        for name, digest in json.loads(manifest.read_text())['files'].items():
            check(sha((HERE / name).read_bytes()) == digest, 'MANIFEST:' + name)
    print(json.dumps({'status': 'LOCAL_MECHANICAL_BINDING_PASS', 'host': host, 'checks': COUNT, 'host_calls': False, 'tensor_execution': False, 'source_execution': False, 'independent_review': False, 'launch_authorized': False}, sort_keys=True))


if __name__ == '__main__':
    main()
