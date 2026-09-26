"""Local stdlib binding checks only; never imports a candidate or accesses hosts."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

D = Path(__file__).resolve().parent
REPO = Path(subprocess.check_output(['git', '-C', str(D), 'rev-parse', '--show-toplevel'], text=True).strip())
PARENT = '1d2f1e97775df2073e347b12f326b84bed9c460b'
PARENT_PATH = 'research/v2/proposals/20260926-startup-adjusted-full-trn22'
COLD = 'fa4f8953917e0e076382a26d5ec0cbc5b8f8c0ff'
COLD_PATH = 'research/v2/proposals/20260926-clock36-recursive-cold-five-trn22'
CHECKS = []


def require(value, label):
    if not value:
        raise ValueError(label)
    CHECKS.append(label)


def git(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=REPO)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def constants(raw):
    result = {}
    for node in ast.parse(raw).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'Path':
                value = value.args[0]
            result[node.targets[0].id] = ast.literal_eval(value)
    return result


parent = json.loads(git(PARENT, PARENT_PATH + '/NATIVE_BINDING.disabled.json'))
cold = json.loads(git(COLD, COLD_PATH + '/CT_BINDING.disabled.json'))
proposal_raw = (D / 'EXPERIMENT_PROPOSAL.json').read_bytes()
proposal = json.loads(proposal_raw)
parent_driver = git(PARENT, PARENT_PATH + '/host_driver.py')
binding = json.loads((D / 'NATIVE_BINDING.disabled.json').read_bytes())
driver = (D / 'host_driver.py').read_bytes()


def validate(b, raw):
    require(b['source_sha256'] == proposal['source_sha256'] == cold['source_sha256'], 'exact_child_source')
    require(b['source_commit'] == proposal['source_commit'], 'source_commit')
    require(all(b[k] is False for k in ('stage_authorized', 'launch_authorized', 'native_launch_authorized', 'submission_authorized', 'source_adoption', 'host_reserved')), 'disabled_flags')
    expected = list(cold['training_command'])
    index = expected.index('--num-steps') + 1
    require(expected[index] == '5', 'cold_five')
    expected[index] = '1000000'
    require(b['training_command'] == expected == proposal['argv'], 'single_num_steps_change')
    require(b['root'] != cold['root'] and b['root'] != parent['root'], 'unique_output_root')
    require(b['label'] == proposal['label'] and b['root'].endswith('/' + b['label'] + '-packet'), 'label_root_binding')
    require(b['budgets'] == parent['budgets'] and b['evaluation'] == parent['evaluation'], 'unchanged_budget_eval')
    require(b['evaluation']['tokens'] == 20971520 and b['evaluation']['seq_len'] == 1024 and b['evaluation']['batch_size'] == 1, 'public20m_b1_1024')
    require(b['original_lock']['device'] == 66305 and b['original_lock']['inode'] == 832589 and b['device_lock'] == parent['device_lock'], 'original_lock')
    require(b['expected_hostname'] == 'ip-172-31-32-253', 'exact_host')
    require(b['cold_gate']['observed_uncapped_fifth_seconds'] is None and b['cold_gate']['five_steps_exit0'] is None and b['cold_gate']['raw_commit'] is None, 'cold_outcome_not_invented')
    require(b['cold_gate']['threshold_seconds'] == 900 and b['cold_gate']['target_seconds'] == 870, 'uncapped_cold_threshold')
    require(b['cold_gate']['label'] == cold['label'] and b['cold_gate']['source_sha256'] == cold['source_sha256'], 'exact_child_cold_not_parent')
    transformed = parent_driver.decode()
    for row in b['derivation']['literal_replacements']:
        require(row['from'] in transformed, 'literal_from_present')
        transformed = transformed.replace(row['from'], row['to'])
    require(transformed.encode() == raw and sha(raw) == b['driver_sha256'], 'entire_driver_literal_identity')
    require(sha(parent_driver) == b['derivation']['parent_driver_sha256'] == '27b4ac4ac93b7487a57b1a3b2d20c08152342d258cc84a276d65244fcc6c3067', 'proven_parent_driver')
    old_ast = ast.parse(parent_driver)
    new_ast = ast.parse(raw)
    for old, new in zip(old_ast.body, new_ast.body):
        if not isinstance(old, ast.Assign):
            require(ast.dump(old) == ast.dump(new), 'unchanged_driver_executable_flow')
    values = constants(raw)
    for key, name in [('ROOT', 'root'), ('PREP', 'prepare_path'), ('HARNESS', 'harness_path'), ('LABEL', 'label'), ('SOURCE_SHA', 'source_sha256'), ('PREP_SHA', 'prepare_sha256'), ('HARNESS_SHA', 'harness_sha256'), ('TRAIN_COMMAND', 'command'), ('EVAL_COMMAND', 'evaluation_command'), ('TRAIN_ENV', 'training_environment'), ('EVAL_ENV', 'evaluation_environment')]:
        require(values[key] == b[name], 'driver_binding_' + key)
    cache_keys = ('TORCH_NEURONX_NEFF_CACHE_DIR', 'TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR', 'TORCH_NEURONX_HLO_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'NEURON_COMPILE_CACHE_DIR', 'TORCH_NEURONX_DEBUG_DIR')
    for key in cache_keys:
        require(b['training_environment'][key] == cold['harness_injected_environment'][key], 'reuse_child_cache_' + key)
        require(b['evaluation_environment'][key] == parent['evaluation_environment'][key], 'unchanged_eval_cache_' + key)
    require(b['training_environment']['NEURON_COMPILE_CACHE_URL'] == b['training_environment']['NEURON_COMPILE_CACHE_DIR'], 'compiler_url_equals_directory')
    require('-O' not in b['command'] and '-O' not in b['evaluation_command'] and '-O' not in b['outer_command'] and '--fresh-caches' not in b['command'], 'normal_python_warm_research')
    require(b['decision_comparator'] == proposal['control'] and b['formal_control'] is None, 'preserve_nonformal_parent_role')
    require(b['no_auto_next_job'] is True, 'no_auto_next')


validate(binding, driver)
require(proposal_raw == git(binding['proposal_commit'], str((D / 'EXPERIMENT_PROPOSAL.json').relative_to(REPO))) and sha(proposal_raw) == binding['proposal_sha256'], 'proposal_bytes_frozen')
require(sha(git(binding['source_commit'], binding['source_git_path'])) == binding['source_sha256'], 'actual_scientific_git_blob')
require(sha(git(binding['harness_source_commit'], binding['harness_source_git_path'])) == binding['harness_sha256'], 'actual_unchanged_harness_blob')
require((D / 'package/launch-command.txt').read_text() == shlex.join(binding['training_command']) + '\n', 'exact_launch_text')
for name, digest in binding['package_support_sha256'].items():
    require(sha((D / 'package' / name).read_bytes()) == digest, 'package_pin_' + name)
for name, key in [('idle_guard.py', 'idle_guard_sha256'), ('dispatch_driver.py', 'dispatch_driver_sha256')]:
    require(sha((REPO / 'research/v2/operator/20260926-common-host-idle-guard' / name).read_bytes()) == binding[key], 'unchanged_shared_' + name)
negative_passes = []
for case in ('launch', 'source', 'steps', 'lr', 'cache', 'eval', 'lock', 'cold'):
    mutated = copy.deepcopy(binding)
    if case == 'launch': mutated['launch_authorized'] = True
    if case == 'source': mutated['source_sha256'] = '0' * 64
    if case == 'steps': mutated['training_command'][mutated['training_command'].index('--num-steps') + 1] = '5'
    if case == 'lr': mutated['training_command'][mutated['training_command'].index('--matrix-lr') + 1] = '0.012'
    if case == 'cache': mutated['training_environment']['TORCH_NEURONX_NEFF_CACHE_DIR'] = '/tmp/unrelated'
    if case == 'eval': mutated['evaluation']['tokens'] = 2000000
    if case == 'lock': mutated['original_lock']['inode'] += 1
    if case == 'cold': mutated['cold_gate']['observed_uncapped_fifth_seconds'] = 1
    try:
        validate(mutated, driver)
    except ValueError as error:
        negative_passes.append({'case': case, 'rejected': str(error)})
    else:
        raise RuntimeError('NEGATIVE_ACCEPTED:' + case)
print(json.dumps({'status': 'LOCAL_STDLIB_BINDING_CHECKS_PASS', 'checks_executed': len(CHECKS), 'negative_cases': negative_passes, 'driver_sha256': sha(driver), 'binding_sha256': sha((D / 'NATIVE_BINDING.disabled.json').read_bytes()), 'host_calls': 0, 'tensor_or_source_imports': 0, 'launch_authority': False}, indent=2))
