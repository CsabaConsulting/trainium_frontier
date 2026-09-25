#!/usr/bin/env python3
"""Local stdlib-only packet checks. Never import or execute any bound workload."""
import ast
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
checks = []


def require(condition, name):
    if not condition:
        raise RuntimeError(name)
    checks.append(name)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def blob(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=REPO)


def literal_assignments(tree):
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'Path':
                value = value.args[0]
            try:
                values[node.targets[0].id] = ast.literal_eval(value)
            except (ValueError, TypeError):
                pass
    return values


def main():
    binding = json.loads((HERE / 'CT_BINDING.disabled.json').read_text())
    proposal = json.loads((HERE / 'EXPERIMENT_PROPOSAL.json').read_text())
    for key in ('stage_authorized', 'launch_authorized', 'source_adoption', 'full_run_automatic'):
        require(binding[key] is False and proposal[key] is False, 'disabled_' + key)
    require(binding['host_reserved'] is False and proposal['operator_assignment'] == 'NOT_ASSIGNED', 'no_host_assignment')
    require(len(binding['pending_gates']) == 3, 'three_outstanding_binding_assignment_remote_gates')
    require(binding['target_startup_seconds'] == 870 and binding['cold_decision']['maximum_seconds'] == 900, 'architecture_target870_maximum900_not_authority')
    source = blob(binding['source_commit'], binding['source_git_path'])
    require(sha(source) == binding['source_sha256'] == proposal['source_sha256'], 'frozen_candidate_bytes')
    reference = blob(proposal['control']['source_commit'], binding['source_git_path'])
    require(sha(reference) == proposal['control']['source_sha256'], 'frozen_reference_bytes')
    require(sha((REPO / 'prepare.py').read_bytes()) == binding['prepare_sha256'], 'unchanged_prepare')
    harness = blob(binding['harness_source_commit'], binding['harness_source_git_path'])
    require(sha(harness) == binding['harness_sha256'], 'unchanged_rehearsal')
    for name, file in (('idle_guard', 'idle_guard.py'), ('dispatch_driver', 'dispatch_driver.py')):
        data = (REPO / 'research/v2/operator/20260926-common-host-idle-guard' / file).read_bytes()
        require(sha(data) == binding[name + '_sha256'], 'unchanged_' + name)
    driver = (HERE / 'host_driver.py').read_bytes()
    require(sha(driver) == binding['driver_sha256'] == proposal['driver_sha256'], 'driver_pin')
    derivation = binding['driver_derivation']
    original = blob(derivation['commit'], derivation['path'])
    require(sha(original) == derivation['sha256'], 'guarded_architecture_driver_pin')
    original_values = literal_assignments(ast.parse(original))
    expected = original.replace(original_values['LABEL'].encode(), binding['label'].encode())
    expected = expected.replace(original_values['SOURCE_SHA'].encode(), binding['source_sha256'].encode())
    require(driver == expected, 'literal_only_driver_port')
    tree = ast.parse(driver)
    values = literal_assignments(tree)
    for var, key in (('ROOT', 'root'), ('PREP', 'prepare_path'), ('HARNESS', 'harness_path'),
                     ('SOURCE_SHA', 'source_sha256'), ('LABEL', 'label'), ('COMMAND', 'command'),
                     ('ENV', 'training_environment')):
        require(values[var] == binding[key], 'driver_binding_' + var)
    guard_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == 'require_idle_under_inherited_lock']
    launches = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name) and n.func.value.id == 'subprocess' and n.func.attr == 'call']
    require(len(guard_calls) == len(launches) == 1 and guard_calls[0].lineno < launches[0].lineno, 'guard_precedes_single_job')
    require(any(k.arg == 'pass_fds' and ast.unparse(k.value) == '(lock_fd,)' for k in launches[0].keywords), 'inherited_lock_passed')
    require(binding['device_lock'] == '/home/ubuntu/.cache/trainium_freshstart/device0.lock', 'same_persistent_lock')
    require(binding['outer_command'] == ['/usr/bin/flock', '-n', binding['device_lock'], '/usr/bin/python3', '-I', '-B', binding['driver_path']], 'exact_outer_command')
    require(binding['detached_command'] == ['/usr/bin/python3', '-I', '-B', binding['root'] + '/dispatch_driver.py'], 'exact_dispatch_command')
    for key in ('command', 'training_command', 'outer_command', 'detached_command'):
        require(sha(json.dumps(binding[key], separators=(',', ':'), ensure_ascii=True).encode()) == binding[key + '_sha256'], 'argv_hash_' + key)
    for file, expected_hash in binding['package_files'].items():
        require(sha((HERE / file).read_bytes()) == expected_hash, 'package_hash_' + file)
    provenance = json.loads((HERE / 'package/provenance.json').read_text())
    launch_text = (HERE / 'package/launch-command.txt').read_text().strip()
    require(launch_text == provenance['launch_command'], 'launch_text_provenance')
    require(shlex.split(launch_text) == binding['training_command'] == proposal['argv'], 'launch_argv_binding')
    require(provenance['train_sha256'] == binding['source_sha256'], 'provenance_source')
    base = json.loads((REPO / 'research/v2/operator/20260926-ct-architecture-startup-only/CT_BINDING.disabled.json').read_text())
    require(binding['base_training_command'] == base['training_command'], 'genuine_base_cold_argv')
    require(binding['training_command'] == base['training_command'] + ['--startup-adjusted-schedule'], 'single_on_flag')
    cpu_on = json.loads((REPO / 'research/v2/proposals/20260926-startup-adjusted-exact-cpu-trn21/on/request.json').read_text())
    require(cpu_on['source_sha256'] == binding['source_sha256'], 'accepted_cpu_source_same')
    expected_argv = cpu_on['argv'].copy()
    require(expected_argv[expected_argv.index('--num-steps') + 1] == '1000000', 'original_production_step_cap')
    expected_argv[expected_argv.index('--num-steps') + 1] = '5'
    require(binding['training_command'][4:] == expected_argv, 'accepted_cpu_config_only_cold_step_cap_differs')
    env = binding['training_environment']
    require(env['NEURON_CC_FLAGS'] == '--optlevel=1 --auto-cast matmult --auto-cast-type bf16', 'unchanged_compiler_flags')
    require(env['NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS'] == '1800' and env['NEURON_COMPETITION_R1_SEED'] == '42', 'budget_seed')
    require('PYTHONOPTIMIZE' not in env and '-O' not in binding['command'], 'no_optimized_bound_harness')
    require(binding['command'][-3:] == ['--mode', 'train', '--fresh-caches'], 'train_only_fresh_caches')
    injected = binding['harness_injected_environment']
    cache_values = [v for k, v in injected.items() if k.endswith('_DIR') and not k.startswith('NEURON_COMPETITION')]
    require(len(cache_values) == len(set(cache_values)) == 6, 'six_distinct_cache_roots')
    require(all(p.startswith(binding['sandbox'] + '/caches/') for p in cache_values), 'private_cache_roots')
    require(env['NEURON_COMPILE_CACHE_URL'] == injected['NEURON_COMPILE_CACHE_DIR'], 'compiler_url_dir_match')
    require(env['NEURON_COMPETITION_R1_CACHE_DIR'] == '/home/ubuntu/.cache/neuron_competition_r1', 'unchanged_data_cache')
    require(binding['no_eval'] is True and binding['no_automatic_full'] is True, 'no_followon_work')
    source_tree = ast.parse(source)
    refs = sorted(n.lineno for n in ast.walk(source_tree) if isinstance(n, ast.Attribute)
                  and isinstance(n.value, ast.Name) and n.value.id == 'args' and n.attr == 'num_steps')
    reference_tree = ast.parse(reference)
    reference_refs = [n for n in ast.walk(reference_tree) if isinstance(n, ast.Attribute)
                      and isinstance(n.value, ast.Name) and n.value.id == 'args' and n.attr == 'num_steps']
    require(len(refs) == len(reference_refs), 'unchanged_num_steps_reference_count')
    require('--recursive-schedule' not in binding['training_command'], 'no_recursive_stack')
    for flag, value in [('--num-steps', '5'), ('--max-train-seconds', '1800'), ('--seed', '42'),
                        ('--matrix-lr', '0.015'), ('--seq-len', '1024'), ('--pack-factor', '32')]:
        require(binding['training_command'][binding['training_command'].index(flag) + 1] == value, 'exact_' + flag)
    require(binding['expected_hostname'] == 'ip-172-31-32-253' and binding['host'] == 'trn22', 'exact_trn22_hostname')
    require('Operator' in binding['host_identity'] and 'no internal hostname assertion' in binding['host_identity'], 'host_gate_location_disclosed')
    require('PPID/PGID/SID' in binding['postflight'] and 'neuron-ls' in binding['postflight'], 'explicit_postflight')
    lr = next(n for n in ast.walk(source_tree) if isinstance(n, ast.FunctionDef) and n.name == 'lr_multiplier')
    require(not any(isinstance(n, ast.Attribute) and n.attr == 'num_steps' for n in ast.walk(lr)), 'no_step_cap_in_lr_formula')
    lines = source.decode().splitlines()
    positions = {name: next(i + 1 for i, line in enumerate(lines) if text in line) for name, text in {
        'first_step_t0': '        t0 = time.monotonic()',
        'initial_refresh': '        if step == 0 and args.initial_cache_refresh:',
        'first_forward': '            loss = model(x, y)',
        'boundary': '        record_step_boundary(step)',
        'step_cap': '        step_cap_hit = step + 1 >= args.num_steps',
    }.items()}
    require(positions['first_step_t0'] < positions['initial_refresh'] < positions['first_forward'], 'refresh_inside_counted_first_step')
    require(positions['boundary'] < positions['step_cap'], 'cap_after_boundary')
    historical = {}
    for arm in ('base', 'architecture'):
        file = REPO / ('research/v2/operator/20260926-ct-' + arm + '-startup-only/completed/train/rehearsal.json')
        receipt = json.loads(file.read_text())
        observed = receipt['observations']
        require(receipt['returncode'] == 0 and [r['step'] for r in observed] == list(range(5)), arm + '_historical_five_boundaries')
        fifth = observed[4]['observed_seconds']
        charge = receipt['wall_seconds'] - min(fifth, 900.0)
        require(math.isfinite(charge) and 0 <= charge <= 1800, arm + '_historical_charge_valid')
        require(charge == receipt['charged_through_process_exit'], arm + '_historical_charge_formula')
        historical[arm] = {'fifth_observed_seconds': fifth, 'wall_seconds': receipt['wall_seconds'], 'charged_through_exit': charge}
    print(json.dumps({'status': 'PASS_STDLIB_PACKET_ONLY', 'checks_count': len(checks), 'checks': checks,
                      'num_steps_reference_lines': refs, 'source_positions': positions,
                      'historical_interface_fixtures_not_new_startup_clock_evidence': historical,
                      'source_imported': False, 'tensor_jobs': 0, 'host_actions': 0,
                      'native_correctness': 'NOT_TESTED', 'launch_authorized': False}, indent=2))


if __name__ == '__main__':
    main()
