"""Stdlib-only receipt/binding audit; never import candidate or host driver."""
import ast
import datetime as dt
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
P = ROOT / 'research/v2/operator/20260926-trn22-arch-initial-cache-cpu-r1'
R = P / 'results'
N = P / 'native'
OLD = ROOT / 'research/v2/operator/20260925-trn22-l2-93ae-cpu-r1'
checks = []

def require(ok, name):
    if not ok:
        raise RuntimeError(name)
    checks.append(name)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def j(path):
    return json.loads(path.read_text())

plan = j(P / 'cpu-g2-plan.json')
retrieval = j(R / 'operator-retrieval.json')
for name, meta in retrieval['files'].items():
    require(sha(R / name) == meta['sha256'] and (R / name).stat().st_size == meta['bytes'], 'hash:' + name)
reference_path = ROOT / plan['authenticated_existing_reference']['local_path']
require(sha(reference_path) == plan['authenticated_existing_reference']['sha256'], 'authenticated original reference')
reference = j(reference_path)
summaries = {}
for name, flag in [('candidate-off', '--no-initial-cache-refresh'), ('candidate-on', '--initial-cache-refresh')]:
    probe = next(p for p in plan['probes'] if p['name'] == name)
    folder = R / name
    receipt = j(folder / 'g2.json')
    terminal = j(folder / 'terminal.json')
    launch = j(folder / 'launch.json')
    require((folder / 'request.json').read_bytes() == (P / name / 'request.json').read_bytes(), name + ': exact request')
    require(receipt['argv'] == probe['argv'] == reference['argv'] + [flag], name + ': exact single flag')
    require(launch['command'] == probe['command'] and launch['source_sha256'] == receipt['train_py_sha256'] == probe['source_sha256'], name + ': source/command binding')
    require(receipt['verdict'] == terminal['verdict'] == 'PASS' and terminal['exit_code'] == receipt['torchrun_rc'] == 0 and not receipt['timed_out'], name + ': successful exits')
    require(receipt['steps'] == 3 and receipt['micro_batch'] == [2, 256] and receipt['nproc'] == 4, name + ': exact reduced-token CPU scope')
    require(receipt['prepare'] == reference['prepare'] and receipt['torch'] == reference['torch'] and receipt['python'] == reference['python'] and receipt['shims'] == reference['shims'], name + ': same runtime and shims')
    require([r['rank'] for r in receipt['ranks']] == [0, 1, 2, 3], name + ': all ranks')
    for rank in receipt['ranks']:
        raw_paths = list(folder.glob('tmp/*/rank' + str(rank['rank']) + '.json'))
        require(len(raw_paths) == 1 and j(raw_paths[0]) == rank, name + ': raw aggregate identity rank ' + str(rank['rank']))
        require(rank['status'] == 'PASS' and rank['n_params'] == rank['n_changed'] == 65 and not rank['cross_rank_mismatches'], name + ': all 65 updated consistently rank ' + str(rank['rank']))
        require(len(rank['losses']) == 3 and all(math.isfinite(x) for x in rank['losses']) and rank['losses'] == reference['ranks'][rank['rank']]['losses'], name + ': three identical losses rank ' + str(rank['rank']))
    r0 = receipt['ranks'][0]
    require(len(r0['final_digests']) == 65 and r0['final_digests'] == reference['ranks'][0]['final_digests'] and receipt['plan_sha256'] == reference['plan_sha256'], name + ': exact final digests and owner plan')
    require(r0['checkpoint_check'] == reference['ranks'][0]['checkpoint_check'], name + ': strict deterministic checkpoint reload')
    compare = subprocess.run([sys.executable, '-B', str(ROOT / 'workflow/tools/preflight_dryrun.py'), '--compare', str(reference_path), str(folder / 'g2.json')], text=True, capture_output=True)
    result = json.loads(compare.stdout)
    require(compare.returncode == 0 and result['identical'] and result['n_params_compared'] == 65, name + ': independent unchanged comparator')
    archived_compare = j(R / ('compare-original-' + name + '.json'))
    require(all(result[k] == archived_compare[k] for k in result if k not in ('a', 'b')), name + ': archived comparator agrees')
    require(j(R / ('compare-original-' + name + '-exit.json'))['exit_code'] == 0, name + ': original comparator exited zero')
    summaries[name] = {'receipt_sha256': sha(folder / 'g2.json'), 'elapsed_seconds': receipt['elapsed_s'], 'rank0_losses': r0['losses'], 'terminal_utc': terminal['utc']}
require(dt.datetime.fromisoformat(j(R / 'candidate-off/terminal.json')['utc']) < dt.datetime.fromisoformat(j(R / 'candidate-on/operator-launch.json')['utc']), 'serial OFF then ON')
require(not retrieval['active_processes'] and retrieval['device_idle'], 'postflight quiescence')
ckpts = retrieval['checkpoints']
require(len(ckpts) == 2 and all(c['bytes'] == 453048594 and c['sha256'] == '2e3d8e442e8c7ddfbb29308787fcd6f0ccba9f8c3f3bc051331f7a2b2a01fd3b' and not c['downloaded'] for c in ckpts), 'OFF ON complete checkpoint hash identity')

binding = j(N / 'native-binding.disabled.json')
driver = (N / 'host_driver.py').read_text()
old_driver = (OLD / 'native/host_driver.py').read_text()
require(sha(N / 'host_driver.py') == binding['driver_sha256'], 'native driver hash')
require(not binding['launch_authorized'] and not binding['stage_authorized'] and binding['independent_acceptance'] is None, 'native remains disabled')
require(binding['command'][4:] == plan['probes'][1]['argv'] and binding['source_sha256'] == plan['probes'][1]['source_sha256'], 'native exact source ON argv')
require(binding['outer_command'][:3] == ['/usr/bin/flock', '-n', plan['device_lock']] and '-O' not in binding['outer_command'], 'native shared lock and assertions retained')
def normalized(src):
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'ROOT', 'TRAIN_SHA', 'LABEL', 'CACHE', 'EVAL_CACHE', 'FLAGS'} for t in node.targets):
            node.value = ast.Constant('BOUND_LITERAL')
    # The one installed-SDK cache routing addition is allowed, not a model change.
    for node in ast.walk(tree):
        if isinstance(node, ast.List):
            node.elts = [v for v in node.elts if not (isinstance(v, ast.Tuple) and len(v.elts) == 2 and isinstance(v.elts[0], ast.Constant) and v.elts[0].value == 'NEURON_COMPILE_CACHE_URL')]
    return ast.dump(tree, include_attributes=False)
require(normalized(driver) == normalized(old_driver), 'driver binding literals plus compiler URL only')
tree = ast.parse(driver)
class FakePath(PurePosixPath):
    def is_dir(self):
        return True
scope = {'Path': FakePath}
selected = [n for n in tree.body if isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id in {'ROOT', 'PREP', 'TRAIN_SHA', 'PREP_SHA', 'LABEL', 'VENV', 'CACHE', 'EVAL_CACHE', 'FLAGS'} for t in n.targets)]
selected += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'environment']
exec(compile(ast.Module(body=selected, type_ignores=[]), 'native_environment_AST_only', 'exec'), scope)
env = scope['environment'](scope['CACHE'])
env.update({'NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS': '1800', 'NEURON_COMPETITION_R1_OUT_DIR': str(scope['ROOT'] / 'out'), 'TORCH_NEURONX_ENABLED_METRIC_TABLES': 'graph_stats', 'TORCH_NEURONX_METRICS_DIR': str(scope['ROOT'] / 'metrics-training')})
require(env == binding['training_environment'], 'exact evaluated training environment binding')
eval_env = scope['environment'](scope['EVAL_CACHE'])
require('TORCH_NEURONX_METRICS_DIR' not in eval_env and 'NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS' not in eval_env, 'eval does not inherit train metrics or budget')
require(env['NEURON_COMPILE_CACHE_URL'] == env['NEURON_COMPILE_CACHE_DIR'] and eval_env['NEURON_COMPILE_CACHE_URL'] == eval_env['NEURON_COMPILE_CACHE_DIR'], 'both compiler cache aliases redirected')
require(len(binding['cold_start']['paths']) == 12 and len(set(binding['cold_start']['paths'])) == 12 and binding['cold_start']['cache_paths_must_be_new_and_empty'], '12 unique planned fresh cache roots')
require(all(p.startswith(binding['cold_start']['stage_root'] + '/caches/') for p in binding['cold_start']['paths']), 'private cache roots')
require('--cache_dir' not in env['NEURON_CC_FLAGS'] and env['NEURON_CC_FLAGS'] == '--optlevel=1 --auto-cast matmult --auto-cast-type bf16', 'compiler flags unchanged')
require(env['NEURON_COMPETITION_R1_CACHE_DIR'] == '/home/ubuntu/.cache/neuron_competition_r1', 'data cache unchanged')
print(json.dumps({'verdict': 'PASS_GPT_TECHNICAL_CPU_AND_DISABLED_BINDING_ONLY', 'cross_family_gate': 'UNMET_PROVIDER_REFUSAL', 'launch_authorized_by_review': False, 'check_count': len(checks), 'checks': checks, 'cpu_summaries': summaries, 'native_binding_sha256': sha(N / 'native-binding.disabled.json'), 'native_driver_sha256': sha(N / 'host_driver.py'), 'checkpoint_sha256': ckpts[0]['sha256'], 'limits': 'Full 8x1024 parameters, synthetic 2x256 per rank, 4 CPU ranks, 3 steps, no compile, fake clock; no native/cold/performance/adoption proof'}, indent=2))
