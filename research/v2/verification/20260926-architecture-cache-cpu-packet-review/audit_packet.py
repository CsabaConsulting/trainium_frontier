"""Read-only stdlib audit; never imports the candidate or invokes the driver."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
P = ROOT / 'research/v2/operator/20260926-trn22-arch-initial-cache-cpu-r1'
checks = []

def check(ok, name):
    if not ok:
        raise RuntimeError(name)
    checks.append(name)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def obj(path):
    return json.loads(path.read_text())

def git_bytes(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT)

for line in (P / 'SHA256SUMS').read_text().splitlines():
    expected, name = line.split(maxsplit=1)
    check(sha((P / name).read_bytes()) == expected, 'manifest:' + name)
plan = obj(P / 'cpu-g2-plan.json')
sources = obj(P / 'sources.json')
for name in ('candidate', 'g2_tool', 'proposal'):
    x = sources[name]
    check(sha(git_bytes(x['git_commit'], x['git_path'])) == x['sha256'], 'git:' + name)
check(sources['baseline'] == plan['authenticated_existing_reference'], 'baseline binding')
refmeta = sources['baseline']
ref = obj(ROOT / refmeta['local_path'])
check(sha((ROOT / refmeta['local_path']).read_bytes()) == refmeta['sha256'], 'reference hash')
check(ref['train_py_sha256'] == refmeta['source_sha256'], 'reference source')
check(ref['argv'] == refmeta['profile']['argv'], 'reference argv')
check(ref['verdict'] == 'PASS' and ref['torchrun_rc'] == 0 and not ref['timed_out'], 'reference exits')
check(ref['nproc'] == 4 and ref['steps'] == 3 and ref['micro_batch'] == [2, 256], 'reference physical shape')
check(ref['prepare']['sha256'] == sources['trusted_prepare']['sha256'] and not ref['prepare']['stub'], 'reference real prepare')
check([r['rank'] for r in ref['ranks']] == [0, 1, 2, 3], 'reference ranks')
check(all(r['status'] == 'PASS' and r['n_params'] == 65 and r['n_changed'] == 65 and len(r['losses']) == 3 for r in ref['ranks']), 'reference updates')
check(len(ref['ranks'][0]['final_digests']) == 65 and ref['ranks'][0]['checkpoint_check']['ok'], 'reference state and reload')
check(not plan['cpu_launch_authorized'] and not plan['native_launch_authorized'], 'no self authorization')
check(plan['run_serially'] and plan['expected_model_parameters'] == 113256724 and plan['expected_parameter_tensors'] == 65, 'model and serial binding')
template = (ROOT / sources['driver_template']['path']).read_text()
check(sha(template.encode()) == sources['driver_template']['sha256'], 'driver template hash')
def normalized_driver(src):
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ROOT' for t in node.targets):
            node.value = ast.Constant('ROOT_PER_PROBE')
    return ast.dump(tree, include_attributes=False)
for probe, flag in zip(plan['probes'], ('--no-initial-cache-refresh', '--initial-cache-refresh'), strict=True):
    folder = P / probe['name']
    request = obj(folder / 'request.json')
    check(all(probe[k] == v for k, v in request.items()), probe['name'] + ': request binding')
    check(sha((folder / 'request.json').read_bytes()) == probe['request_sha256'], probe['name'] + ': request hash')
    check(sha((folder / 'cpu_driver.py').read_bytes()) == probe['driver_sha256'], probe['name'] + ': driver hash')
    check(normalized_driver((folder / 'cpu_driver.py').read_text()) == normalized_driver(template), probe['name'] + ': ROOT-only driver')
    check(request['argv'] == ref['argv'] + [flag], probe['name'] + ': only causal flag')
    check(request['command'][request['command'].index('--') + 1:] == request['argv'], probe['name'] + ': command argv')
    check(all(request[k] == refmeta['profile'][k] for k in ('g2_tool_sha256', 'prepare_sha256', 'prepare_dir', 'steps', 'nproc', 'micro_rows', 'micro_tokens', 'timeout_seconds', 'trace_rmsprop', 'keep', 'shape', 'cpu_only')), probe['name'] + ': reference profile')
    check(request['source_sha256'] == sources['candidate']['sha256'], probe['name'] + ': candidate source')
    check(probe['detached_outer_command'][:3] == ['/usr/bin/flock', '-n', plan['device_lock']], probe['name'] + ': shared lock')
    check(probe['detached_outer_command'][-1] == request['cwd'] + '/cpu_driver.py', probe['name'] + ': driver target')
    check('--stub-prepare' not in request['command'] and '--real-data' not in request['command'], probe['name'] + ': trusted prepare with synthetic data')
    compare = next(c for c in plan['comparisons'] if c['after_probe'] == probe['name'])
    check(compare['required_identical'] and compare['command'][-2:] == [refmeta['remote_path'], request['receipt']], probe['name'] + ': original reference comparison')
check(plan['probes'][0]['label'] != plan['probes'][1]['label'], 'distinct labels')
check(plan['probes'][0]['cwd'] != plan['probes'][1]['cwd'], 'distinct retained roots')
print(json.dumps({'verdict': 'PASS_CONDITIONAL_CPU_PACKET_ONLY', 'checks': len(checks), 'plan_sha256': sha((P / 'cpu-g2-plan.json').read_bytes()), 'reference_sha256': refmeta['sha256'], 'candidate_sha256': sources['candidate']['sha256'], 'names': checks}, indent=2))
