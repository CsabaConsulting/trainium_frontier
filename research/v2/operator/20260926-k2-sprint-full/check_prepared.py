"""Mechanical checks; no source imports, tensors, remote calls or new gates."""
import ast, base64, hashlib, json, shlex
from pathlib import Path
here = Path(__file__).resolve().parent
checks = 0
def require(value):
    global checks
    if not value: raise RuntimeError('FULL_FOLLOWUP_MECHANICAL_CHECK_FAILED')
    checks += 1
def sha(raw): return hashlib.sha256(raw).hexdigest()
summary = json.loads((here / 'PREPARED.json').read_text())
for row in summary['packets']:
    raw = (here / row['packet']).read_bytes()
    require(sha(raw) == row['packet_sha256'])
    packet = json.loads(raw); b = packet['binding']
    files = {name: base64.b64decode(value['b64'], validate=True) for name, value in packet['files'].items()}
    for name, value in files.items(): require(sha(value) == packet['files'][name]['sha256'])
    require(json.loads(files['NATIVE_BINDING.disabled.json']) == b)
    require('RELEASE.json' not in files and 'EXPERIMENT_PROPOSAL.json' not in files)
    require(not b['launch_authorized'] and not b['stage_authorized'] and not b['full_proposal_present'])
    require(not b['fresh_private_training_cache'] and not b['fresh_cache_directories'])
    require(b['source_sha256'] == sha(files['package/train.py']) == row['source_sha256'])
    require(b['driver_sha256'] == sha(files['host_driver.py']) == row['driver_sha256'])
    require(files['package/launch-command.txt'].decode().strip() == b['launch_command'])
    require(json.loads(files['package/provenance.json'])['launch_command'] == b['launch_command'])
    short = json.loads((here.parent / '20260926-k2-sprint-timing' / (row['host'] + '-on-prepared.json')).read_text())
    parent = json.loads((here.parent / '20260926-final-sprint-p0' / (row['host'] + '-bound-packet.json')).read_text())
    require(packet['files']['package/train.py'] == short['files']['package/train.py'])
    for name in ('idle_guard.py', 'lifetime_guard.py', 'dispatch_driver.py'):
        require(packet['files'][name] == parent['files'][name])
    sa = shlex.split(short['binding']['launch_command']); fa = shlex.split(b['launch_command']); index = sa.index('--num-steps')
    require(sa[index + 1] == '45' and sa[:index] + sa[index + 2:] == fa)
    require('--k2-embedding' in fa and '--no-k2-embedding' not in fa)
    tree = ast.parse(files['host_driver.py'])
    previous_text = base64.b64decode(parent['files']['host_driver.py']['b64']).decode()
    # preflight embeds these two authenticated package pins inside its function;
    # compare logic after exactly those mandatory literal hash rebindings.
    for name in ('package/launch-command.txt', 'package/provenance.json'):
        old_pin = parent['files'][name]['sha256']
        new_pin = packet['files'][name]['sha256']
        require(previous_text.count(old_pin) == 1)
        previous_text = previous_text.replace(old_pin, new_pin)
    previous = ast.parse(previous_text)
    functions = lambda value: [ast.dump(n, include_attributes=False) for n in value.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    require(functions(tree) == functions(previous))
    assignments = {target.id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) for target in node.targets if isinstance(target, ast.Name) and target.id in ('TRAIN_ENV', 'EVAL_ENV', 'TRAIN_COMMAND', 'EVAL_COMMAND', 'LABEL', 'SOURCE_SHA')}
    for name, key in (('TRAIN_ENV', 'train_environment'), ('EVAL_ENV', 'eval_environment'), ('TRAIN_COMMAND', 'train_command'), ('EVAL_COMMAND', 'eval_command'), ('LABEL', 'label'), ('SOURCE_SHA', 'source_sha256')):
        require(assignments[name] == b[key])
    for key in ('NEURON_COMPILE_CACHE_URL', 'NEURON_COMPILE_CACHE_DIR', 'TORCH_NEURONX_NEFF_CACHE_DIR', 'TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR', 'TORCH_NEURONX_HLO_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'TORCH_NEURONX_DEBUG_DIR'):
        require(b['train_environment'][key] == short['binding']['train_environment'][key])
    require(b['train_environment']['NEURON_COMPETITION_R1_SEED'] == '42')
    require(b['train_environment']['NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS'] == '1800')
    require(b['lock_identity'] == parent['binding']['lock_identity'])
    phases = [n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'phase' and n.args and isinstance(n.args[0], ast.Constant)]
    require(phases == ['training', 'fresh-public20m'])
    require(b['eval_command'][b['eval_command'].index('--eval-tokens') + 1] == '20971520')
    require(b['eval_command'][b['eval_command'].index('--checkpoint') + 1] == b['root'] + '/train/out/final.pt')
    compile(tree, row['label'] + '-driver', 'exec')
print(json.dumps({'status': 'MECHANICAL_PREPARATION_PASS', 'checks': checks, 'host_calls': 0, 'native_started': False, 'full_proposal_and_actual_timing_pending': True}))
