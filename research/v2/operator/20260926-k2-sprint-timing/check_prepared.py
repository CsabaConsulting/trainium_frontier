"""Local mechanical packet checks only: no model imports or host calls."""
import ast, base64, hashlib, json, shlex
from pathlib import Path

here = Path(__file__).resolve().parent
summary = json.loads((here / 'PREPARED.json').read_text())
checks = 0
def require(value):
    global checks
    if not value:
        raise RuntimeError('MECHANICAL_PACKET_CHECK_FAILED')
    checks += 1
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
for row in summary['packets']:
    raw = (here / row['packet']).read_bytes()
    require(sha(raw) == row['packet_sha256'])
    packet = json.loads(raw)
    b = packet['binding']
    files = {}
    for name, entry in packet['files'].items():
        files[name] = base64.b64decode(entry['b64'], validate=True)
        require(sha(files[name]) == entry['sha256'])
    require(json.loads(files['NATIVE_BINDING.json']) == b)
    require(sha(files['package/train.py']) == b['source_sha256'] == row['source_sha256'])
    require(sha(files['host_driver.py']) == b['driver_sha256'] == row['driver_sha256'])
    tree = ast.parse(files['host_driver.py'])
    assignments = {target.id: ast.literal_eval(node.value)
                   for node in tree.body if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)
                   and target.id in {'TRAIN_ENV', 'TRAIN_COMMAND', 'LABEL'}}
    require(assignments['TRAIN_ENV'] == b['train_environment'])
    require(assignments['TRAIN_COMMAND'] == b['train_command'])
    require(assignments['LABEL'] == b['label'])
    require(files['package/launch-command.txt'].decode().strip() == b['launch_command'])
    require(json.loads(files['package/provenance.json'])['launch_command'] == b['launch_command'])
    argv = shlex.split(b['launch_command'])
    require(argv[argv.index('--num-steps') + 1] == '45')
    require(('--k2-embedding' in argv) == (row['mode'] == 'on'))
    require(('--no-k2-embedding' in argv) == (row['mode'] == 'off'))
    require(b['no_eval'] and not b['launch_authorized'] and not b['stage_authorized'])
    phases = [node.args[0].value for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == 'phase' and node.args and isinstance(node.args[0], ast.Constant)]
    require(phases == ['training'])
    original = json.loads((here.parent / '20260926-final-sprint-p0' / (row['host'] + '-bound-packet.json')).read_text())
    for name in ('dispatch_driver.py', 'idle_guard.py', 'lifetime_guard.py'):
        require(packet['files'][name] == original['files'][name])
    require(b['lock_identity'] == original['binding']['lock_identity'])
    require(b['outer_command'][:3] == original['binding']['outer_command'][:3])
    require(b['outer_command'][-1] == b['root'] + '/host_driver.py')
    if row['mode'] == 'on':
        require(len(b['fresh_cache_directories']) == 6)
        require(all(p.startswith(b['root'] + '/cold-training-caches/') for p in b['fresh_cache_directories']))
    else:
        require(not b['fresh_cache_directories'])
        require(b['protected_cache_owner_label'] == original['binding']['label'])
    require(b['timing']['comparison'] if 'comparison' in b['timing'] else True)
    compile(tree, b['label'] + '-host_driver.py', 'exec')
    require(True)
for host in ('trn21', 'trn22'):
    off = json.loads((here / (host + '-off-prepared.json')).read_text())['binding']
    on = json.loads((here / (host + '-on-prepared.json')).read_text())['binding']
    require(off['launch_command'].replace('--no-k2-embedding', '--k2-embedding') == on['launch_command'])
    require(off['source_sha256'] == on['source_sha256'])
print(json.dumps({'verdict': 'MECHANICAL_PREPARATION_PASS', 'checks': checks, 'host_calls': 0, 'model_imports': 0, 'native_result': None}))
