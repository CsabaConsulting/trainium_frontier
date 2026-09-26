"""Disabled exact-source prospective controls; mechanical binding only."""
import ast, base64, hashlib, json
from pathlib import Path

here = Path(__file__).resolve().parent
repo = Path('/home/poojith-devan/Desktop/Trainium')
proposal_path = repo / 'research/v2/proposals/20260926-e1e2-prospective-controls/EXPERIMENT_PROPOSAL.json'
proposal_raw = proposal_path.read_bytes()
proposal = json.loads(proposal_raw)
source = (repo / proposal['candidate']['source']).read_bytes()
command = (repo / proposal['candidate']['launch_command_file']).read_bytes()
def sha(raw): return hashlib.sha256(raw).hexdigest()
def require(value, message):
    if not value: raise RuntimeError(message)
require(sha(source) == proposal['control']['source_sha256'], 'SOURCE_PIN')
require(sha(command) == proposal['candidate']['launch_sha256'], 'COMMAND_PIN')
rows = []
for run in proposal['runs']:
    host, label = run['host'], run['label']
    original_path = here.parent / '20260926-final-sprint-p0' / (host + '-bound-packet.json')
    original = json.loads(original_path.read_text())
    ob = original['binding']
    old = {name: base64.b64decode(entry['b64'], validate=True) for name, entry in original['files'].items()}
    for name, raw in old.items(): require(sha(raw) == original['files'][name]['sha256'], 'PARENT_PIN:' + name)
    root = str(Path(ob['root']).parent / (label + '-packet'))
    provenance = {'label': label, 'kind': 'control', 'base_sha256': sha(source), 'train_sha256': sha(source), 'launch_command': command.decode().strip(), 'single_causal_change': 'none; prospective seed42 exact-source control', 'submission_authorized': False}
    provenance_raw = (json.dumps(provenance, indent=2) + '\n').encode()
    driver = old['host_driver.py'].decode().replace(ob['root'], root).replace(ob['label'], label).replace(ob['source_sha256'], sha(source))
    driver = driver.replace(sha(old['package/launch-command.txt']), sha(command)).replace(sha(old['package/provenance.json']), sha(provenance_raw))
    variant = ('20260926-sprint-p0-clean-recursive/variants/recursive-trn21' if host == 'trn21' else '20260926-sprint-p0-clean-batch/variants/batch-lr1-trn22')
    reviewed = json.loads((repo / 'research/v2/proposals' / variant / 'NATIVE_BINDING.disabled.json').read_text())
    env = {key: value.replace(ob['root'], root).replace(ob['label'], label) for key, value in reviewed['train_environment'].items()}
    tree = ast.parse(driver)
    node = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'TRAIN_ENV' for target in node.targets))
    require(node.lineno == node.end_lineno, 'ENV_LITERAL')
    lines = driver.splitlines(keepends=True)
    lines[node.lineno - 1] = 'TRAIN_ENV = ' + repr(env) + '\n'
    driver = ''.join(lines)
    compile(ast.parse(driver), label + '-driver', 'exec')
    require("phase('fresh-public20m',EVAL_COMMAND,EVAL_ENV,guardian)" in driver, 'ATTACHED_EVAL_RETAINED')
    binding = {key: value for key, value in ob.items() if key not in ('original_reviewed_driver_sha256', 'cold_bound_driver_sha256', 'fresh_cache_directories')}
    binding.update(label=label, root=root, source_sha256=sha(source), base_sha256=sha(source), kind='control', launch_command=command.decode().strip(), stage_authorized=False, launch_authorized=False, release_file_created=False, source_adoption=False, submission_authorized=False, fresh_private_training_cache=False, train_environment=env, fresh_cache_directories=[], cache_mode='EXISTING_PROVEN_SHARED_RESEARCH_NAMESPACE_HIT_UNPROVEN', driver_sha256=sha(driver.encode()), proposal_sha256=sha(proposal_raw), scheduling='THIRD_PENDING_FALLBACK_AFTER_P0_AND_K2', native_resource_preflight_pending=True)
    for key in ('train_command', 'eval_command', 'outer_command'):
        binding[key] = [value.replace(ob['root'], root).replace(ob['label'], label) for value in ob[key]]
    binding['eval_environment'] = {key: value.replace(ob['root'], root).replace(ob['label'], label) for key, value in ob['eval_environment'].items()}
    files = {name: old[name] for name in ('dispatch_driver.py', 'idle_guard.py', 'lifetime_guard.py')}
    files.update({'host_driver.py': driver.encode(), 'package/train.py': source, 'package/launch-command.txt': command, 'package/provenance.json': provenance_raw, 'EXPERIMENT_PROPOSAL.json': proposal_raw, 'NATIVE_BINDING.json': (json.dumps(binding, indent=2) + '\n').encode()})
    packet = {'binding': binding, 'files': {name: {'sha256': sha(raw), 'b64': base64.b64encode(raw).decode()} for name, raw in files.items()}, 'local_only': True, 'host_calls': 0, 'native_started': False, 'derivation': {'parent_driver_sha256': sha(old['host_driver.py']), 'logic_changed': False, 'science_changed': False, 'constants_only': True}}
    out = here / (host + '-prepared.json')
    with out.open('x') as f: json.dump(packet, f, indent=2); f.write('\n')
    rows.append({'host': host, 'label': label, 'packet': out.name, 'packet_sha256': sha(out.read_bytes()), 'driver_sha256': sha(driver.encode()), 'source_sha256': sha(source), 'launch_sha256': sha(command), 'status': 'SOURCE_QUALIFIED_LOCAL_PREPARED_FRESH_NATIVE_RESOURCE_CHECKS_PENDING'})
with (here / 'PREPARED.json').open('x') as f:
    json.dump({'proposal_sha256': sha(proposal_raw), 'packets': rows, 'no_host_calls': True, 'native_started': False, 'do_not_relabel_historical_result': True}, f, indent=2); f.write('\n')
print(json.dumps({'prepared': rows, 'no_host_calls': True}))
