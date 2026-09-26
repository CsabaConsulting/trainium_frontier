"""Disabled full-run alternatives, mechanically derived; no host/model calls."""
import ast, base64, hashlib, json, shlex
from pathlib import Path

here = Path(__file__).resolve().parent
timing = here.parent / '20260926-k2-sprint-timing'
p0 = here.parent / '20260926-final-sprint-p0'
source_sha = '63925519ac6c0aab1a1d598e96fef3f4413fbd083d9b6d177901a2e972060db5'
review_sha = '2ddb87ac3c38897abf306e323d06c747b0db41423e2f6126ede94e98cadcf502'
cache_keys = ('NEURON_COMPILE_CACHE_URL', 'NEURON_COMPILE_CACHE_DIR', 'TORCH_NEURONX_NEFF_CACHE_DIR', 'TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR', 'TORCH_NEURONX_HLO_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'TORCH_NEURONX_DEBUG_DIR')
def sha(raw): return hashlib.sha256(raw).hexdigest()
def require(condition, error):
    if not condition: raise RuntimeError(error)
def unpack(packet):
    files = {name: base64.b64decode(entry['b64'], validate=True) for name, entry in packet['files'].items()}
    for name, raw in files.items(): require(sha(raw) == packet['files'][name]['sha256'], 'INPUT_PIN:' + name)
    return files
rows = []
for host in ('trn21', 'trn22'):
    parent_raw = (p0 / (host + '-bound-packet.json')).read_bytes()
    parent = json.loads(parent_raw)
    original = unpack(parent)
    ob = parent['binding']
    short_raw = (timing / (host + '-on-prepared.json')).read_bytes()
    short = json.loads(short_raw)
    short_files = unpack(short)
    sb = short['binding']
    require(sha(short_files['package/train.py']) == sb['source_sha256'] == source_sha, 'K2_SOURCE')
    require(sb['independent_review_sha256'] == review_sha, 'REVIEW')
    command = sb['launch_command']
    require(command.count(' --num-steps 45') == 1, 'EXACT_SHORT_STOP_FLAG')
    full_command = command.replace(' --num-steps 45', '')
    short_argv, full_argv = shlex.split(command), shlex.split(full_command)
    index = short_argv.index('--num-steps')
    require(short_argv[:index] + short_argv[index + 2:] == full_argv, 'ONLY_STOP_LIMIT_REMOVED')
    require('--k2-embedding' in full_argv and '--num-steps' not in full_argv, 'K2_FULL_ARGV')
    label = 'fs-20260926-k2-sprint-full-' + host + '-r1'
    root = str(Path(ob['root']).parent / (label + '-packet'))
    command_raw = (full_command + '\n').encode()
    provenance = {'label': label, 'train_sha256': source_sha, 'base_sha256': 'e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43', 'launch_command': full_command, 'scientific_patch': 'K2_embedding_backward', 'run_purpose': 'conditional_fixed1800_quality_arm', 'cache_owner_label': sb['label'], 'cache_mode': 'WARM_RESEARCH_ONLY_HIT_UNPROVEN', 'submission_authorized': False}
    provenance_raw = (json.dumps(provenance, indent=2) + '\n').encode()
    driver = original['host_driver.py'].decode().replace(ob['root'], root).replace(ob['label'], label).replace(ob['source_sha256'], source_sha)
    driver = driver.replace(sha(original['package/launch-command.txt']), sha(command_raw)).replace(sha(original['package/provenance.json']), sha(provenance_raw))
    env = {key: value.replace(ob['root'], root).replace(ob['label'], label) for key, value in ob['train_environment'].items()}
    for key in cache_keys: env[key] = sb['train_environment'][key]
    tree = ast.parse(driver)
    node = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'TRAIN_ENV' for target in node.targets))
    require(node.lineno == node.end_lineno, 'ENV_LITERAL')
    lines = driver.splitlines(keepends=True)
    lines[node.lineno - 1] = 'TRAIN_ENV = ' + repr(env) + '\n'
    driver = ''.join(lines)
    compile(ast.parse(driver), label + '-driver', 'exec')
    binding = {key: value for key, value in ob.items() if key not in ('original_reviewed_driver_sha256', 'cold_bound_driver_sha256', 'native_short_timing')}
    binding.update(schema='disabled-k2-sprint-full-followup-v1', root=root, label=label, source_sha256=source_sha, scientific_patch='K2_embedding_backward', train_environment=env, launch_command=full_command, stage_authorized=False, launch_authorized=False, release_file_created=False, host_reserved=False, source_adoption=False, submission_authorized=False, independent_review_sha256=review_sha, operator_assignment_sha256=None, full_proposal_present=False, full_proposal_sha256=None, fresh_private_training_cache=False, fresh_cache_directories=[], driver_sha256=sha(driver.encode()), cache_owner_label=sb['label'], cache_mode='RETAINED_ON45_PRIVATE_NAMESPACE_WARM_RESEARCH_ONLY_HIT_UNPROVEN', cache_paths_must_already_exist=sorted(set(env[key] for key in cache_keys)), seed=42, max_train_seconds=1800, no_automatic_next_job=True)
    for key in ('train_command', 'eval_command', 'outer_command'):
        binding[key] = [value.replace(ob['root'], root).replace(ob['label'], label) for value in ob[key]]
    binding['eval_environment'] = {key: value.replace(ob['root'], root).replace(ob['label'], label) for key, value in ob['eval_environment'].items()}
    binding['conditional_requirements'] = ['Supervisor supplies completed full-run proposal before host dispatch.', 'Same-host exact-source OFF and ON45 both valid and finite with actual complete boundary, budget, exit and explicit quiescence evidence.', 'Precommitted whole-window ON/OFF tokens-per-second ratio >=1.03, not printed-step or kernel-only timing.', 'Only the host with the qualifying pair may use this alternative unless separately assigned.', 'Preserve both short-run checkpoints and ON45 caches; fresh disk must include retained outputs plus unchanged full-run reserve.', 'Fresh original inode lifetime-lock, ps, neuron-ls, fuser, STOP, source/argv and resource checks.', 'Current P0 and attached evaluation finish naturally and are retrieved before any successor.', 'This warm full research run is not a cold full submission replay; short cold startup evidence does not replace one.']
    files = {name: original[name] for name in ('dispatch_driver.py', 'idle_guard.py', 'lifetime_guard.py')}
    files.update({'host_driver.py': driver.encode(), 'package/train.py': short_files['package/train.py'], 'package/launch-command.txt': command_raw, 'package/provenance.json': provenance_raw, 'NATIVE_BINDING.disabled.json': (json.dumps(binding, indent=2) + '\n').encode()})
    packet = {'binding': binding, 'files': {name: {'sha256': sha(raw), 'b64': base64.b64encode(raw).decode()} for name, raw in files.items()}, 'local_only': True, 'host_calls': 0, 'native_started': False, 'missing_before_dispatch': ['completed_full_EXPERIMENT_PROPOSAL.json', 'actual_valid_same_host_OFF_ON45>=1.03_receipts', 'fresh_machine_resources_and_quiescence', 'matching_full_assignment'], 'mechanical_derivation': {'p0_parent_packet_sha256': sha(parent_raw), 'p0_parent_driver_sha256': sha(original['host_driver.py']), 'short_on_packet_sha256': sha(short_raw), 'source_modified': False, 'driver_logic_changed': False, 'changes': ['literal root/label/source/package pins', 'TRAIN_ENV reuses qualifying ON45 cache paths', 'science argv removes exactly --num-steps 45; ON flag remains'], 'guard_dispatch_unchanged': True}}
    out = here / (host + '-prepared.json')
    with out.open('x') as f: json.dump(packet, f, indent=2); f.write('\n')
    rows.append({'host': host, 'label': label, 'packet': out.name, 'packet_sha256': sha(out.read_bytes()), 'driver_sha256': sha(driver.encode()), 'source_sha256': source_sha, 'launch_command_sha256': sha(command_raw), 'cache_owner_label': sb['label'], 'status': 'DISABLED_LOCAL_ONLY_TIMING_AND_FULL_PROPOSAL_PENDING'})
with (here / 'PREPARED.json').open('x') as f:
    json.dump({'packets': rows, 'one_chosen_host_alternative_after_qualifying_pair': True, 'no_host_calls': True, 'native_started': False, 'source_review_reused': review_sha, 'full_proposal_pending': True}, f, indent=2); f.write('\n')
print(json.dumps({'prepared': rows, 'no_host_calls': True}))
