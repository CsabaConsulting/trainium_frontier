"""Prospective seed43 candidate packet, warm research cache, no host calls."""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FINAL = REPO / 'workflow/candidates/final-batch262144-lr021'
PARENT = REPO / 'research/v2/operator/20260926-operator-recovery/manual-jobs/trn22-warmdown085/packet.json'


def sha(raw): return hashlib.sha256(raw).hexdigest()
def enc(obj): return (json.dumps(obj, indent=2) + '\n').encode()


raw = PARENT.read_bytes()
manifest = json.loads((PARENT.parent.parent / 'manifest.json').read_bytes())
if sha(raw) != manifest['trn22-warmdown085']['packet_sha256']:
    raise ValueError('PARENT_PACKET_DRIFT')
parent = json.loads(raw)
old = parent['binding']
original = {k: base64.b64decode(v['b64'], validate=True) for k, v in parent['files'].items()}
for k, v in original.items():
    if sha(v) != parent['files'][k]['sha256']: raise ValueError('PARENT_MEMBER_DRIFT:' + k)
source = (FINAL / 'train.py').read_bytes()
if sha(source) != 'cd0c4aa59754f46705355d964fad4528cf1de77324469b0829bcc0de149b757d':
    raise ValueError('FINAL_SOURCE_DRIFT')
accepted = (FINAL / 'launch-command.txt').read_bytes()
if sha(accepted) != 'b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd':
    raise ValueError('ACCEPTED_COMMAND_DRIFT')
command = accepted.decode().strip() + ' --seed 43'
command_raw = (command + '\n').encode()
label = 'fs-20260926-final-batch262144-lr021-seed43-trn22-r1'
root = str(Path(old['root']).parent / (label + '-packet'))
proposal = {
    'proposal_id': label, 'arm_id': 'final-batch-lr021-seed43', 'kind': 'seed_confirmation',
    'status': 'PREPARED_DISABLED', 'authority': 'Poojith: overnight cold replay, second-seed confirmation, then remaining arms',
    'single_causal_change': 'Seed42 ->43 only versus clean confirmed batch/LR; same selected shape, budget and optimizer',
    'candidate': {'source_sha256': sha(source), 'launch_command': command, 'seed': 43,
                  'max_train_seconds': 1800, 'total_batch_size': 262144, 'matrix_lr': 0.021},
    'evaluation': {'fresh_process': True, 'tokens': 20971520, 'context': 1024, 'batch_size': 1},
    'comparison': 'Candidate43 measures variation; no matched seed43 causal gain unless actual accepted8L seed43 control exists. Do not pick best seed.',
    'cache_policy': 'Existing trn22 research caches, hit unproven; not cold submission evidence',
    'pending': ['Selected final source confirmed', 'Fresh original lock/machine/source admission', 'Full1800 and actual fresh20M'],
    'dispatch': {'host': 'trn22', 'label': label, 'launch_authorized': False, 'stage_authorized': False},
    'submission_authorized': False,
}
proposal_raw = enc(proposal)
provenance = {'label': label, 'train_sha256': sha(source), 'source_sha256': sha(source),
              'launch_command': command, 'base_sha256': old['base_sha256'], 'seed': 43,
              'run_purpose': 'second-seed-variation', 'submission_authorized': False}
provenance_raw = enc(provenance)
changes = [(old['root'], root), (old['label'], label), (old['source_sha256'], sha(source)),
           (sha(original['package/launch-command.txt']), sha(command_raw)),
           (sha(original['package/provenance.json']), sha(provenance_raw))]


def transform(value):
    if isinstance(value, str):
        for a, b in changes: value = value.replace(a, b)
        return value
    if isinstance(value, dict): return {k: transform(v) for k, v in value.items()}
    if isinstance(value, list): return [transform(v) for v in value]
    return value


driver = transform(original['host_driver.py'].decode())
reverse = driver
for a, b in reversed(changes): reverse = reverse.replace(b, a)
if reverse != original['host_driver.py'].decode(): raise ValueError('DRIVER_LOGIC_DRIFT')
lines = driver.splitlines(keepends=True)
envs = {}
for n in ast.parse(driver).body:
    if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('TRAIN_ENV', 'EVAL_ENV'):
        name = n.targets[0].id
        if n.lineno != n.end_lineno: raise ValueError('ENV_NOT_LITERAL_LINE')
        env = ast.literal_eval(n.value)
        if env['NEURON_COMPETITION_R1_SEED'] != '42': raise ValueError('PARENT_SEED_NOT42')
        env['NEURON_COMPETITION_R1_SEED'] = '43'
        envs[name] = env
        lines[n.lineno - 1] = name + ' = ' + repr(env) + '\n'
if set(envs) != {'TRAIN_ENV', 'EVAL_ENV'}: raise ValueError('ENV_BINDING_MISSING')
driver = ''.join(lines)
ast.parse(driver)
binding = transform(copy.deepcopy(old))
binding.update(schema='disabled-final-second-seed-v1', source_commit=None,
               source_sha256=sha(source), driver_sha256=sha(driver.encode()), launch_command=command,
               kind='seed_confirmation', arm_id=proposal['arm_id'], scientific_patch=proposal['single_causal_change'],
               train_environment=envs['TRAIN_ENV'], eval_environment=envs['EVAL_ENV'],
               completed_proposal_path=str((HERE / 'EXPERIMENT_PROPOSAL.json').relative_to(REPO)),
               completed_proposal_sha256=sha(proposal_raw), independent_review_sha256=None,
               operator_assignment_sha256=None, stage_authorized=False, launch_authorized=False,
               release_file_created=False, selected_cache_policy=proposal['cache_policy'])
if binding['fresh_cache_directories']: raise ValueError('RESEARCH_CACHE_EXPECTED')
members = {k: v for k, v in original.items() if k not in (
    'NATIVE_BINDING.json', 'NATIVE_BINDING.disabled.json', 'RELEASE.json',
    'EXACT_SOURCE_REVIEW.md', 'SUPERVISOR_ASSIGNMENT.md')}
members.update({'host_driver.py': driver.encode(), 'package/train.py': source,
                'package/launch-command.txt': command_raw, 'package/provenance.json': provenance_raw,
                'EXPERIMENT_PROPOSAL.json': proposal_raw, 'CPU_RECEIPT.json': (FINAL / 'CPU_RECEIPT.json').read_bytes(),
                'NATIVE_BINDING.disabled.json': enc(binding)})
packet = enc({'binding': binding, 'files': {k: {'sha256': sha(v), 'b64': base64.b64encode(v).decode()} for k, v in members.items()},
              'derivation': {'parent_sha256': sha(raw), 'driver_literal_reversal_exact_before_seed_env': True,
                             'only_additional_driver_change': 'TRAIN_ENV/EVAL_ENV seed42->43',
                             'guardian_bytes_unchanged': True, 'host_calls': 0}})
for name, data in [('EXPERIMENT_PROPOSAL.json', proposal_raw), ('packet.disabled.json', packet)]:
    path = HERE / name
    if path.exists():
        if path.read_bytes() != data: raise ValueError('EXISTING_PREPARATION_CHANGED:' + name)
    else:
        with path.open('xb') as stream: stream.write(data)
print(json.dumps({'label': label, 'source_sha256': sha(source), 'seed': 43,
                  'packet_sha256': sha(packet), 'launch_authorized': False, 'host_calls': 0}))
