"""Disabled cold native201 rebind of known timing mechanics; no SSH/launch."""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def sha(raw): return hashlib.sha256(raw).hexdigest()
def enc(obj): return (json.dumps(obj, indent=2) + '\n').encode()


source = (HERE / 'train.py').read_bytes()
if sha(source) != '949f262c092c141f715bdd71532c42e0bcf502d926a1ee20bc885a2255062daa':
    raise ValueError('SOURCE_CHANGED')
cpu = (HERE / 'CPU_RECEIPT.json').read_bytes()
reference = (HERE / 'NATIVE_TIMING_REFERENCE.json').read_bytes()
parent_path = REPO / 'research/v2/operator/20260926-operator-recovery/trn21-warmdown060-prepared.disabled.json'
raw = parent_path.read_bytes()
if sha(raw) != 'f402300af29399caf6aff0be6f9eb186a659a76679ff21ee05b5baa438582ab3':
    raise ValueError('PARENT_CHANGED')
parent = json.loads(raw)
old = parent['binding']
files = {k: base64.b64decode(v['b64'], validate=True) for k, v in parent['files'].items()}
for k, v in files.items():
    if sha(v) != parent['files'][k]['sha256']: raise ValueError('MEMBER_CHANGED:' + k)
label = 'fs-20260926-k2-outside-compile-native201-trn21-r1'
root = str(Path(old['root']).parent / (label + '-packet'))
command = old['launch_command'].split(' --warmdown-ratio')[0] + ' --num-steps 201 --k2-embedding'
command_raw = (command + '\n').encode()
proposal = {
    'proposal_id': label, 'arm_id': 'k2-outside-compile-native201', 'kind': 'kernel_timing',
    'status': 'LOCAL_CPU_PASS_INDEPENDENT_REVIEW_PENDING_NOT_RELEASED',
    'authority': 'Poojith latest: K2 native timing on one host before momentum150, cold net>=3%, loss through200',
    'single_causal_change': 'Exact K2 backward moved outside captured transformer; unchanged batch262144/LR.021/ordinary.75/momentum300/seed42',
    'candidate': {'source_sha256': sha(source), 'launch_command': command, 'launch_command_sha256': sha(command_raw), 'steps': 201},
    'local_evidence': {'cpu_receipt_sha256': sha(cpu), 'reference_sha256': sha(reference), 'independent_review': 'PENDING'},
    'control': json.loads(reference)['rows'][0],
    'qualification': 'max(0,1800-max(0,cold_startup-900))*boundary9..200 aggregate_tok_s >= 521514012.2158874; finite loss through200 versus frozen samples, correct checkpoint/budget',
    'loss_comparison_scope': 'Wall-time LR may differ; report every matched sample and first-step difference, do not assert bit identity from CPU test.',
    'accounting': {'budget': 1800, 'startup_cap': 900, 'cold': True},
    'native_scope': 'One201step diagnostic, not full BPB, no automatic table/full launch and no evaluation',
    'dispatch': {'host': 'trn21', 'label': label, 'launch_authorized': False, 'stage_authorized': False},
    'submission_authorized': False,
}
proposal_raw = enc(proposal)
prov = {'label': label, 'train_sha256': sha(source), 'source_sha256': sha(source),
        'launch_command': command, 'base_sha256': old['base_sha256'], 'run_purpose': 'cold-native201-timing-only',
        'submission_authorized': False}
prov_raw = enc(prov)
changes = [(old['root'], root), (old['label'], label), (old['source_sha256'], sha(source)),
           (sha(files['package/launch-command.txt']), sha(command_raw)),
           (sha(files['package/provenance.json']), sha(prov_raw))]


def rebind(value):
    if isinstance(value, str):
        for a, b in changes: value = value.replace(a, b)
        return value
    if isinstance(value, dict): return {k: rebind(v) for k, v in value.items()}
    if isinstance(value, list): return [rebind(v) for v in value]
    return value


driver = rebind(files['host_driver.py'].decode())
reverse = driver
for a, b in reversed(changes): reverse = reverse.replace(b, a)
if reverse != files['host_driver.py'].decode(): raise ValueError('DRIVER_LITERAL_DRIFT')
begin = driver.index('    eval_fd=guard.require_idle_under_inherited_lock(')
end = driver.index('except BaseException as exc:', begin)
removed = driver[begin:end]
if "rc=phase('fresh-public20m',EVAL_COMMAND,EVAL_ENV,guardian)" not in removed:
    raise ValueError('EVAL_BLOCK_CHANGED')
driver = driver[:begin] + "    record('terminal.json',{'utc':utc(),'label':LABEL,'status':'TIMING_TRAINING_COMPLETE_PENDING_ANALYSIS','train_exit_code':rc,'no_eval':True,'no_automatic_full':True,'submission_authorized':False})\n" + driver[end:]
ast.parse(driver)
binding = rebind(copy.deepcopy(old))
binding.update(schema='disabled-k2-outside-compile-native201-v1', source_commit=None,
               source_sha256=sha(source), launch_command=command, driver_sha256=sha(driver.encode()),
               steps=201, no_eval=True, kind='kernel_timing', arm_id=proposal['arm_id'],
               run_purpose='cold-native201-timing-only', independent_review_sha256=None,
               completed_proposal_path=str((HERE / 'NATIVE_PROPOSAL.json').relative_to(REPO)),
               completed_proposal_sha256=sha(proposal_raw), operator_assignment_sha256=None,
               stage_authorized=False, launch_authorized=False, release_file_created=False,
               timing_reference_sha256=sha(reference), scientific_patch=proposal['single_causal_change'])
for k in ('NATIVE_BINDING.disabled.json', 'NATIVE_BINDING.json', 'RELEASE.json', 'EXACT_SOURCE_REVIEW.md', 'SUPERVISOR_ASSIGNMENT.md'):
    files.pop(k, None)
files.update({'host_driver.py': driver.encode(), 'package/train.py': source,
              'package/launch-command.txt': command_raw, 'package/provenance.json': prov_raw,
              'EXPERIMENT_PROPOSAL.json': proposal_raw, 'CPU_RECEIPT.json': cpu,
              'NATIVE_TIMING_REFERENCE.json': reference, 'NATIVE_BINDING.disabled.json': enc(binding)})
packet = enc({'binding': binding, 'files': {k: {'sha256': sha(v), 'b64': base64.b64encode(v).decode()} for k, v in files.items()},
              'derivation': {'parent_sha256': sha(raw), 'literal_reversal_exact': True,
                             'driver_logic_change': 'Replace attached eval block by timing terminal only, as prior native201 machinery',
                             'guardians_unchanged': True, 'host_calls': 0}})
for name, data in [('NATIVE_PROPOSAL.json', proposal_raw), ('native201.disabled.json', packet)]:
    path = HERE / name
    if path.exists():
        if path.read_bytes() != data: raise ValueError('PREPARATION_CHANGED:' + name)
    else:
        with path.open('xb') as stream: stream.write(data)
print(json.dumps({'label': label, 'packet_sha256': sha(packet), 'source_sha256': sha(source),
                  'cold': True, 'steps': 201, 'independent_review': 'PENDING', 'launch_authorized': False, 'host_calls': 0}))
