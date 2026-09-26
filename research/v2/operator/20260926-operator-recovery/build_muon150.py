"""Mechanical packet derivation for Poojith's one-literal warmup arm. No SSH."""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
JOBS = HERE / 'manual-jobs'
SPEC = REPO / 'research/v2/proposals/20260926-batch-lr021-muon150'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2) + '\n').encode()


def write_once(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            assert not any((path.parent / name).exists() for name in ('attempt.json','dispatch.json','results-bundle.json')), 'CONSUMED_ARTIFACT_IMMUTABLE'
            path.write_bytes(raw)  # Regenerate only unconsumed local packaging metadata.
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)


source = (SPEC / 'train.py').read_bytes()
cpu = (SPEC / 'CPU_RECEIPT.json').read_bytes()
scope = (SPEC / 'SCALAR_SCOPE.md').read_bytes()
assignment = (SPEC / 'SPEC.md').read_bytes()
assert sha(source) == '4b0641db9f009af1601ff48e6944a0c6d851164f42011f026d06ed20761c3020'
assert json.loads(cpu)['source_sha256'] == sha(source)
assert json.loads(cpu)['status'] == 'PASS'
manifest_path = JOBS / 'manifest.json'
manifest = json.loads(manifest_path.read_bytes())
for host in ('trn21', 'trn22'):
    parent_key = host + ('-warmdown085' if host == 'trn21' else '-warmdown060')
    original_raw = (JOBS / parent_key / 'packet.json').read_bytes()
    assert sha(original_raw) == manifest[parent_key]['packet_sha256']
    parent = json.loads(original_raw)
    old = parent['binding']
    originals = {k: base64.b64decode(v['b64'], validate=True) for k, v in parent['files'].items()}
    assert all(sha(v) == parent['files'][k]['sha256'] for k, v in originals.items())
    key = host + '-muon150'
    label = 'fs-20260926-batch-lr021-muon150-' + host + '-r1'
    root = str(Path(old['root']).parent / (label + '-packet'))
    command = old['launch_command'].split(' --warmdown-ratio')[0]
    command_raw = (command + '\n').encode()
    proposal = json.loads(originals['EXPERIMENT_PROPOSAL.json'])
    proposal.update(proposal_id='20260926-' + key, arm_id='batch262144-lr021-muon150',
                    owner_role='human_selected_scalar_packaged_by_main', prepared_utc='2026-09-26T11:58:22Z',
                    authority='Poojith direct MUON_MOMENTUM_WARMUP_STEPS=150 instruction,26 September',
                    status='COMPLETE_HUMAN_SCALAR_CPU_PASS_PENDING_MACHINE', kind='schedule_arm',
                    mechanism_class='momentum_warmup_token_length',
                    single_causal_change='MUON_MOMENTUM_WARMUP_STEPS300 ->150; ordinary warmdown.75 and batch262144/LR.021 unchanged',
                    hypothesis='Restore pre-batch-doubling token-length of momentum warmup; final-BPB effect unknown.',
                    predicted_bpb_delta=None, p_improve=None,
                    implementation_spec=str((SPEC / 'SPEC.md').relative_to(REPO)),
                    implementation_spec_sha256=sha(assignment),
                    local_scope='Human-selected scalar, byte reversal proves unchanged parent; no fresh independent-session review claimed.')
    proposal['parent']['site_evidence'] = 'Only line62 constant changes; momentum function1320-1328 and all other functions byte-identical'
    proposal['host_alternatives'] = [{'host':h,'label':'fs-20260926-batch-lr021-muon150-'+h+'-r1'} for h in ('trn21','trn22')]
    proposal['gates'] = {'source_frozen':True,'single_literal_byte_ast_identity':True,'cpu_reference_pass':True,
                         'human_selected_scalar':True,'fresh_independent_session_review':False,
                         'actual_selected_host_comparator_bound':True,'fresh_machine_admission':False,
                         'native_passed':False,'full_result':False,'formal_selector_control_bound':False,'submission_authorized':False}
    proposal['evidence_contract'] = {'source':'SPEC.md and SCALAR_SCOPE.md; all parent bytes recovered by one-literal reversal',
                                     'cpu':'28,308 actual scalar/reference cases, normal and -O; not native/model training',
                                     'native':'Initial execution of full run, no extra smoke or native PASS claimed'}
    proposal['prior_evidence'] = {'current_base':'Confirmed two-host public mean.986879270651192',
                                 'schedule_stack':'Actual .9887829006, raw delta+.0017974; do not stack',
                                 'warmup':'300*262144=78.6Mtokens;150 restores39.3M; quality unknown'}
    proposal['human_approved_by'] = 'Poojith direct instruction: next arm after warmdowns is MUON_MOMENTUM_WARMUP_STEPS150'
    proposal['bound_evidence'] = {'source_sha256':sha(source),'source_commit':None,
                                 'cpu_receipt':str((SPEC/'CPU_RECEIPT.json').relative_to(REPO)), 'cpu_receipt_sha256':sha(cpu),
                                 'scope_receipt':str((SPEC/'SCALAR_SCOPE.md').relative_to(REPO)), 'scope_sha256':sha(scope),
                                 'fresh_independent_review':False}
    proposal['artifacts'] = 'Exact source/CPU/scalar scope, proposal/command, same-host comparator, actual host/lock/runtime/startup/accounting/fresh20M/checkpoint; no new independent review claimed'
    proposal['candidate'].update(source=str((SPEC / 'train.py').relative_to(REPO)), source_sha256=sha(source),
                                 source_commit=None, source_status='CPU_SCALAR_AND_BYTE_IDENTITY_PASS',
                                 launch_command=command, launch_command_sha256=sha(command_raw),
                                 launch_string_sha256=sha(command.encode()), argv=command.split('train.py ',1)[1].split(),
                                 warmdown_ratio=0.75, matrix_lr=0.021, muon_momentum_warmup_steps=150)
    # Do not inherit a review of different bytes or stale warmdown implementation claims.
    proposal['review'] = {'scope':'Human-selected scalar; unchanged functions covered by parent evidence',
                          'fresh_independent_session':False, 'receipt':str((SPEC / 'SCALAR_SCOPE.md').relative_to(REPO))}
    for name in ('verification', 'implementation', 'source_review', 'cpu_gate', 'g2'):
        if name in proposal:
            proposal[name] = {'scope':'See exact CPU_RECEIPT.json and SCALAR_SCOPE.md for this source', 'inherited_claim_removed':True}
    proposal['dispatch'].update(selected_host=host, selected_label=label, ready=True, stage_authorized=True,
                                launch_authorized=True, operator='/root current sole Operator',
                                order='Next after current warmdown and attached20M; no preemption')
    proposal['control']['comparison'] = 'Actual same-host confirmed batch262144/LR.021 ordinary warmdown.75 and momentum300; old ledger roles unchanged'
    proposal_raw = encode(proposal)
    provenance = json.loads(originals['package/provenance.json'])
    provenance.update(label=label, launch_command=command, kind='schedule_arm', source_sha256=sha(source),
                      source_commit=None, human_selected_scalar=True, submission_authorized=False)
    provenance_raw = encode(provenance)
    replacements = [(old['root'],root),(old['label'],label),(old['source_sha256'],sha(source)),
                    (sha(originals['package/launch-command.txt']),sha(command_raw)),
                    (sha(originals['package/provenance.json']),sha(provenance_raw))]
    replacements = [(a,b) for a,b in replacements if a != b]

    def transform(value):
        if isinstance(value,str):
            for a,b in replacements:
                value=value.replace(a,b)
            return value
        if isinstance(value,dict):
            return {k:transform(v) for k,v in value.items()}
        if isinstance(value,list):
            return [transform(v) for v in value]
        return value

    driver = transform(originals['host_driver.py'].decode())
    reversed_driver = driver
    for a,b in reversed(replacements):
        reversed_driver = reversed_driver.replace(b,a)
    assert reversed_driver == originals['host_driver.py'].decode(), 'DRIVER_LOGIC_DRIFT'
    ast.parse(driver)
    binding = transform(copy.deepcopy(old))
    proposal_rel = str((JOBS / key / 'EXPERIMENT_PROPOSAL.json').relative_to(REPO))
    binding.update(source_commit=None, source_sha256=sha(source), launch_command=command,
                   arm_id=proposal['arm_id'], scientific_patch=proposal['single_causal_change'],
                   completed_proposal_path=proposal_rel, completed_proposal_sha256=sha(proposal_raw),
                   operator_assignment_sha256=sha(assignment), driver_sha256=sha(driver.encode()),
                   independent_review_sha256=None, scalar_scope_sha256=sha(scope),
                   no_new_logic_human_scalar_authorization=True)
    binding['actual_same_host_comparator']['comparison'] = proposal['control']['comparison']
    files = dict(originals)
    files['PARENT_WARMDOWN_REVIEW_ONLY.md'] = originals['EXACT_SOURCE_REVIEW.md']
    files.update({'host_driver.py':driver.encode(),'package/train.py':source,
                  'package/launch-command.txt':command_raw,'package/provenance.json':provenance_raw,
                  'EXPERIMENT_PROPOSAL.json':proposal_raw,'CPU_RECEIPT.json':cpu,
                  'EXACT_SOURCE_REVIEW.md':scope,'SUPERVISOR_ASSIGNMENT.md':assignment,
                  'NATIVE_BINDING.json':encode(binding)})
    release = json.loads(originals['RELEASE.json'])
    release.update(label=label,source_sha256=sha(source),assignment_path=str((SPEC/'SPEC.md').relative_to(REPO)),
                   assignment_sha256=sha(assignment),independent_review_sha256=None,
                   scalar_scope_sha256=sha(scope),human_selected_scalar=True)
    files['RELEASE.json']=encode(release)
    for name in ('lifetime_guard.py','idle_guard.py','dispatch_driver.py'):
        assert files[name] == originals[name]
    packet=encode({'binding':binding,'files':{k:{'sha256':sha(v),'b64':base64.b64encode(v).decode()} for k,v in files.items()},
                   'mechanical_derivation':{'parent_packet_sha256':sha(original_raw),'driver_reverse_exact':True,
                                            'scientific_source_scalar_only':True,'fresh_independent_review':False}})
    write_once(JOBS/key/'EXPERIMENT_PROPOSAL.json',proposal_raw)
    write_once(JOBS/key/'packet.json',packet)
    manifest[key]={'host':host,'order':5,'predecessor':host+('-warmdown060' if host=='trn21' else '-warmdown085'),
                   'label':label,'packet':key+'/packet.json','packet_sha256':sha(packet),'source_sha256':sha(source),
                   'command':command,'source_review':'PARENT_LOGIC_UNCHANGED_HUMAN_SCALAR_CPU_PASS',
                   'machine_admission':'PENDING','status':'READY_NOT_LAUNCHED'}
manifest_path.write_bytes(encode(manifest))
print(json.dumps({'prepared':['trn21-muon150','trn22-muon150'],'source_sha256':sha(source),'host_calls':0}))
