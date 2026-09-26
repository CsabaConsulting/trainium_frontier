"""Local-only literal rebinding of the two actually executed full/eval packets."""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path('/home/poojith-devan/Desktop/Trainium')
BASE_SHA = '8c6d7af73e3237690c7f9a61c24544393b21224996d60484502ab379303c34ee'
SOURCE_SHA = '4ec8b1b46562612cdcbfe3d7f66b0c8d25666bdbac3516b30bd9149c2dc08d2d'
ASSIGNMENT = 'research/v2/assignments/20260926-adopted-batch-warmdown-brackets.md'
ASSIGNMENT_SHA = '79edb9bc639e996488d899604895dd7f0ed53e0c0dd3f8d281b95bb1b326a824'
REVIEW_SHA = 'b8b22eb3c8384fc34ff18a8f04c012ddcb2f1ce6671dbba26a5077f8d4ceab68'
CPU_SHA = '28799a7ac2fd3cd89fea92712fd42823db20df9fdf7320f3d452e9d71e998f67'
CONFIGS = (
    ('trn21', '060', 'cac28be6762a4b5649f7f2d1672fb85d5a86dae6648751dde7a5e9472e9ad036',
     'trn21-batch-recursive-stack-bound-packet.json', 'e1e04fe22f6e6cc5dc6b1e4109d8185bdc19473de5e70a85c2b8941007a888da'),
    ('trn22', '085', '41fadd39b29f3e242e44fcc3b4c1911105b69156956585a608b2414658b190b5',
     'trn22-batch-lr02625-bound-packet.json', 'da8e93292edaa0eca5c16a6aead0e40f767b2ff82ac5d0d30461d0531c046160'),
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2) + '\n').encode()


def save_new(path, raw):
    with path.open('xb') as stream:
        stream.write(raw)


assignment = (REPO / ASSIGNMENT).read_bytes()
assert sha(assignment) == ASSIGNMENT_SHA
summaries = []
for host, ratio, proposal_sha, parent_name, parent_sha in CONFIGS:
    proposal_path = f'research/v2/proposals/20260926-batch-lr021-warmdown{ratio}/EXPERIMENT_PROPOSAL.json'
    proposal_raw = (REPO / proposal_path).read_bytes()
    assert sha(proposal_raw) == proposal_sha
    p = json.loads(proposal_raw)
    source = (REPO / p['candidate']['source']).read_bytes()
    assert sha(source) == p['candidate']['source_sha256'] == SOURCE_SHA
    assert p['parent']['source_sha256'] == BASE_SHA
    review = (REPO / p['bound_evidence']['review']).read_bytes()
    cpu = (REPO / p['bound_evidence']['cpu_receipt']).read_bytes()
    assert sha(review) == p['bound_evidence']['review_sha256'] == REVIEW_SHA
    assert sha(cpu) == p['bound_evidence']['cpu_receipt_sha256'] == CPU_SHA
    assert p['dispatch']['selected_host'] == host
    label = p['dispatch']['selected_label']
    assert label == f'fs-20260926-batch-lr021-warmdown{ratio}-{host}-r1'
    assert p['candidate']['seed'] == 42 and p['candidate']['max_train_seconds'] == 1800
    assert p['candidate']['matrix_lr'] == .021 and p['candidate']['total_batch_size'] == 262144
    command = p['candidate']['launch_command']
    command_raw = (command + '\n').encode()
    assert sha(command.encode()) == p['candidate']['launch_string_sha256']
    assert sha(command_raw) == p['candidate']['launch_command_sha256']
    assert '--recursive-schedule' not in command
    assert command.endswith('--matrix-lr 0.021 --warmdown-ratio ' + ('0.60' if host == 'trn21' else '0.85'))

    parent_raw = (HERE.parent / '20260926-final-sprint-successors' / parent_name).read_bytes()
    assert sha(parent_raw) == parent_sha
    parent = json.loads(parent_raw)
    old = parent['binding']
    files = {n: base64.b64decode(v['b64'], validate=True) for n, v in parent['files'].items()}
    for name, raw in files.items():
        assert sha(raw) == parent['files'][name]['sha256'], name
    assert old['host'] == host and old['budget']['source_seconds'] == 1800
    root = str(Path(old['root']).parent / (label + '-packet'))
    cold = host == 'trn21'
    provenance = {
        'label': label, 'train_sha256': SOURCE_SHA, 'base_sha256': BASE_SHA,
        'kind': 'schedule_arm', 'run_purpose': 'quality', 'launch_command': command,
        'same_host_scientific_comparator': p['control'],
        'cache_mode': 'NEW_PRIVATE_COLD_TRAINING_EXISTING_RESEARCH_EVAL' if cold else 'EXISTING_RESEARCH_ROOTS_NO_HIT_OR_COLD_CLAIM',
        'submission_authorized': False,
    }
    if not cold:
        provenance['cache_owner_label'] = old['cache_owner_label']
    provenance_raw = encoded(provenance)
    replacements = [
        (old['root'], root), (old['label'], label), (old['source_sha256'], SOURCE_SHA),
        (sha(files['package/launch-command.txt']), sha(command_raw)),
        (sha(files['package/provenance.json']), sha(provenance_raw)),
    ]
    old_driver = files['host_driver.py'].decode()
    driver = old_driver
    for before, after in replacements:
        assert before in driver, before
        driver = driver.replace(before, after)
    reverse = driver
    for before, after in reversed(replacements):
        reverse = reverse.replace(after, before)
    assert reverse == old_driver, 'DRIVER_NOT_EXACT_LITERAL_REBIND'
    ast.parse(driver)

    def transform(value):
        if isinstance(value, str):
            for before, after in replacements:
                value = value.replace(before, after)
            return value
        if isinstance(value, list):
            return [transform(v) for v in value]
        if isinstance(value, dict):
            return {k: transform(v) for k, v in value.items()}
        return value

    b = transform(copy.deepcopy(old))
    b.update(
        schema='prepared-batch-lr021-warmdown-full-v1', source_sha256=SOURCE_SHA,
        base_sha256=BASE_SHA, source_commit=p['candidate']['source_commit'], kind='schedule_arm',
        run_purpose='quality', arm_id=p['arm_id'], launch_command=command,
        scientific_patch='ordinary_warmdown_' + ratio, stage_authorized=False,
        launch_authorized=False, operator_assignment_sha256=ASSIGNMENT_SHA,
        independent_review_sha256=REVIEW_SHA, completed_proposal_path=proposal_path,
        completed_proposal_sha256=proposal_sha, driver_sha256=sha(driver.encode()),
        actual_same_host_comparator=p['control'], source_adoption=False,
        submission_authorized=False, release_file_created=False, host_reserved=False,
        runtime_pins=None, actual_host_facts=None, no_automatic_next_job=True,
        selected_cache_policy=p['selected_cache_policy'],
    )
    if cold:
        b['cold_bound_driver_sha256'] = sha(driver.encode())
        assert len(b['fresh_cache_directories']) == len(set(b['fresh_cache_directories'])) == 6
        assert all(path.startswith(root + '/cold-training-caches/') for path in b['fresh_cache_directories'])
    else:
        assert not b['fresh_cache_directories']
        for phase in ('train_environment', 'eval_environment'):
            for key, value in old[phase].items():
                if 'CACHE' in key or key == 'TORCH_NEURONX_DEBUG_DIR':
                    assert b[phase][key] == value, (phase, key)
    assert b['eval_environment'] == dict(old['eval_environment'], TRN_EXP_LABEL=label)
    assert b['lock_identity'] == [66305, 596823 if cold else 832589]
    assert b['budget'] == old['budget']
    assert b['resource_reserve'] == old['resource_reserve']
    assert b['resource_reserve']['initial_floor_bytes'] >= 14495514624
    assert b['resource_reserve']['runtime_floor_bytes'] >= 11274289152
    assert b['prepare_sha256'] == old['prepare_sha256'] and b['harness_sha256'] == old['harness_sha256']
    files = {n: v for n, v in files.items() if n not in ('RELEASE.json', 'NATIVE_BINDING.json', 'NATIVE_BINDING.disabled.json')}
    files.update({
        'host_driver.py': driver.encode(), 'package/train.py': source,
        'package/launch-command.txt': command_raw, 'package/provenance.json': provenance_raw,
        'EXPERIMENT_PROPOSAL.json': proposal_raw, 'EXACT_SOURCE_REVIEW.md': review,
        'CPU_RECEIPT.json': cpu, 'SUPERVISOR_ASSIGNMENT.md': assignment,
        'NATIVE_BINDING.disabled.json': encoded(b),
    })
    for name in ('lifetime_guard.py', 'idle_guard.py', 'dispatch_driver.py'):
        assert sha(files[name]) == parent['files'][name]['sha256']
    packet = {
        'binding': b,
        'files': {n: {'sha256': sha(raw), 'b64': base64.b64encode(raw).decode()} for n, raw in files.items()},
        'local_only': True, 'host_calls': 0,
        'mechanical_derivation': {
            'executed_parent_packet_sha256': parent_sha, 'driver_literal_reverse_exact': True,
            'driver_logic_changed': False, 'source_modified': False,
            'guardian_prepare_harness_eval_unchanged': True,
            'cold_private_training_caches' if cold else 'all_train_eval_cache_paths_unchanged': True,
        },
    }
    out = HERE / f'{host}-warmdown{ratio}-prepared.disabled.json'
    save_new(out, encoded(packet))
    summaries.append({
        'host': host, 'label': label, 'status': 'LOCAL_MECHANICAL_PASS_NOT_STAGED_NOT_LAUNCHED',
        'packet': out.name, 'packet_sha256': sha(out.read_bytes()), 'member_count': len(files),
        'source_sha256': SOURCE_SHA, 'proposal_sha256': proposal_sha, 'review_sha256': REVIEW_SHA,
        'command_file_sha256': sha(command_raw), 'driver_sha256': sha(driver.encode()),
        'parent_packet_sha256': parent_sha, 'host_calls': 0,
        'source_review_repeated': False, 'native_gate_claimed': False,
        'initial_floor_bytes': b['resource_reserve']['initial_floor_bytes'],
        'runtime_floor_bytes': b['resource_reserve']['runtime_floor_bytes'],
        'memory_available_minimum_bytes': 68719476736,
        'cache_policy': p['selected_cache_policy'],
    })
save_new(HERE / 'WARMDOWN_PREPARED.json', encoded(summaries))
print(json.dumps(summaries, indent=2))
