"""Bind the existing cold full-run transport to the clean fallback; no SSH/release.

This is packaging, not a new launcher. It deliberately stays outside the live
manual-job manifest/queue until the chosen final source is frozen.
"""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PARENT = REPO / 'research/v2/operator/20260926-operator-recovery/trn21-warmdown060-prepared.disabled.json'
SOURCE_SHA = 'cd0c4aa59754f46705355d964fad4528cf1de77324469b0829bcc0de149b757d'
COMMAND_SHA = 'b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(obj):
    return (json.dumps(obj, indent=2) + '\n').encode()


def main():
    raw = PARENT.read_bytes()
    if sha(raw) != 'f402300af29399caf6aff0be6f9eb186a659a76679ff21ee05b5baa438582ab3':
        raise ValueError('PARENT_PACKET_CHANGED')
    parent = json.loads(raw)
    original = {k: base64.b64decode(v['b64'], validate=True) for k, v in parent['files'].items()}
    for name, value in original.items():
        if sha(value) != parent['files'][name]['sha256']:
            raise ValueError('PARENT_MEMBER_CHANGED:' + name)
    source, command, cpu = ((HERE / name).read_bytes() for name in ('train.py', 'launch-command.txt', 'CPU_RECEIPT.json'))
    if sha(source) != SOURCE_SHA or sha(command) != COMMAND_SHA:
        raise ValueError('FINAL_SOURCE_OR_ACCEPTED_COMMAND_CHANGED')
    receipt = json.loads(cpu)
    if receipt['status'] != 'PASS' or receipt['candidate_sha256'] != SOURCE_SHA or receipt['launch_command_sha256'] != COMMAND_SHA:
        raise ValueError('CPU_RECEIPT_WRONG_SOURCE')
    old = parent['binding']
    label = 'fs-20260926-final-batch262144-lr021-cold-trn21-r1'
    root = str(Path(old['root']).parent / (label + '-packet'))
    proposal = {
        'proposal_id': label, 'arm_id': 'final-batch262144-lr021-cold',
        'kind': 'submission_validation', 'run_purpose': 'quality',
        'status': 'PREPARED_NOT_QUEUED_OR_RELEASED',
        'authority': 'Poojith: confirmed batch/LR is final candidate unless beaten; cold replay + Claude audit before human upload',
        'hypothesis': 'Clean two-constant port preserves confirmed research configuration under actual cold accounting.',
        'single_causal_change': 'No scientific change versus confirmed batch/LR; final clean source and cold-cache validation.',
        'candidate': {'source_sha256': SOURCE_SHA, 'launch_command': command.decode().strip(),
                      'launch_command_sha256': COMMAND_SHA, 'seed': 42, 'max_train_seconds': 1800,
                      'total_batch_size': 262144, 'matrix_lr': 0.021, 'warmdown_ratio': 0.75,
                      'muon_momentum_warmup_steps': 300},
        'control': copy.deepcopy(old['actual_same_host_comparator']),
        'evaluation': {'fresh_process': True, 'mode': 'unchanged prepare.py eval-public',
                       'tokens': 20971520, 'context': 1024, 'batch_size': 1},
        'accounting': {'training_budget_seconds': 1800, 'startup_cap_seconds': 900,
                       'require_over_budget_false': True, 'record_cold_startup_and_charge': True},
        'cache_policy': 'Unique empty private training caches proven by original preflight; warm eval cache does not waive fresh process.',
        'source_evidence': {'cpu_receipt_sha256': sha(cpu), 'scope': receipt['scope'],
                            'fresh_independent_review': False, 'claude_audit_passed': False},
        'dispatch': {'selected_host': 'trn21', 'selected_label': label,
                     'stage_authorized': False, 'launch_authorized': False,
                     'order': 'Final validation slot after selecting best stack; never preempt current warmdown/momentum work.'},
        'submission_authorized': False,
        'pending': ['Final-source selection', 'Fresh locked machine admission',
                    'Actual cold full run and fresh20M result', 'Selected-stack second-seed confirmation', 'Claude audit'],
    }
    proposal['control']['comparison'] = 'Clean-port validation versus confirmed batch/LR source8c6; no warmdown change.'
    proposal_raw = encode(proposal)
    provenance = json.loads(original['package/provenance.json'])
    provenance.update(label=label, kind=proposal['kind'], source_sha256=SOURCE_SHA,
                      train_sha256=SOURCE_SHA, same_host_scientific_comparator=proposal['control'],
                      source_commit=None, launch_command=command.decode().strip(), submission_authorized=False)
    provenance_raw = encode(provenance)
    changes = [(old['root'], root), (old['label'], label), (old['source_sha256'], SOURCE_SHA),
               (sha(original['package/launch-command.txt']), COMMAND_SHA),
               (sha(original['package/provenance.json']), sha(provenance_raw))]

    def transform(value):
        if isinstance(value, str):
            for before, after in changes:
                value = value.replace(before, after)
            return value
        if isinstance(value, dict):
            return {k: transform(v) for k, v in value.items()}
        if isinstance(value, list):
            return [transform(v) for v in value]
        return value

    driver = transform(original['host_driver.py'].decode())
    reversed_driver = driver
    for before, after in reversed(changes):
        reversed_driver = reversed_driver.replace(after, before)
    if reversed_driver != original['host_driver.py'].decode():
        raise ValueError('DRIVER_LOGIC_CHANGED')
    ast.parse(driver)
    binding = transform(copy.deepcopy(old))
    binding.update(schema='prepared-final-clean-cold-v1', source_commit=None,
                   launch_command=command.decode().strip(), kind=proposal['kind'], arm_id=proposal['arm_id'],
                   scientific_patch=proposal['single_causal_change'],
                   completed_proposal_path=str((HERE / 'COLD_PROPOSAL.json').relative_to(REPO)),
                   completed_proposal_sha256=sha(proposal_raw), driver_sha256=sha(driver.encode()),
                   independent_review_sha256=None, operator_assignment_sha256=None,
                   stage_authorized=False, launch_authorized=False, release_file_created=False,
                   actual_same_host_comparator=proposal['control'],
                   selected_cache_policy=proposal['cache_policy'])
    members = {k: v for k, v in original.items() if k not in (
        'NATIVE_BINDING.disabled.json', 'NATIVE_BINDING.json', 'RELEASE.json',
        'EXACT_SOURCE_REVIEW.md', 'SUPERVISOR_ASSIGNMENT.md')}
    members.update({'host_driver.py': driver.encode(), 'package/train.py': source,
                    'package/launch-command.txt': command, 'package/provenance.json': provenance_raw,
                    'EXPERIMENT_PROPOSAL.json': proposal_raw, 'CPU_RECEIPT.json': cpu,
                    'NATIVE_BINDING.disabled.json': encode(binding)})
    if not binding['fresh_private_training_cache'] or len(binding['fresh_cache_directories']) != 6:
        raise ValueError('COLD_CACHE_BINDING_MISSING')
    for name in ('lifetime_guard.py', 'idle_guard.py', 'dispatch_driver.py'):
        if members[name] != original[name]:
            raise ValueError('GUARDIAN_CHANGED')
    packet = encode({'binding': binding,
                     'files': {k: {'sha256': sha(v), 'b64': base64.b64encode(v).decode()} for k, v in members.items()},
                     'derivation': {'parent_packet_sha256': sha(raw), 'driver_reverse_exact': True,
                                    'host_calls': 0, 'no_manifest_or_queue_change': True}})
    for name, data in [('COLD_PROPOSAL.json', proposal_raw), ('cold-packet.disabled.json', packet)]:
        path = HERE / name
        if path.exists():
            if path.read_bytes() != data:
                if any((HERE / n).exists() for n in ('attempt.json', 'dispatch.json', 'results-bundle.json', 'RELEASE.json')):
                    raise ValueError('CONSUMED_PREPARATION_CHANGED:' + name)
                path.write_bytes(data)  # Mechanical regeneration of unconsumed local preparation only.
        else:
            with path.open('xb') as stream:
                stream.write(data)
    print(json.dumps({'source_sha256': SOURCE_SHA, 'command_sha256': COMMAND_SHA,
                      'packet_sha256': sha(packet), 'driver_reverse_exact': True,
                      'private_caches': 6, 'launch_authorized': False, 'host_calls': 0}))


if __name__ == '__main__':
    main()
