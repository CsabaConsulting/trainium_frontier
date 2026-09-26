"""Prepare six flag-only fallback jobs from the executed/reviewed full-run machinery.

Local mechanical packaging only. No SSH, source edits, new model or new gate.
"""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DEST = HERE / 'manual-jobs'
SOURCE = '4ec8b1b46562612cdcbfe3d7f66b0c8d25666bdbac3516b30bd9149c2dc08d2d'
PARENTS = {
    'trn21': ('trn21-warmdown060-prepared.disabled.json', 'f402300af29399caf6aff0be6f9eb186a659a76679ff21ee05b5baa438582ab3'),
    'trn22': ('trn22-warmdown085-prepared.disabled.json', '61e3356f8c3fe0d1e30927bd1b64dc289d5e06724c1cfd0d7c90da40057eafb1'),
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2) + '\n').encode()


def write_once(path, raw):
    if path.name == 'manifest.json':
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)  # Mechanical generated index, not an input receipt.
        return
    if path.exists():
        assert path.read_bytes() == raw, 'EXISTING_ARTIFACT_DRIFT:' + str(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)


frozen = DEST / 'trn21-warmdown060/packet.json'
assignment = (base64.b64decode(json.loads(frozen.read_bytes())['files']['SUPERVISOR_ASSIGNMENT.md']['b64'])
              if frozen.exists() else (REPO / 'research/v2/LOW_USAGE_CONTROLLER.md').read_bytes())
entries = json.loads((DEST / 'manifest.json').read_bytes()) if (DEST / 'manifest.json').exists() else {}
for host in PARENTS:
    filename, expected = PARENTS[host]
    raw = (HERE / filename).read_bytes()
    assert sha(raw) == expected
    parent = json.loads(raw)
    old = parent['binding']
    originals = {k: base64.b64decode(v['b64'], validate=True) for k, v in parent['files'].items()}
    for name, value in originals.items():
        assert sha(value) == parent['files'][name]['sha256']
    assert sha(originals['package/train.py']) == SOURCE
    assert sha(originals['EXACT_SOURCE_REVIEW.md']) == 'b8b22eb3c8384fc34ff18a8f04c012ddcb2f1ce6671dbba26a5077f8d4ceab68'
    assert sha(originals['CPU_RECEIPT.json']) == '28799a7ac2fd3cd89fea92712fd42823db20df9fdf7320f3d452e9d71e998f67'
    ratios = ('0.60', '0.85') if host == 'trn21' else ('0.85', '0.60')
    for pos, (ratio, lr) in enumerate([(ratios[0], '0.021'), (ratios[1], '0.021'), ('0.75', '0.0168'), ('0.75', '0.021')], 1):
        suffix = 'warmdown' + ratio.replace('.', '') if pos < 3 else 'lr0168' if pos == 3 else 'control'
        key = host + '-' + suffix
        label = 'fs-20260926-batch-lr021-' + suffix + '-' + host + '-r1' if pos < 3 else 'fs-20260926-batch262144-lr0168-' + host + '-r1'
        if pos == 4:
            label = 'fs-20260926-batch262144-lr021-control-' + host + '-r1'
        root = str(Path(old['root']).parent / (label + '-packet'))
        proposal = json.loads(originals['EXPERIMENT_PROPOSAL.json'])
        command = proposal['candidate']['launch_command'].rsplit('--matrix-lr', 1)[0] + '--matrix-lr ' + lr + ' --warmdown-ratio ' + ratio
        command_raw = (command + '\n').encode()
        proposal.update(proposal_id='20260926-manual-' + key, arm_id='batch262144-' + suffix,
                        status='COMPLETE_FLAG_ONLY_BOUND_PENDING_MACHINE_ADMISSION',
                        authority='research/v2/LOW_USAGE_CONTROLLER.md; latest human FINAL SPRINT LR/warmdown brackets',
                        single_causal_change=('ordinary warmdown .75 -> ' + ratio if pos < 3 else 'matrix LR .021 -> .0168; ordinary warmdown .75 unchanged'),
                        hypothesis='Measure the fixed-budget generalization effect of this one scalar on the confirmed batch/LR base.',
                        local_scope='Mechanically bound fallback proposal; scientific source and existing independent review unchanged.')
        proposal['candidate'].update(launch_command=command, launch_string_sha256=sha(command.encode()),
                                     launch_command_sha256=sha(command_raw), argv=command.split('train.py ', 1)[1].split(),
                                     warmdown_ratio=float(ratio), matrix_lr=float(lr))
        proposal['dispatch'].update(selected_host=host, selected_label=label, ready=True,
                                    stage_authorized=True, launch_authorized=True,
                                    operator='Current single root/host Operator or Poojith manual fallback',
                                    order='Fallback order per MANUAL_NEXT_JOBS; qualified replicate/new measured base takes priority.')
        proposal['kind'] = 'schedule_arm' if pos < 3 else 'lr_arm'
        if pos == 4:
            proposal.update(kind='control', arm_id='batch262144-lr021-prospective-control',
                            single_causal_change='None: genuine prospectively labelled control for adopted8c6, using reviewed default-preserving4ec8 source.',
                            hypothesis='Supply a genuine current-base control for unchanged decide.py; never retrospectively relabel previous batch-arm records.')
        proposal['control']['comparison'] = 'Actual same-host confirmed batch262144/LR.021/ordinary warmdown.75; existing ledger roles retained.'
        proposal['success_criteria']['submission'] = 'Research-only; final candidate needs genuine cold replay, fresh20M and Claude audit; Poojith uploads.'
        proposal['selected_cache_policy'] = 'Inherited original host full-run cache policy; see binding and actual startup. No warm submission claim.'
        proposal_rel = 'research/v2/operator/20260926-operator-recovery/manual-jobs/' + key + '/EXPERIMENT_PROPOSAL.json'
        proposal_raw = encode(proposal)
        provenance = json.loads(originals['package/provenance.json'])
        provenance.update(label=label, kind=proposal['kind'], launch_command=command,
                          manual_fallback=True, submission_authorized=False)
        provenance_raw = encode(provenance)
        replacements = [(old['root'], root), (old['label'], label),
                        (sha(originals['package/launch-command.txt']), sha(command_raw)),
                        (sha(originals['package/provenance.json']), sha(provenance_raw))]
        replacements = [(a, b) for a, b in replacements if a != b]

        def transform(value):
            if isinstance(value, str):
                for before, after in replacements:
                    value = value.replace(before, after)
                return value
            if isinstance(value, dict):
                return {k: transform(v) for k, v in value.items()}
            if isinstance(value, list):
                return [transform(v) for v in value]
            return value

        driver = transform(originals['host_driver.py'].decode())
        reverse = driver
        for before, after in reversed(replacements):
            reverse = reverse.replace(after, before)
        assert reverse == originals['host_driver.py'].decode(), 'DRIVER_LOGIC_CHANGED'
        ast.parse(driver)
        binding = transform(copy.deepcopy(old))
        binding.update(stage_authorized=True, launch_authorized=True, submission_authorized=False,
                       driver_sha256=sha(driver.encode()), kind=proposal['kind'], arm_id=proposal['arm_id'],
                       launch_command=command, completed_proposal_path=proposal_rel,
                       completed_proposal_sha256=sha(proposal_raw), operator_assignment_sha256=sha(assignment),
                       scientific_patch=proposal['single_causal_change'])
        files = {k: v for k, v in originals.items() if k not in ('NATIVE_BINDING.disabled.json', 'NATIVE_BINDING.json', 'RELEASE.json')}
        files.update({'host_driver.py': driver.encode(), 'package/launch-command.txt': command_raw,
                      'package/provenance.json': provenance_raw, 'EXPERIMENT_PROPOSAL.json': proposal_raw,
                      'SUPERVISOR_ASSIGNMENT.md': assignment, 'NATIVE_BINDING.json': encode(binding),
                      'RELEASE.json': encode({'label': label, 'launch_authorized': True, 'submission_authorized': False,
                          'assignment_path': 'research/v2/LOW_USAGE_CONTROLLER.md', 'assignment_sha256': sha(assignment),
                          'independent_review_sha256': sha(files['EXACT_SOURCE_REVIEW.md']), 'source_sha256': SOURCE,
                          'guardian_sha256': sha(files['lifetime_guard.py']), 'no_automatic_next_job': True})})
        assert binding['lock_identity'] == [66305, 596823 if host == 'trn21' else 832589]
        for name in ('lifetime_guard.py', 'idle_guard.py', 'dispatch_driver.py', 'package/train.py'):
            assert files[name] == originals[name]
        packet = encode({'binding': binding, 'files': {k: {'sha256': sha(v), 'b64': base64.b64encode(v).decode()} for k, v in files.items()},
                         'mechanical_derivation': {'parent_packet_sha256': expected, 'driver_reverse_exact': True, 'scientific_source_unchanged': True}})
        write_once(DEST / key / 'EXPERIMENT_PROPOSAL.json', proposal_raw)
        write_once(DEST / key / 'packet.json', packet)
        entries[key] = {'host': host, 'order': pos, 'label': label, 'packet': key + '/packet.json',
                        'packet_sha256': sha(packet), 'source_sha256': SOURCE, 'command': command,
                        'source_review': 'PASS_SHARED_PATCH', 'machine_admission': 'PENDING', 'status': 'READY_FALLBACK_NOT_LAUNCHED',
                        'priority_rule': 'Qualifying cross-host replica first; do not continue old-base fallback after new-base adoption.'}
write_once(DEST / 'manifest.json', encode(entries))
print(json.dumps({'prepared': list(entries), 'host_calls': 0, 'scientific_source_edits': 0}))
