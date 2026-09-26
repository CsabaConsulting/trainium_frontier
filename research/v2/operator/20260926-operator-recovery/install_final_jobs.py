"""Register human-requested final scalar/seed transports; no host calls."""
import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
JOBS = HERE / 'manual-jobs'


def sha(raw): return hashlib.sha256(raw).hexdigest()
def enc(obj): return (json.dumps(obj, indent=2) + '\n').encode()


assignment = (REPO / 'research/v2/LOW_USAGE_CONTROLLER.md').read_bytes()
scope = b'''Final-package source scope: human-selected two baked constants on accepted e1e2.
Actual CPU identity receipt covers effective CLI and reduced model/optimizer;
this is NOT a fresh independent review, native cold result or Claude audit.
Seed43 is a human-requested variance diagnostic, not a chosen best seed.
Full native cold/fresh20M and later Claude audit remain required for upload.
'''
manifest_path = JOBS / 'manifest.json'
manifest = json.loads(manifest_path.read_bytes())
rows = [
    ('trn21-finalcold', 'workflow/candidates/final-batch262144-lr021/cold-packet.disabled.json', 'trn21-k2rescue'),
    ('trn22-finalseed43', 'research/v2/proposals/20260926-final-second-seed/packet.disabled.json', 'trn22-muon150'),
]
for key, path, predecessor in rows:
    original = json.loads((REPO / path).read_bytes())
    binding = original['binding']
    files = {k: base64.b64decode(v['b64'], validate=True) for k, v in original['files'].items()}
    for name, raw in files.items():
        if sha(raw) != original['files'][name]['sha256']: raise ValueError('MEMBER_DRIFT:'+name)
    if binding['source_sha256'] != 'cd0c4aa59754f46705355d964fad4528cf1de77324469b0829bcc0de149b757d':
        raise ValueError('FINAL_SOURCE_WRONG')
    folder = JOBS / key
    if any((folder / n).exists() for n in ('attempt.json', 'dispatch.json', 'results-bundle.json')):
        raise ValueError('CONSUMED_FINAL_JOB')
    binding.update(stage_authorized=True, launch_authorized=True, independent_review_sha256=None,
                   operator_assignment_sha256=sha(assignment), scalar_scope_sha256=sha(scope),
                   no_new_logic_human_scalar_authorization=True,
                   completed_proposal_path=str((folder / 'EXPERIMENT_PROPOSAL.json').relative_to(REPO)))
    proposal = json.loads(files['EXPERIMENT_PROPOSAL.json'])
    proposal['status'] = 'HUMAN_SCALAR_CPU_PASS_PENDING_FRESH_MACHINE_ADMISSION'
    proposal['dispatch'].update(stage_authorized=True, launch_authorized=True)
    proposal_raw = enc(proposal)
    binding['completed_proposal_sha256'] = sha(proposal_raw)
    files.pop('NATIVE_BINDING.disabled.json', None)
    files.update({'NATIVE_BINDING.json': enc(binding), 'EXPERIMENT_PROPOSAL.json': proposal_raw,
                  'EXACT_SOURCE_REVIEW.md': scope, 'SUPERVISOR_ASSIGNMENT.md': assignment,
                  'RELEASE.json': enc({'label': binding['label'], 'launch_authorized': True,
                                      'submission_authorized': False, 'source_sha256': binding['source_sha256'],
                                      'independent_review_sha256': None, 'scalar_scope_sha256': sha(scope),
                                      'assignment_sha256': sha(assignment), 'no_automatic_next_job': True})})
    packet = enc({'binding': binding, 'files': {k: {'sha256': sha(v), 'b64': base64.b64encode(v).decode()} for k, v in files.items()},
                  'derivation': original['derivation'], 'host_calls': 0})
    folder.mkdir(exist_ok=True)
    for name, raw in [('packet.json', packet), ('EXPERIMENT_PROPOSAL.json', proposal_raw)]:
        dest = folder / name
        if dest.exists() and dest.read_bytes() != raw: raise ValueError('EXISTING_FINAL_PREPARATION_DRIFT:'+name)
        if not dest.exists():
            with dest.open('xb') as stream: stream.write(raw)
    manifest[key] = {'host': binding['host'], 'order': 6, 'predecessor': predecessor,
                     'label': binding['label'], 'packet': key+'/packet.json', 'packet_sha256': sha(packet),
                     'source_sha256': binding['source_sha256'], 'command': binding['launch_command'],
                     'status': 'READY_PENDING_PREDECESSOR_AND_FRESH_MACHINE',
                     'source_review': 'HUMAN_SCALAR_SCOPE_NOT_FRESH_AUDIT', 'machine_admission': 'PENDING'}
    if key == 'trn22-finalseed43':
        manifest[key]['requires_cold_bundle'] = str((JOBS / 'trn21-finalcold/results-bundle.json').relative_to(REPO))
manifest_path.write_bytes(enc(manifest))
print(json.dumps({'prepared': [x[0] for x in rows], 'host_calls': 0, 'submissions': 0}))
