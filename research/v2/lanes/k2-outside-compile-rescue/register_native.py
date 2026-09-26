"""Register ONE rescue native201 only after actual bounded review; no SSH."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
JOBS = REPO / 'research/v2/operator/20260926-operator-recovery/manual-jobs'


def sha(raw): return hashlib.sha256(raw).hexdigest()
def enc(obj): return (json.dumps(obj, indent=2) + '\n').encode()


parser = argparse.ArgumentParser()
parser.add_argument('--review', type=Path)
parser.add_argument('--receipt', type=Path)
parser.add_argument('--human-timing-only', action='store_true')
args = parser.parse_args()
source = (HERE / 'train.py').read_bytes()
if args.human_timing_only:
    if args.review or args.receipt: raise ValueError('NO_FABRICATED_REVIEW_WITH_WAIVER')
    review = b'Poojith direct: launch research-only K2 timing NOW; independent review deferred, never submission authority.\n'
    receipt = {'status': 'DEFERRED_BY_HUMAN_RESEARCH_TIMING_ONLY', 'source_sha256': sha(source),
               'fresh_context': False, 'reviewer_id': None, 'independent_cpu_test_exit_code': None,
               'human_timing_only_authority': True, 'submission_authorized': False}
    receipt_raw = enc(receipt)
else:
    if not args.review or not args.receipt: parser.error('review+receipt or explicit human-timing-only required')
    review, receipt_raw = args.review.read_bytes(), args.receipt.read_bytes()
    receipt = json.loads(receipt_raw)
if not args.human_timing_only and (receipt['status'] != 'PASS_NATIVE_TIMING_SCOPE' or receipt['source_sha256'] != sha(source)):
    raise ValueError('REVIEW_REFUSE_OR_WRONG_SOURCE')
if not args.human_timing_only and (receipt['fresh_context'] is not True or receipt['reviewer_id'] == '/root' or receipt['independent_cpu_test_exit_code'] != 0):
    raise ValueError('INDEPENDENT_REFERENCE_NOT_PASSED')
for key, name in [('diff_sha256', 'SOURCE.diff'), ('spec_sha256', 'SPEC.md'), ('cpu_receipt_sha256', 'CPU_RECEIPT.json')]:
    if not args.human_timing_only and receipt[key] != sha((HERE / name).read_bytes()): raise ValueError('REVIEW_INPUT_DRIFT:'+name)
if not args.human_timing_only and receipt['review_sha256'] != sha(review): raise ValueError('REVIEW_REPORT_DRIFT')
original = json.loads((HERE / 'native201.disabled.json').read_bytes())
binding = original['binding']
files = {k: base64.b64decode(v['b64'], validate=True) for k, v in original['files'].items()}
for k, v in files.items():
    if sha(v) != original['files'][k]['sha256']: raise ValueError('PACKET_MEMBER_DRIFT:'+k)
if sha(source) != binding['source_sha256']: raise ValueError('NATIVE_SOURCE_DRIFT')
assignment = (REPO / 'research/v2/LOW_USAGE_CONTROLLER.md').read_bytes()
key = 'trn21-k2rescue'
folder = JOBS / key
if folder.exists(): raise ValueError('K2_ALREADY_REGISTERED_NO_RETRY')
proposal = json.loads(files['EXPERIMENT_PROPOSAL.json'])
proposal['status'] = 'HUMAN_RESEARCH_TIMING_ONLY_REVIEW_DEFERRED' if args.human_timing_only else 'CPU_AND_ONE_INDEPENDENT_REVIEW_PASS_PENDING_LOCKED_HOST_ADMISSION'
proposal['dispatch'].update(stage_authorized=True, launch_authorized=True)
proposal['local_evidence']['independent_review'] = 'DEFERRED_BY_HUMAN_TIMING_ONLY' if args.human_timing_only else sha(review)
proposal['human_timing_only_authority'] = args.human_timing_only
proposal_raw = enc(proposal)
binding.update(stage_authorized=True, launch_authorized=True, independent_review_sha256=sha(review),
               operator_assignment_sha256=sha(assignment), completed_proposal_sha256=sha(proposal_raw),
               completed_proposal_path=str((folder / 'EXPERIMENT_PROPOSAL.json').relative_to(REPO)))
if args.human_timing_only:
    binding.update(independent_review_sha256=None, human_timing_only_authority=True, review_required_before_submission=True)
files.pop('NATIVE_BINDING.disabled.json', None)
files.update({'NATIVE_BINDING.json': enc(binding), 'EXPERIMENT_PROPOSAL.json': proposal_raw,
              'EXACT_SOURCE_REVIEW.md': review, 'INDEPENDENT_REVIEW_RECEIPT.json': receipt_raw,
              'SUPERVISOR_ASSIGNMENT.md': assignment,
              'RELEASE.json': enc({'label': binding['label'], 'launch_authorized': True,
                                  'submission_authorized': False, 'source_sha256': binding['source_sha256'],
                                  'independent_review_sha256': None if args.human_timing_only else sha(review),
                                  'human_timing_only_authority': args.human_timing_only, 'assignment_sha256': sha(assignment),
                                  'no_automatic_next_job': True})})
packet = enc({'binding': binding, 'files': {k: {'sha256': sha(v), 'b64': base64.b64encode(v).decode()} for k, v in files.items()},
              'derivation': original['derivation'], 'host_calls': 0})
folder.mkdir()
for name, raw in [('packet.json', packet), ('EXPERIMENT_PROPOSAL.json', proposal_raw)]:
    with (folder / name).open('xb') as stream: stream.write(raw)
manifest_path = JOBS / 'manifest.json'
manifest = json.loads(manifest_path.read_bytes())
manifest[key] = {'host': 'trn21', 'order': 7, 'predecessor': 'trn21-warmdown060',
                 'label': binding['label'], 'packet': key+'/packet.json', 'packet_sha256': sha(packet),
                 'source_sha256': binding['source_sha256'], 'command': binding['launch_command'],
                 'status': proposal['status'], 'no_eval': True}
manifest_path.write_bytes(enc(manifest))
queue_path = REPO / 'research/v2/queue.json'
queue = json.loads(queue_path.read_bytes())
q = queue['current_low_usage_queue']
q['trn21'] = [key] + [k for k in q['trn21'] if k != key]
q['predecessors'][key] = 'trn21-warmdown060'
q['priority_reservation']['trn21'] = 'Human timing-only review waiver; next guarded dispatch' if args.human_timing_only else 'K2 review passed; next guarded dispatch'
queue_path.write_bytes(enc(queue))
print(json.dumps({'job': key, 'source_sha256': sha(source),
                  'review_sha256': None if args.human_timing_only else sha(review),
                  'timing_authority_sha256': sha(review) if args.human_timing_only else None, 'host_calls': 0}))
