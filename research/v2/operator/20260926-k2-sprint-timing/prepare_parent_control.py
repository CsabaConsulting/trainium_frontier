"""Literal local-only ON45 rebind to the prospectively selected live parent."""
import ast
import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path('/home/poojith-devan/Desktop/Trainium')
def sha(x): return hashlib.sha256(x).hexdigest()
oldraw = (HERE/'trn22-on-prepared.json').read_bytes()
assert sha(oldraw) == '85d55cd4d6877a6fd0ff0f9a89b7a3d7ed5392803b465244b77952565fc7799d'
old = json.loads(oldraw)
original = {n:base64.b64decode(v['b64']) for n,v in old['files'].items()}
for n,v in original.items(): assert sha(v) == old['files'][n]['sha256']
assignment_path = 'research/v2/assignments/20260926-k2-use-live-parent-control.md'
assignment = (REPO/assignment_path).read_bytes()
assert b'35*131072/(t44-t9)' in assignment and b'14,495,514,624' in assignment
review = (REPO/'research/v2/verification/20260926-k2-sprint-independent/REVIEW.md').read_bytes()
assert sha(review) == '2ddb87ac3c38897abf306e323d06c747b0db41423e2f6126ede94e98cadcf502'
label = 'fs-20260926-k2-sprint-on-parentcontrol-trn22-r1'
oldlabel = old['binding']['label']
oldroot = old['binding']['root']
root = oldroot.replace(oldlabel,label)
assert label.encode() in assignment and oldlabel not in label
def rebind(value):
    if isinstance(value,str): return value.replace(oldroot,root).replace(oldlabel,label)
    if isinstance(value,list): return [rebind(x) for x in value]
    if isinstance(value,dict): return {k:rebind(v) for k,v in value.items()}
    return value
binding = rebind(old['binding'])
files = dict(original)
oldprov = original['package/provenance.json']
prov = rebind(json.loads(oldprov))
newprov = (json.dumps(prov,indent=2)+'\n').encode()
driver = original['host_driver.py'].decode().replace(oldroot,root).replace(oldlabel,label)
assert sha(oldprov) in driver
driver = driver.replace(sha(oldprov),sha(newprov)).encode()
assert ast.dump(ast.parse(driver)) == ast.dump(ast.parse(original['host_driver.py'].decode().replace(oldroot,root).replace(oldlabel,label).replace(sha(oldprov),sha(newprov))))
assert original['package/train.py'] == (REPO/'research/v2/lanes/k2-final-sprint/train.py').read_bytes()
assert sha(original['package/train.py']) == '63925519ac6c0aab1a1d598e96fef3f4413fbd083d9b6d177901a2e972060db5'
assert binding['launch_command'] == old['binding']['launch_command']
parent = json.loads((HERE.parent/'20260926-final-sprint-successors/trn22-e1e2-control-bound-packet.json').read_text())['binding']
assert parent['label'] == 'fs-20260926-sprint-e1e2-control-trn22-r1'
assert parent['source_sha256'] == 'e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43'
binding.update(
    schema='conditional-k2-on45-live-parent-v1',
    driver_sha256=sha(driver), operator_assignment_sha256=sha(assignment),
    proposal_amendment_path=assignment_path, proposal_amendment_sha256=sha(assignment),
    stage_authorized=False, launch_authorized=False, release_file_created=False,
    comparison_control={'label':parent['label'],'source_sha256':parent['source_sha256'],
      'root':parent['root'],'role':'genuine full1800 kind=control plus fresh20M, unchanged',
      'completed_bundle':'research/v2/operator/20260926-final-sprint-successors/trn22-e1e2-control-completed/results-bundle.json',
      'actual_terminal_and_timing_pending':True},
    extra_off45_cancelled_prospectively=True,
    timing_precommit_commit='d3da2359',
    native_gate_conditions=[
      'Current exact parent full1800 and attached20M terminal/retrieved; all observed descendants/compiler/device quiescent.',
      'Authenticate same parent runtime/science, finite sequential first45 observed boundaries t9/t44 and trustworthy unstalled/unrecompiled window; missing evidence refuses this comparison.',
      'Exact original832589 lifetime lock, guardian, fresh ps/neuron-ls/fuser/STOP/hash/64GiB RAM and unchanged13.5GiB initial/10.5GiB runtime floors.',
      'ON six private cache directories absent then created empty; actual four-rank source/env identity captured.',
      'No extra OFF45 run or second3GiB reserve; allocated parent artifacts remain protected.',
      'ON45 has no quality eval; valid finite complete timing and ON/parent rate>=1.03 needed before separate full followup.'])
binding['timing'] = dict(binding['timing'],
    primary_statistic='35*131072/(observed boundary time at step44 minus boundary time at step9), identical observer/statistic for live exact parent and ON',
    screen_pass='45 finite completed steps, valid native execution/accounting, rateON/rateCONTROL>=1.03',
    cold_cache='ON six new private empty roots. Parent retained research cache provenance separately recorded; no parent cold/hit claim.')
files.update({'host_driver.py':driver,'package/provenance.json':newprov,
              'PROPOSAL_AMENDMENT.md':assignment,
              'PARENT_CONTROL_PRECOMMIT.md':(HERE/'PARENT_CONTROL_PRECOMMIT.md').read_bytes()})
files['NATIVE_BINDING.json']=(json.dumps(binding,indent=2)+'\n').encode()
for n in ('package/train.py','package/launch-command.txt','dispatch_driver.py','idle_guard.py','lifetime_guard.py','EXPERIMENT_PROPOSAL.json'):
    assert files[n] == original[n], n
assert len(binding['fresh_cache_directories']) == 6
assert all(p.startswith(root+'/cold-training-caches/') for p in binding['fresh_cache_directories'])
assert binding['lock_identity'] == [66305,832589]
assert binding['resource_reserve']['initial_floor_bytes'] == 14495514624
out = HERE/'trn22-on-parentcontrol-prepared.json'
assert not out.exists()
packet={'binding':binding,'files':{n:{'sha256':sha(v),'b64':base64.b64encode(v).decode()} for n,v in files.items()},
        'mechanical_derivation':{'original_packet_sha256':sha(oldraw),
           'original_driver_sha256':sha(original['host_driver.py']),
           'literal_driver_changes':['root','label','corresponding provenance hash'],
           'scientific_source_argv_guardian_and_dispatch_unchanged':True,
           'original_proposal_preserved_with_supervisor_amendment':True},
        'local_only':True,'host_calls':0,'native_started':False}
with out.open('x') as f: json.dump(packet,f,indent=2); f.write('\n')
print(json.dumps({'status':'LOCAL_LITERAL_REBIND_PASS_NOT_STAGED','packet':str(out),
                  'packet_sha256':sha(out.read_bytes()),'driver_sha256':sha(driver),
                  'source_sha256':binding['source_sha256'],'launch_command_sha256':sha(files['package/launch-command.txt']),
                  'initial_floor_bytes':14495514624,'actual_parent_result_pending':True}))
