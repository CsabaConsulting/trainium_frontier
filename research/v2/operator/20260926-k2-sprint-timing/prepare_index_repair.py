"""Local literal binding of the independently reviewed two-line index repair."""
import ast,base64,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
def sha(v):return hashlib.sha256(v).hexdigest()
def encode(v):return {'sha256':sha(v),'b64':base64.b64encode(v).decode()}
raw=(HERE/'trn22-on-parentcontrol-prepared.json').read_bytes()
assert sha(raw)=='d3ab702d0839043ad6267d01a27a0aeaa02f81236a451e5155720f0255c5614e'
old=json.loads(raw)
old_label=old['binding']['label'];new_label='fs-20260926-k2-indexrepair-on-parentcontrol-trn22-r1'
old_source=old['binding']['source_sha256'];new_source='a3181d0dd183ca04efdbf9dcf5dcd7e3b556a892a3c9a30a2761e73d37c2aa71'
source=(REPO/'research/v2/lanes/k2-native-index-repair/train.py').read_bytes();assert sha(source)==new_source
proposal_path='research/v2/proposals/20260926-k2-native-index-repair-timing/EXPERIMENT_PROPOSAL.json'
proposal=(REPO/proposal_path).read_bytes();assert sha(proposal)=='0b2f56940734765e9a85e46b250fd3d1ed8a989fa701c5ef7bfa0d908bcb7e5c'
review=(REPO/'research/v2/verification/20260926-k2-native-index-repair/REVIEW.md').read_bytes();assert sha(review)=='b689fb6ed0dd86b73bb945da2fb6bb31a3e8a524b15dc3c984624a5dea08c6dc'
assignment_path='research/v2/assignments/20260926-k2-native-index-repair.md'
assignment=(REPO/assignment_path).read_bytes();assert new_source.encode() in assignment and new_label.encode() in assignment
b=json.loads(json.dumps(old['binding']).replace(old_label,new_label).replace(old_source,new_source))
files={n:base64.b64decode(v['b64']) for n,v in old['files'].items()}
for n,v in files.items():assert sha(v)==old['files'][n]['sha256']
old_provenance=files['package/provenance.json']
prov=json.loads(old_provenance.decode().replace(old_label,new_label).replace(old_source,new_source))
prov['scientific_causal_change']='K2 ON; independently reviewed two-line int32/int64 guard repair, no added cast'
files['package/provenance.json']=(json.dumps(prov,indent=2)+'\n').encode()
prior_driver=files['host_driver.py']
driver=prior_driver.decode().replace(old_label,new_label).replace(old_source,new_source).replace(sha(old_provenance),sha(files['package/provenance.json']))
assert driver.replace(new_label,old_label).replace(new_source,old_source).replace(sha(files['package/provenance.json']),sha(old_provenance))==prior_driver.decode()
ast.parse(driver);files['host_driver.py']=driver.encode();files['package/train.py']=source
files['PRIOR_EXPERIMENT_PROPOSAL.json']=files['EXPERIMENT_PROPOSAL.json']
files['EXPERIMENT_PROPOSAL.json']=proposal;files['PROPOSAL_AMENDMENT.md']=assignment;files['INDEX_REPAIR_REVIEW.md']=review
b.update(driver_sha256=sha(files['host_driver.py']),independent_review_sha256=sha(review),operator_assignment_sha256=sha(assignment),proposal_amendment_path=assignment_path,proposal_amendment_sha256=sha(assignment),completed_proposal_path=proposal_path,completed_proposal_sha256=sha(proposal))
b['comparison_control']['actual_terminal_and_timing_pending']=False
b['comparison_control']['completed_bundle_sha256']='f17ea4ebbbef720cee5d02f810fcb103fc887a5d56fa7c906910f28346dae6c7'
b['native_short_timing']='One diagnosed ON45 retry; no extra OFF/full/eval'
assert json.loads(proposal)['candidate']['launch_command']==b['launch_command']
assert files['package/launch-command.txt']==base64.b64decode(old['files']['package/launch-command.txt']['b64'])
for n in ('dispatch_driver.py','idle_guard.py','lifetime_guard.py'):assert files[n]==base64.b64decode(old['files'][n]['b64'])
files['NATIVE_BINDING.json']=(json.dumps(b,indent=2)+'\n').encode()
packet={'binding':b,'files':{n:encode(v) for n,v in files.items()},'mechanical_derivation':{'prior_packet_sha256':sha(raw),'prior_driver_sha256':sha(prior_driver),'literal_driver_changes':['label/root/private-cache namespace','reviewed source SHA','corresponding provenance SHA'],'driver_logic_scientific_argv_guardian_dispatch_unchanged':True,'source_exact_independently_reviewed_guard_repair':True},'local_only':True,'host_calls':0,'native_started':False}
out=HERE/'trn22-indexrepair-prepared.json'
with out.open('x') as f:json.dump(packet,f,indent=2);f.write('\n')
print(json.dumps({'packet_sha256':sha(out.read_bytes()),'driver_sha256':b['driver_sha256'],'proposal_sha256':sha(proposal),'source_sha256':new_source,'label':new_label,'no_host_action':True}))
