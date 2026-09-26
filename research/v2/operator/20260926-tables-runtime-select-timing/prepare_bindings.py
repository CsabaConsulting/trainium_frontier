"""Local mechanical P2 alternatives; no host calls, source execution or launch."""
import ast
import base64
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
def sha(x):return hashlib.sha256(x).hexdigest()
proposal_path='research/v2/proposals/20260926-runtime-select-fp32-last-layer-tables-timing/EXPERIMENT_PROPOSAL.json'
proposal_bytes=(REPO/proposal_path).read_bytes();proposal=json.loads(proposal_bytes)
assert sha(proposal_bytes)=='52c57afb106f3bf64a0004f2c2368998e26d226723f809b8e22dd7be17372ecc'
source=(REPO/proposal['candidate']['source']).read_bytes()
assert sha(source)==proposal['candidate']['source_sha256']=='414cdd76df85450659375c0928b7f0c882b963088529ca049c19be32814fec34'
review_path='research/v2/lanes/k2-last-layer-tables/review/REVIEW.md'
review=(REPO/review_path).read_bytes()
assert b'PASS_FOR_SHORT_NATIVE_CHECK' in review
assert b'not launch authorization' in review
composition_path='research/v2/verification/20260926-tables-runtime-select/COMPOSITION.md'
composition=(REPO/composition_path).read_bytes()
assert sha(composition)=='07021ea1a6e6b35ef46e04951b1699a3993e244c7af83084b0263a718181f75e'
assert b'Source414cdd76 is frozen' in composition
evidence_paths={
    'SPEC.md':'research/v2/lanes/tables-runtime-select/SPEC.md',
    'SOURCE.diff':'research/v2/lanes/tables-runtime-select/SOURCE.diff',
    'STATUS.md':'research/v2/lanes/tables-runtime-select/STATUS.md',
    'COMPOSITION.md':composition_path,
    'check_addresses.py':'research/v2/verification/20260926-tables-runtime-select/check_addresses.py',
    'P2_original_REVIEW.md':review_path,
    'guard_REVIEW.md':'research/v2/verification/20260926-k2-native-index-repair/REVIEW.md',
    'constants_REVIEW.md':'research/v2/verification/20260926-k2-compiler-local-constants/REVIEW.md',
}
for key,item in proposal['scope_binding']['execution_evidence'].items():
    assert sha((REPO/item['path']).read_bytes())==item['sha256']
    evidence_paths[key+'.json']=item['path']
for key in ('spec','diff','status_report'):
    assert sha((REPO/proposal['scope_binding'][key]).read_bytes())==proposal['scope_binding'][key+'_sha256']
evidence={name:(REPO/p).read_bytes() for name,p in evidence_paths.items()}
rows=[]
for host in ('trn21','trn22'):
    original_path=HERE.parent/'20260926-k2-sprint-timing'/(host+'-on-prepared.json')
    original_raw=original_path.read_bytes();old=json.loads(original_raw);ob=old['binding']
    original={n:base64.b64decode(v['b64']) for n,v in old['files'].items()}
    for n,v in original.items():assert sha(v)==old['files'][n]['sha256']
    assert ob['source_sha256']=='63925519ac6c0aab1a1d598e96fef3f4413fbd083d9b6d177901a2e972060db5'
    label=proposal['candidate']['run_label_template'].replace('<host>',host)
    root=ob['root'].replace(ob['label'],label)
    def rebind(v):
        if isinstance(v,str):return v.replace(ob['root'],root).replace(ob['label'],label).replace(ob['source_sha256'],sha(source))
        if isinstance(v,list):return [rebind(x) for x in v]
        if isinstance(v,dict):return {k:rebind(x) for k,x in v.items()}
        return v
    binding=rebind(ob)
    command=proposal['candidate']['launch_command']
    assert command==ob['launch_command'].replace('--no-ngram-ve','--ngram-ve')
    cmdbytes=(command+'\n').encode()
    prov=rebind(json.loads(original['package/provenance.json']))
    prov.update(launch_command=command,base_sha256=proposal['control']['source_sha256'],
                source_commit=proposal['candidate']['source_commit'],
                proposal_sha256=sha(proposal_bytes),composition_sha256=sha(composition),
                scientific_causal_change='Fixed last-layer FP32 bigram/trigram masters, gradients and RMS on qualified K2',
                run_purpose='P2 45-step native table timing/correctness, not quality')
    provbytes=(json.dumps(prov,indent=2)+'\n').encode()
    driver=rebind(original['host_driver.py'].decode())
    driver=driver.replace(sha(original['package/launch-command.txt']),sha(cmdbytes)).replace(sha(original['package/provenance.json']),sha(provbytes)).encode()
    ast.parse(driver)
    for before,after in ((original['package/launch-command.txt'],cmdbytes),(original['package/provenance.json'],provbytes)):
        assert sha(before).encode() not in driver and sha(after).encode() in driver
    files={n:original[n] for n in ('dispatch_driver.py','idle_guard.py','lifetime_guard.py')}
    files.update({'host_driver.py':driver,'package/train.py':source,'package/launch-command.txt':cmdbytes,
                  'package/provenance.json':provbytes,'EXPERIMENT_PROPOSAL.json':proposal_bytes})
    files.update({'EVIDENCE/'+name:contents for name,contents in evidence.items()})
    binding.update(schema='disabled-p2-fp32-runtime-select45-v1',source_sha256=sha(source),driver_sha256=sha(driver),
        stage_authorized=False,launch_authorized=False,release_file_created=False,submission_authorized=False,
        proposal_path=proposal_path,proposal_sha256=sha(proposal_bytes),independent_review_path=review_path,
        independent_review_sha256=sha(review),source_adoption=False,launch_command=command,
        source_commit=proposal['candidate']['source_commit'],
        base_sha256=proposal['control']['source_sha256'],
        operator_assignment_sha256=None,operator_assignment_pending=True,
        independent_composition_path=composition_path,independent_composition_sha256=sha(composition),
        original_review_scope='Original P2 review plus existing guard/constants reviews; current runtime-address composition by Root, not another independent review.',
        current_evidence={name:{'path':evidence_paths[name],'sha256':sha(raw)} for name,raw in evidence.items()},
        scientific_patch='P2_FP32_last_layer_tables_runtime_select_on_K2',actual_k2_parent_qualification_pending=True,
        actual_k2_dependency=proposal['k2_live_dependency'],
        run_purpose='45-step-native-tables-timing-correctness-no-quality-eval',no_eval=True)
    binding['resource_reserve']={'runtime_floor_bytes':11274289152,
        'existing_prospective_footprint_bytes':3221225472,'additional_table_checkpoint_bytes':536870912,
        'projected_new_footprint_bytes':3758096384,'initial_floor_bytes':15032385536,
        'memory_available_minimum_bytes':68719476736,
        'device_memory_note':'Two FP32 tables add1.5GiB weights+dense gradients+RMS per rank before temporaries; native fit unproved.',
        'fresh_host_resources_pending':True,'no_protected_or_unscored_deletion_assumed':True}
    binding['timing']=proposal['timing']
    binding['native_gate_conditions']=[
        'Disabled: actual parent K2 valid native ON/parent rate>=1.03 must be retrieved/authenticated first.',
        'Current full job and attached20M naturally terminal/retrieved and all descendants/compiler/device quiescent.',
        'Exact source/review/proposal, original host lock plus fresh ps/neuron-ls/fuser/STOP/resource checks; no shared or current artifacts changed.',
        'Fresh six private cold caches, same seed42/world4/LNC2/batch131072/context1024/pack32.',
        '45 finite steps and finite nonzero table gradients/updates, correct source/checkpoint and normal accounting; no minimum speed gain for tables themselves.',
        'No automatic full/eval/adoption or second review; actual K2 full-quality followup keeps priority.']
    files['NATIVE_BINDING.json']=(json.dumps(binding,indent=2)+'\n').encode()
    assert len(binding['fresh_cache_directories'])==6
    assert all(p.startswith(root+'/cold-training-caches/') for p in binding['fresh_cache_directories'])
    assert sha(files['lifetime_guard.py'])=='5df6897deeaea3435a963f79b912ce49f320d3a65624f60a89762249c67ecb6e'
    for unchanged in ('dispatch_driver.py','idle_guard.py','lifetime_guard.py'):
        assert files[unchanged]==original[unchanged]
    assert binding['lock_identity']==ob['lock_identity'] and binding['lock_path']==ob['lock_path']
    out=HERE/(host+'-prepared.disabled.json');assert not out.exists()
    packet={'binding':binding,'files':{n:{'sha256':sha(v),'b64':base64.b64encode(v).decode()} for n,v in files.items()},
            'derivation':{'parent_packet_sha256':sha(original_raw),'parent_driver_sha256':sha(original['host_driver.py']),
               'changes':'Literal root/label/source/command/provenance hashes and proposal/resource/evidence metadata only. No semantic launcher or guardian edit.',
               'guardian_dispatch_idle_guard_unchanged':True,'scientific_source_unchanged':True},
            'local_only':True,'host_calls':0,'native_started':False}
    with out.open('x') as f:json.dump(packet,f,indent=2);f.write('\n')
    rows.append({'host':host,'label':label,'packet_sha256':sha(out.read_bytes()),'driver_sha256':sha(driver),
                 'initial_floor_bytes':15032385536,'state':'DISABLED_PENDING_ACTUAL_K2_QUALIFICATION_AND_FRESH_RESOURCE_CHECKS'})
with (HERE/'PREPARED.json').open('x') as f:json.dump({'source_sha256':sha(source),'review_sha256':sha(review),'composition_sha256':sha(composition),'proposal_sha256':sha(proposal_bytes),'packets':rows,'host_calls':0,'stage_authorized':False,'launch_authorized':False},f,indent=2);f.write('\n')
print(json.dumps({'status':'TWO_LOCAL_DISABLED_HOST_ALTERNATIVES','packets':rows}))
