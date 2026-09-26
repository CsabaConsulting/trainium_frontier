"""Rebind the proven native201 packet, disabled; no SSH or host side effects."""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
def sha(raw): return hashlib.sha256(raw).hexdigest()
def enc(x): return (json.dumps(x,indent=2)+'\n').encode()

source = (HERE/'train.py').read_bytes()
assert sha(source) == 'ba5add518bfb93206313476b2cd929cd678b0ed63b0045d65ad547970a9b7ee7'
receipts = {}
for dtype in ('int32','int64'):
    raw = (HERE/('CPU_'+dtype+'.json')).read_bytes()
    receipt = json.loads(raw)
    assert receipt['status'] == 'PASS_REDUCED_CPU' and receipt['source_sha256'] == sha(source)
    assert receipt['native_executed'] is False
    for name, digest in receipt['input_sha256'].items():
        assert sha((REPO/name).read_bytes()) == digest, name
    receipts[dtype] = {'sha256': sha(raw), 'result': receipt}
cpu = enc({'status':'PASS_REDUCED_CPU_ONLY','source_sha256':sha(source),'receipts':receipts,
           'native_measured':False,'new_independent_review':False})
parent = json.loads((HERE.parent/'k2-outside-compile-rescue/native201.disabled.json').read_bytes())
old = parent['binding']
assert old['source_sha256'] == '949f262c092c141f715bdd71532c42e0bcf502d926a1ee20bc885a2255062daa'
files = {k:base64.b64decode(v['b64'],validate=True) for k,v in parent['files'].items()}
for k,raw in files.items(): assert sha(raw) == parent['files'][k]['sha256'], k
label = 'fs-20260926-tables-outside-compile-native201-trn21-r1'
root = str(Path(old['root']).parent/(label+'-packet'))
command = old['launch_command'].replace('--no-ngram-ve','--ngram-ve')
assert command != old['launch_command'] and command.count('--ngram-ve') == 1
assert '--total-batch-size 262144 --matrix-lr 0.021' in command
command_raw = (command+'\n').encode()
proposal = {
    'proposal_id':label,'arm_id':'last-layer-fp32-tables-outside-compile','kind':'kernel_timing',
    'authority':'Poojith: tables timing next free slot ONLY if current K2 rescue native net>=3%',
    'status':'PREPARED_CONDITIONAL_NOT_RELEASED','host':'trn21','label':label,
    'candidate':{'source_sha256':sha(source),'launch_command':command,'steps':201},
    'single_causal_change':'Last-layer b7/t7 FP32 tables on949f; gather outside graph; batch262144/LR.021/.75/momentum300/seed42 unchanged',
    'controls':{'source_sha256':old['source_sha256'],'required_native_label':old['label']},
    'prerequisite':'Actual K2 cold net>=+3% and acceptable loss/correctness; no unmeasured promotion',
    'measure':['cold startup','startup charge above900','boundary9..200 steady tok/s','net tokens','loss through200'],
    'limits':{'training_seconds':1800,'full_chip':True,'no_eval':True,'no_automatic_full':True},
    'cpu_receipt_sha256':sha(cpu),'scope':'New reduced CPU evidence; unchanged component backend evidence is not a new whole-model compile',
    'stage_authorized':False,'launch_authorized':False,'submission_authorized':False,
}
proposal_raw = enc(proposal)
provenance = enc({'label':label,'train_sha256':sha(source),'source_sha256':sha(source),
                  'launch_command':command,'base_sha256':old['source_sha256'],
                  'run_purpose':'cold-native201-tables-timing-only','submission_authorized':False})
changes = [(old['root'],root),(old['label'],label),(old['source_sha256'],sha(source)),
           (old['launch_command'],command),
           (sha(files['package/launch-command.txt']),sha(command_raw)),
           (sha(files['package/provenance.json']),sha(provenance))]
def rebind(x):
    if isinstance(x,str):
        for before,after in changes: x=x.replace(before,after)
        return x
    if isinstance(x,dict): return {k:rebind(v) for k,v in x.items()}
    if isinstance(x,list): return [rebind(v) for v in x]
    return x
driver = rebind(files['host_driver.py'].decode())
reverse = driver
for before,after in reversed(changes): reverse=reverse.replace(after,before)
assert reverse == files['host_driver.py'].decode(), 'DRIVER_LOGIC_DRIFT'
ast.parse(driver)
binding = rebind(copy.deepcopy(old))
binding.update(schema='disabled-table-outside-graph-native201-v1',source_sha256=sha(source),
               launch_command=command,driver_sha256=sha(driver.encode()),arm_id=proposal['arm_id'],
               run_purpose='cold-native201-tables-timing-only',base_sha256=old['source_sha256'],
               completed_proposal_path=str((HERE/'NATIVE_PROPOSAL.json').relative_to(REPO)),
               completed_proposal_sha256=sha(proposal_raw),independent_review_sha256=None,
               scientific_patch=proposal['single_causal_change'],
               stage_authorized=False,launch_authorized=False,release_file_created=False)
files.update({'host_driver.py':driver.encode(),'package/train.py':source,
              'package/launch-command.txt':command_raw,'package/provenance.json':provenance,
              'EXPERIMENT_PROPOSAL.json':proposal_raw,'CPU_RECEIPT.json':cpu,
              'NATIVE_BINDING.disabled.json':enc(binding)})
files.pop('NATIVE_TIMING_REFERENCE.json',None)
binding.pop('timing_reference_sha256',None)
files['NATIVE_BINDING.disabled.json']=enc(binding)
packet=enc({'binding':binding,'files':{k:{'sha256':sha(v),'b64':base64.b64encode(v).decode()} for k,v in files.items()},
            'derivation':{'driver_literal_reverse_exact':True,'guardians_unchanged':True,'host_calls':0}})
for name,raw in [('CPU_RECEIPT.json',cpu),('NATIVE_PROPOSAL.json',proposal_raw),('native201.disabled.json',packet)]:
    path=HERE/name
    if path.exists(): assert path.read_bytes()==raw, 'EXISTING_PREPARATION_DRIFT:'+name
    else:
        with path.open('xb') as stream: stream.write(raw)
print(json.dumps({'source_sha256':sha(source),'packet_sha256':sha(packet),'host_calls':0,
                  'launch_authorized':False,'stage_authorized':False,'steps':201}))
