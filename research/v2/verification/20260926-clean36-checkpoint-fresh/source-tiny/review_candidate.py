import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import resource
import torch

torch.set_num_threads(1)
BASE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('candidate',BASE/'candidate.py')
h=importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
checks=[]
def yes(name, condition):
    if not condition: raise RuntimeError(name)
    checks.append({'name':name,'result':'PASS'})
def no(name,fn,substring):
    try: fn()
    except h.CheckFailure as e:
        if substring not in str(e): raise
        checks.append({'name':name,'result':'REJECTED','message':str(e)})
    else: raise RuntimeError('FALSE_ACCEPT:'+name)

yes('frozen_candidate_sha',hashlib.sha256((BASE/'candidate.py').read_bytes()).hexdigest()=='65e7eb1a21b0e880ee2362461ff3664c4dbd52933816e5ec0684645c92cd75e8')
for length in (4,7,65):
    for window in (1,3,length):
        mask=torch.full((1,1,length,length),float('-inf'))
        for r in range(length):
            for c in range(max(0,r-window+1),r+1):mask[0,0,r,c]=0
        rec=h.mask_record(mask,length,window,'independent')
        yes('scalar_mask_'+str((length,window)),rec['zero_count']==sum(min(r+1,window) for r in range(length)))
        for label,coord,value,reason in [('zero',(0,1),0,'MASK_VALUES'),('nan',(0,0),float('nan'),'MASK_NAN'),('posinf',(0,0),float('inf'),'MASK_POSINF'),('negzero',(0,0),-0.0,'MASK_VALUES')]:
            bad=mask.clone(); bad[0,0,coord[0],coord[1]]=value
            no('mask_'+label+str((length,window)),lambda:h.mask_record(bad,length,window,'bad'),reason)
no('mask_shape',lambda:h.mask_record(torch.zeros(4,4),4,4,'bad'),'MASK_SHAPE')
no('mask_dtype',lambda:h.mask_record(torch.zeros(1,1,4,4,dtype=torch.float64),4,4,'bad'),'MASK_DTYPE')

class Tiny(torch.nn.Module):
    def __init__(self, mode='valid'):
        super().__init__(); self.mode=mode; self.calls=0
        self.weight=torch.nn.Parameter(torch.ones(1),requires_grad=False)
        self.register_buffer('scratch',torch.zeros(1),persistent=False)
        self.child=torch.nn.Identity()
        self.eval()
    def forward(self,x):
        self.calls+=1
        if self.mode=='weight': self.weight.add_(1)
        if self.mode=='buffer': self.scratch.add_(1)
        if self.mode=='input': x[0,0]+=1
        if self.mode=='childmode': self.child.train()
        y=torch.nn.functional.one_hot(x,16).float()
        if self.mode=='future': y=y+x[:,-1,None,None].float()
        if self.mode=='batch': y=y+x.shape[0]
        if self.mode=='shape': y=y[:,:,:-1]
        if self.mode=='nan': y[0,0,0]=float('nan')
        if self.mode=='alias':
            if not hasattr(self,'out'): self.out=y.clone()
            self.out.copy_(y+self.calls); y=self.out
        return y

for batch,length in h.SHAPES:
    x=torch.randint(1,16,(batch,length),generator=torch.Generator().manual_seed(7))
    output,rec=h.verify_shape(Tiny(),x.clone(),16)
    yes('valid_shape_'+str((batch,length)),torch.equal(output,torch.nn.functional.one_hot(x,16).float()) and rec['causality']['prefix_exact'])
x=torch.tensor([[1,2,3,4,5,6,7,8],[2,3,4,5,6,7,8,9],[3,4,5,6,7,8,9,10]])
for mode,reason in [('weight','MODEL_STATE_CHANGED'),('buffer','MODEL_STATE_CHANGED'),('input','INPUT_CHANGED'),('future','SYNTHETIC_PREFIX_CAUSALITY'),('batch','ROWWISE_LOGITS_BYTES'),('shape','OUTPUT_SHAPE'),('nan','NONFINITE_TENSOR'),('childmode','MODEL_MODE_CHANGED'),('alias','REPEATED_LOGITS_BYTES')]:
    no(mode,lambda:h.verify_shape(Tiny(mode),x.clone(),16),reason)
no('degenerate_suffix',lambda:h.verify_shape(Tiny(),torch.ones(3,8,dtype=torch.int64),16),'DEGENERATE_SUFFIX_FIXTURE')
hooked=Tiny(); hooked.register_forward_hook(lambda *args:None)
no('hook',lambda:h.verify_shape(hooked,x.clone(),16),'MODEL_HOOK')
no('input_dtype',lambda:h.verify_shape(Tiny(),x.float(),16),'SYNTHETIC_INPUT_TYPE')
bad=x.clone();bad[0,0]=16
no('input_range',lambda:h.verify_shape(Tiny(),bad,16),'SYNTHETIC_INPUT_RANGE')
no('cross_source_bytes',lambda:h.strict_equal(torch.ones(2),torch.zeros(2),'CROSS'),'CROSS')
no('cross_source_shape',lambda:h.strict_equal(torch.ones(2),torch.ones(1,2),'CROSS'),'CROSS')
no('cross_source_dtype',lambda:h.strict_equal(torch.ones(2),torch.ones(2,dtype=torch.float64),'CROSS'),'CROSS')
m=Tiny();payload={'model_state':m.state_dict()};h.loaded_state(m,payload)
payload=copy.deepcopy(payload);payload['model_state']['weight']+=1
no('loaded_not_saved',lambda:h.loaded_state(m,payload),'LOADED_STATE_BYTES')
payload['model_state']['extra']=torch.zeros(1)
no('loaded_key_mismatch',lambda:h.loaded_state(m,payload),'LOADED_STATE_KEYS')
p={'model_config':{'n_layer':8,'n_embd':1024,'vocab_size':8192,'sequence_len':257},'model_state':{str(i):torch.ones(1) for i in range(65)}}
no('payload_count',lambda:h.full_payload(p),'FULL_PARAMETER_COUNT')
p['model_state'].pop('0');no('payload_keys',lambda:h.full_payload(p),'FULL_STATE_KEYS')
p['model_config']['n_layer']=7;no('payload_config',lambda:h.full_payload(p),'FULL_CONFIG')

# Authentication is exercised only on the reviewer's own small source file.
path=BASE/'review_candidate.py';pin=h.file_record(path)
h.authenticate_files({str(path):pin});yes('file_positive',True)
no('file_sha',lambda:h.authenticate_files({str(path):{**pin,'sha256':'0'*64}}),'FILE_IDENTITY')
no('file_bytes',lambda:h.authenticate_files({str(path):{**pin,'bytes':pin['bytes']+1}}),'FILE_IDENTITY')
no('relative_path',lambda:h.file_record(Path('candidate.py')),'ABSOLUTE_CANONICAL_PATH')
no('missing_path',lambda:h.file_record(BASE/'absent'),'MISSING_FILE')
no('unfrozen_manifest',lambda:h.validate_manifest({'status':'PENDING'}),'REAL_INPUTS_PENDING')
manifest={'status':'FROZEN_REAL_INPUTS','root':str(h.ROOT),'arms':{},'files':{}}
for name in h.SOURCES:
    cp=str(h.ROOT/name/'tmp/g2-independent/out/final.pt')
    manifest['arms'][name]={'checkpoint_path':cp,'checkpoint_sha256':'a'*64,'checkpoint_bytes':1}
    manifest['files'][cp]={'sha256':'a'*64,'bytes':1}
    for f in ('request.json','g2.json','terminal.json','checkpoint-identity.json','train.py'):
        manifest['files'][str(h.ROOT/name/f)]={'sha256':'a'*64,'bytes':1}
    manifest['files'][str(h.ROOT/name/'train.py')]['sha256']=h.SOURCES[name]
    manifest['files'][str(h.ROOT/name/'request.json')]['sha256']=h.REQUESTS[name]
for path in (h.PREPARE,h.ROOT/'preflight_dryrun.py',h.ROOT/'standard-compare.json',h.ROOT/'standard-compare-exit.json',h.ROOT/'terminal.json'):
    manifest['files'][str(path)]={'sha256':'a'*64,'bytes':1}
manifest['files'][str(h.PREPARE)]['sha256']=h.PREPARE_SHA
manifest['files'][str(h.ROOT/'preflight_dryrun.py')]['sha256']=h.G2_SHA
h.validate_manifest(manifest);yes('manifest_structure_positive_only',True)
def manifest_bad(name, mutate, reason):
    m=copy.deepcopy(manifest); mutate(m);no(name,lambda:h.validate_manifest(m),reason)
manifest_bad('wrong_source',lambda m:m['files'][str(h.ROOT/'candidate/train.py')].update(sha256='0'*64),'SOURCE_PIN')
manifest_bad('wrong_checkpoint',lambda m:m['arms']['candidate'].update(checkpoint_sha256='0'*64),'CHECKPOINT_MANIFEST_PIN')
manifest_bad('wrong_arm_path',lambda m:m['arms']['candidate'].update(checkpoint_path=str(h.ROOT/'reference/tmp/g2-x/out/final.pt')),'CHECKPOINT_OUTSIDE_ARM')
manifest_bad('wrong_private_path',lambda m:m['arms']['candidate'].update(checkpoint_path=str(h.ROOT/'candidate/tmp/other/out/final.pt')),'CHECKPOINT_PRIVATE_PATH')
manifest_bad('extra_manifest_file',lambda m:m['files'].update({'/other':{'sha256':'a'*64,'bytes':1}}),'EXACT_INPUT_FILE_SET')
manifest_bad('wrong_prepare',lambda m:m['files'][str(h.PREPARE)].update(sha256='0'*64),'PREPARE_PIN')

request={'source_sha256':h.SOURCES['reference'],'argv':['fixture']}
g={'verdict':'PASS','timed_out':False,'torchrun_rc':0,'train_py_sha256':request['source_sha256'],'argv':['fixture'],'steps':3,'micro_batch':[2,256],'nproc':4,'prepare':{'sha256':h.PREPARE_SHA,'stub':False},'plan_sha256':'a'*64,'ranks':[]}
for r in range(4):g['ranks'].append({'rank':r,'status':'PASS','n_params':65,'cross_rank_mismatches':[],'plan_sha256':'a'*64,'losses':[1.,2.,3.]})
g['ranks'][0].update(checkpoint_check={'ok':True},final_digests={str(i):'b'*16 for i in range(65)})
h.qualify_g2(g,request);yes('g2_positive_structure_only',True)
for name,mutate,reason in [('g2_fail',lambda d:d.update(verdict='FAIL'),'G2_PASS_REQUIRED'),('g2_timeout',lambda d:d.update(timed_out=True),'G2_PASS_REQUIRED'),('g2_source',lambda d:d.update(train_py_sha256='0'*64),'G2_SOURCE_ARGV'),('g2_bad_loss',lambda d:d['ranks'][2].update(losses=[1.,float('nan'),3.]),'G2_FINITE_THREE_LOSSES'),('g2_rank',lambda d:d['ranks'][3].update(rank=2),'G2_FOUR_RANKS'),('g2_digest',lambda d:d['ranks'][0]['final_digests'].pop('0'),'G2_65_DIGESTS'),('g2_checkpoint',lambda d:d['ranks'][0].update(checkpoint_check={'ok':False}),'G2_CHECKPOINT')]:
    d=copy.deepcopy(g);mutate(d);no(name,lambda:h.qualify_g2(d,request),reason)
print(json.dumps({'scope':'independent tiny helper tests only; real checkpoints/native/BPB pending','python':platform.python_version(),'torch':torch.__version__,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'count':len(checks),'checks':checks},indent=2))
