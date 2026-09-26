"""Own tiny CPU/static diagnostics only. No train.py or retained checkpoint import."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('checkpoint_probe',HERE/'compare_checkpoints.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
RESULTS=[]

def positive(name,fn):
    result=fn();RESULTS.append({'case':name,'outcome':'PASS'});return result

def negative(name,fn,expected):
    try:fn()
    except (h.CheckFailure,RuntimeError) as error:
        if not str(error).startswith(expected):raise RuntimeError('WRONG_FAILURE:'+name+':'+str(error)) from error
        RESULTS.append({'case':name,'outcome':'REFUSED_AS_EXPECTED','error':str(error),
                        'details':getattr(error,'details',None)})
    else:raise RuntimeError('FALSE_ACCEPT:'+name)

def mask(length,window):
    return torch.tensor([[0.0 if c<=r and c>r-window else -float('inf') for c in range(length)]
                          for r in range(length)]).reshape(1,1,length,length)

class Tiny(torch.nn.Module):
    def __init__(self,mode='good',masked=True):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.25))
        self.config=SimpleNamespace(sequence_len=9)
        self.register_buffer('cos',torch.ones(1,2),persistent=False)
        self.register_buffer('sin',torch.zeros(1,2),persistent=False)
        if masked:
            self.register_buffer('mask_full',mask(9,9),persistent=False)
            self.register_buffer('mask_short',mask(9,4),persistent=False)
        self.mode=mode;self.calls=0;self.scratch=torch.empty(3,9,4);self.eval()
    def forward(self,x):
        self.calls+=1
        if self.mode=='parameter':self.weight.add_(1)
        if self.mode=='buffer':self.cos.add_(1)
        if self.mode=='input':x.add_(1)
        if self.mode=='mode':self.train()
        if self.mode=='mask':self.mask_full[0,0,0,1]=0.0
        out=torch.nn.functional.one_hot(x%4,4).float()+self.weight
        if self.mode in ('future','alias_future'):out=out+x[:,-1,None,None]
        if self.mode=='batch':out=out+x.shape[0]
        if self.mode=='repeat':out=out+self.calls
        if self.mode=='nonfinite':out[0,0,0]=float('nan')
        if self.mode=='shape':out=out[:,:-1]
        if self.mode=='signed_zero':out=torch.zeros_like(out).copysign(torch.tensor(-1. if self.calls%2==0 else 1.))
        if self.mode=='alias_future':
            view=self.scratch[:x.shape[0],:x.shape[1]];view.copy_(out);return view
        return out

def manifest():
    arms={n:{'checkpoint_path':str(h.ROOT/n/'tmp/g2-fixture/out/final.pt'),
             'checkpoint_sha256':'a'*64,'checkpoint_bytes':1} for n in h.SOURCES}
    files={str(p):{'sha256':'a'*64,'bytes':1} for p in (h.PREPARE,h.ROOT/'preflight_dryrun.py',
           h.ROOT/'standard-compare.json',h.ROOT/'standard-compare-exit.json',h.ROOT/'terminal.json')}
    files[str(h.PREPARE)]['sha256']=h.PREPARE_SHA
    files[str(h.ROOT/'preflight_dryrun.py')]['sha256']=h.G2_SHA
    for name in arms:
        files[arms[name]['checkpoint_path']]={'sha256':'a'*64,'bytes':1}
        for f in ('request.json','g2.json','terminal.json','checkpoint-identity.json','train.py'):
            files[str(h.ROOT/name/f)]={'sha256':'a'*64,'bytes':1}
        files[str(h.ROOT/name/'train.py')]['sha256']=h.SOURCES[name]
        files[str(h.ROOT/name/'request.json')]['sha256']=h.REQUESTS[name]
    return {'status':'FROZEN_REAL_INPUTS','root':str(h.ROOT),'arms':arms,'files':files}

def g2():
    ranks=[{'rank':i,'status':'PASS','n_params':65,'cross_rank_mismatches':[],
            'plan_sha256':'b'*64,'losses':[1.,.9,.8]} for i in range(4)]
    ranks[0].update(checkpoint_check={'ok':True},final_digests={str(i):'a'*16 for i in range(65)})
    return {'verdict':'PASS','timed_out':False,'torchrun_rc':0,'train_py_sha256':h.SOURCES['reference'],
            'argv':['fixture'],'steps':3,'micro_batch':[2,256],'nproc':4,
            'prepare':{'sha256':h.PREPARE_SHA,'stub':False},'ranks':ranks,'plan_sha256':'b'*64}

def main():
    old=HERE.parent/'20260926-clock36-checkpoint-mask-r3/compare_checkpoints.py'
    h.require(hashlib.sha256(old.read_bytes()).hexdigest()=='1b3714e050e473b176a3e1f1fbd59387579867f3d06d14ecf71248a6d8a1714c','OLD_HELPER_PIN')
    trees=[{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
           for p in (old,HERE/'compare_checkpoints.py')]
    for name in ('raw_record','restored_mask_spec','mask_record','model_state','tensor_record','require_state_equal'):
        positive('preserved_AST_'+name,lambda n=name:h.require(trees[0][n]==trees[1][n],'MASK_HELPER_SEMANTICS'))
    module=SimpleNamespace(EVAL_MIN_SEQ_LEN=9,SHORT_WINDOW_FRAC=.5,_EVAL_TOGGLE_DEFAULTS={'SHORT_WINDOW_FRAC':.5})
    payload={'model_config':{'sequence_len':8},'toggles':{'SHORT_WINDOW_FRAC':.5}}
    ms=positive('restored_reference_masks',lambda:h.restored_mask_spec(Tiny(),module,payload))
    for b in (1,2,3):
        for length in (8,9):
            x=torch.randint(1,4,(b,length),generator=torch.Generator().manual_seed(7))
            a,record=positive('masked_batch_context_'+str((b,length)),lambda x=x:h.verify_shape(Tiny(),x,4,ms))
            c,_=positive('unmasked_batch_context_'+str((b,length)),lambda x=x:h.verify_shape(Tiny(masked=False),x,4,None))
            positive('cross_source_'+str((b,length)),lambda a=a,c=c:h.strict_equal(c,a,'CROSS_SOURCE'))
            h.require(record['causality']['prefix_exact'] and len(record['calls'])==b+3,'COMPLETE_SHAPE_COVERAGE')
    expected={'future':'SYNTHETIC_PREFIX_CAUSALITY','alias_future':'SYNTHETIC_PREFIX_CAUSALITY',
              'parameter':'MODEL_STATE_CHANGED','buffer':'MODEL_STATE_CHANGED','input':'INPUT_CHANGED',
              'mode':'MODEL_MODE_CHANGED','mask':'MASK_VALUES','batch':'ROWWISE_LOGITS_BYTES',
              'repeat':'REPEATED_LOGITS_BYTES','nonfinite':'NONFINITE_TENSOR','shape':'OUTPUT_SHAPE',
              'signed_zero':'REPEATED_LOGITS_BYTES'}
    x=torch.tensor([[1,2,3,1,2,3,1,2],[2,3,1,2,3,1,2,3]])
    for mode,error in expected.items():
        negative(mode,lambda mode=mode:h.verify_shape(Tiny(mode),x.clone(),4,ms),error)
    m=Tiny();m.register_forward_hook(lambda *a:None)
    negative('forward_hook',lambda:h.verify_shape(m,x.clone(),4,ms),'MODEL_HOOK')
    def bad_mask(value,where):
        m=mask(9,9);m[0,0,0,where]=value
        return h.mask_record(m,9,9,'mask_full')
    for val,where,tag in ((float('nan'),0,'nan'),(float('inf'),1,'posinf'),(0.,1,'hidden_zero'),(-float('inf'),0,'visible_inf'),(-0.,0,'negative_zero')):
        negative('mask_'+tag,lambda v=val,w=where:bad_mask(v,w),'MASK_')
    negative('mask_shape',lambda:h.mask_record(mask(8,8),9,9,'mask_full'),'MASK_SHAPE')
    negative('mask_dtype',lambda:h.mask_record(mask(9,9).double(),9,9,'mask_full'),'MASK_DTYPE')
    m=Tiny();m.cos.fill_(-float('inf'))
    negative('other_buffer_nonfinite',lambda:h.model_state(m,ms),'NONFINITE_TENSOR')
    m=Tiny();m.weight.data.fill_(float('inf'))
    negative('parameter_nonfinite',lambda:h.model_state(m,ms),'NONFINITE_TENSOR')
    m=Tiny();positive('loaded_state_tiny',lambda:h.loaded_state(m,{'model_state':m.state_dict()}))
    negative('loaded_state_wrong_bytes',lambda:h.loaded_state(m,{'model_state':{'weight':torch.tensor(.5)}}),'LOADED_STATE_BYTES')
    negative('loaded_state_wrong_keys',lambda:h.loaded_state(m,{'model_state':{}}),'LOADED_STATE_KEYS')
    negative('cross_source_wrong_bytes',lambda:h.strict_equal(torch.ones(1,2,4),torch.zeros(1,2,4),'CROSS_SOURCE'),'CROSS_SOURCE')
    negative('cross_source_wrong_shape',lambda:h.strict_equal(torch.ones(1,2,4),torch.zeros(1,3,4),'CROSS_SOURCE'),'CROSS_SOURCE')
    negative('cross_source_wrong_dtype',lambda:h.strict_equal(torch.ones(1,2,4),torch.ones(1,2,4,dtype=torch.float64),'CROSS_SOURCE'),'CROSS_SOURCE')
    small={'model_config':{'n_layer':8,'n_embd':1024,'vocab_size':8192,'sequence_len':1024},'model_state':{str(i):torch.zeros(1) for i in range(65)}}
    negative('reject_tiny_as_full113M',lambda:h.full_payload(small),'FULL_PARAMETER_COUNT')
    small['model_config']['n_layer']=6
    negative('wrong_full_config',lambda:h.full_payload(small),'FULL_CONFIG')
    small['model_config']['n_layer']=8;small['model_state'].pop('0')
    negative('wrong_state_count',lambda:h.full_payload(small),'FULL_STATE_KEYS')
    clean=Tiny(masked=False);cleanmodule=SimpleNamespace(GPT=Tiny,EVAL_MIN_SEQ_LEN=9)
    positive('clean_no_mask_exception',lambda:h.mask_configuration('candidate',clean,cleanmodule,payload))
    negative('candidate_unexpected_masks',lambda:h.mask_configuration('candidate',Tiny(),cleanmodule,payload),'CANDIDATE_BUFFER_LAYOUT')
    negative('candidate_wrong_class',lambda:h.mask_configuration('candidate',clean,SimpleNamespace(GPT=torch.nn.Module),payload),'CANDIDATE_NOT_ORDINARY_GPT')
    positive('pending_manifest_refuses',lambda:negative('pending',lambda:h.validate_manifest({'status':'PENDING'}),'REAL_INPUTS_PENDING'))
    positive('manifest_shape_only_not_real_inputs',lambda:h.validate_manifest(manifest()))
    for tag,mutate,error in (
        ('source',lambda q:q['files'][str(h.ROOT/'candidate/train.py')].update(sha256='b'*64),'SOURCE_PIN'),
        ('checkpoint',lambda q:q['arms']['candidate'].update(checkpoint_sha256='b'*64),'CHECKPOINT_MANIFEST_PIN'),
        ('outside',lambda q:q['arms']['candidate'].update(checkpoint_path='/tmp/final.pt'),'CHECKPOINT_OUTSIDE_ARM'),
        ('traversal',lambda q:q['arms']['candidate'].update(checkpoint_path=str(h.ROOT/'candidate/tmp/g2-a/../out/final.pt')),'CHECKPOINT_PRIVATE_PATH'),
        ('missing',lambda q:q['files'].pop(str(h.PREPARE)),'EXACT_INPUT_FILE_SET'),
        ('missing_sha',lambda q:q['arms']['reference'].update(checkpoint_sha256=None),'CHECKPOINT_PIN_PENDING'),
        ('extra_arm',lambda q:q['arms'].update(off=q['arms']['reference']),'EXACT_TWO_ARMS'),
        ('prepare',lambda q:q['files'][str(h.PREPARE)].update(sha256='b'*64),'PREPARE_PIN')):
        q=manifest();mutate(q);negative('manifest_'+tag,lambda q=q:h.validate_manifest(q),error)
    with tempfile.TemporaryDirectory(prefix='clean-checkpoint-helper-') as directory:
        f=Path(directory)/'fixture';f.write_bytes(b'own tiny immutable fixture')
        rec=h.file_record(f)
        positive('file_identity',lambda:h.authenticate_files({str(f):rec}))
        f.write_bytes(b'changed')
        negative('post_file_mutation',lambda:h.authenticate_files({str(f):rec}),'FILE_IDENTITY')
        link=Path(directory)/'symlink';link.symlink_to(f)
        negative('symlink_refuses',lambda:h.file_record(link),'SYMLINK')
    request={'source_sha256':h.SOURCES['reference'],'argv':['fixture']}
    positive('synthetic_g2_metadata_not_actual_outcome',lambda:h.qualify_g2(g2(),request))
    for tag,mutate,error in (
        ('rank_missing',lambda q:q['ranks'].pop(),'G2_FOUR_RANKS'),
        ('source',lambda q:q.update(train_py_sha256='x'),'G2_SOURCE_ARGV'),
        ('nan_loss',lambda q:q['ranks'][2]['losses'].__setitem__(0,float('nan')),'G2_FINITE'),
        ('timeout',lambda q:q.update(timed_out=True),'G2_PASS_REQUIRED'),
        ('digest',lambda q:q['ranks'][0]['final_digests'].update({'0':'wrong'}),'G2_65_DIGESTS'),
        ('checkpoint',lambda q:q['ranks'][0].update(checkpoint_check={'ok':False}),'G2_CHECKPOINT')):
        q=g2();mutate(q);negative('g2_'+tag,lambda q=q:h.qualify_g2(q,request),error)
    print(json.dumps({'status':'IMPLEMENTER_TINY_HELPER_TESTS_ONLY','torch':torch.__version__,
                     'threads':torch.get_num_threads(),'max_tensor_elements':108,'cases':RESULTS,'count':len(RESULTS),
                     'helper_sha256':hashlib.sha256((HERE/'compare_checkpoints.py').read_bytes()).hexdigest(),
                     'trained_checkpoint_loaded':False,'scientific_source_imported':False,
                     'native_executed':False,'real_inputs':'PENDING','independent_review':False},
                    indent=2,allow_nan=False))

if __name__=='__main__':main()
