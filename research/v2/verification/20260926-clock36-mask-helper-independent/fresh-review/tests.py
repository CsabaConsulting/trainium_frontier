import hashlib, importlib.util, json, sys
from pathlib import Path
from types import SimpleNamespace as NS
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
root=Path(__file__).resolve().parent.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs={p.name:sha(p) for p in root.iterdir() if p.is_file()}
spec=json.loads((root/'NEUTRAL_SPEC.json').read_text())
if inputs['new_helper.py']!=spec['candidate_helper_sha256'] or inputs['old_helper.py']!=spec['baseline_helper_sha256'] or inputs['results-bundle.json']!=spec['raw_original_failure_bundle_sha256']:
    raise RuntimeError('INPUT_PIN_FAILURE')
loader=importlib.util.spec_from_file_location('review_helper',root/'new_helper.py')
h=importlib.util.module_from_spec(loader); loader.loader.exec_module(h)
cases=[]
def check(label,fn,error=None):
    try: result=fn()
    except Exception as exc:
        if error is None or error not in str(exc): raise
        cases.append({'case':label,'rejected':str(exc)}); return
    if error is not None: raise RuntimeError('UNEXPECTED_ACCEPT:'+label)
    cases.append({'case':label,'passed':True})
    return result

def mask(n,w):
    # Scalar distance reference: no broadcasting, triu, or candidate constructor.
    return torch.tensor([[0.0 if 0<=r-c<w else -float('inf') for c in range(n)] for r in range(n)],dtype=torch.float32).reshape(1,1,n,n)
for n in (1,4,7,65):
    for w in (1,2,n,n+2):
        got=check('scalar_reference_%d_%d'%(n,w),lambda:h.mask_record(mask(n,w),n,w,'buffer:mask_full'))
        expected=sum(min(r+1,w) for r in range(n))
        if got['zero_count']!=expected: raise RuntimeError('COUNT')
for value,position,err,label in [(float('nan'),(0,0),'MASK_NAN','nan'),(float('inf'),(0,1),'MASK_POSINF','posinf'),(-0.0,(0,0),'MASK_VALUES','negative_zero'),(0.,(0,1),'MASK_VALUES','future_zero'),(-float('inf'),(0,0),'MASK_VALUES','diagonal_inf'),(1.,(0,0),'MASK_VALUES','nonzero')]:
    t=mask(4,2);t[0,0,position[0],position[1]]=value
    check(label,lambda:h.mask_record(t,4,2,'buffer:mask_short'),err+':buffer:mask_short')
check('dtype',lambda:h.mask_record(mask(4,2).double(),4,2,'m'),'MASK_DTYPE_DEVICE_GRAD')
check('shape',lambda:h.mask_record(mask(4,2).reshape(4,4),4,2,'m'),'MASK_SHAPE')
check('grad',lambda:h.mask_record(mask(4,2).requires_grad_(),4,2,'m'),'MASK_DTYPE_DEVICE_GRAD')
check('wrong_window',lambda:h.mask_record(mask(4,3),4,2,'m'),'MASK_VALUES')

cfg={'sequence_len':6,'full_window':6,'short_window':3}
class Tiny(torch.nn.Module):
    def __init__(self,mode='good'):
        super().__init__();self.mode=mode;self.calls=0
        self.p=torch.nn.Parameter(torch.ones(1))
        self.register_buffer('mask_full',mask(6,6),persistent=False)
        self.register_buffer('mask_short',mask(6,3),persistent=False)
        self.register_buffer('ordinary',torch.zeros(1),persistent=False)
        self.eval()
    def forward(self,x):
        self.calls+=1
        out=x.float().unsqueeze(-1).repeat(1,1,3)
        if self.mode=='param_mutation': self.p.add_(1)
        if self.mode=='buffer_mutation': self.ordinary.add_(1)
        if self.mode=='mask_mutation': self.mask_full[0,0,0,0]=-0.0
        if self.mode=='input_mutation': x.add_(1)
        if self.mode=='repeat': out.add_(self.calls)
        if self.mode=='leak': out.add_(x[0,-1].float())
        if self.mode=='nonfinite': out[0,0,0]=float('nan')
        if self.mode=='shape': out=out[:,:,:2]
        if self.mode=='mode': self.train()
        return out
def verify(mode):return h.verify_trained_model(Tiny(mode),torch.arange(1,7).reshape(1,6),3,cfg)
positive=check('positive_inference',lambda:verify('good'))
for mode,error in [('param_mutation','MODEL_STATE_CHANGED:parameter:p'),('buffer_mutation','MODEL_STATE_CHANGED:buffer:ordinary'),('mask_mutation','MASK_VALUES:buffer:mask_full'),('input_mutation','INPUT_CHANGED'),('repeat','REPEATED_LOGITS_BYTES'),('leak','SYNTHETIC_PREFIX_CAUSALITY'),('nonfinite','NONFINITE_TENSOR:output'),('shape','OUTPUT_SHAPE'),('mode','MODEL_MODE_CHANGED')]:
    check(mode,lambda:verify(mode),error)
for kind in ('parameter','nested','ordinary'):
    m=Tiny()
    if kind=='parameter':
        del m.mask_full;m.register_parameter('mask_full',torch.nn.Parameter(torch.tensor([-float('inf')])))
        error='NONFINITE_TENSOR:parameter:mask_full'
    elif kind=='nested':
        m.child=torch.nn.Module();m.child.register_buffer('mask_full',torch.tensor([-float('inf')]),persistent=False)
        error='NONFINITE_TENSOR:buffer:child.mask_full'
    else:
        m.ordinary.fill_(-float('inf'));error='NONFINITE_TENSOR:buffer:ordinary'
    check('finite_only_'+kind,lambda:h.model_state(m,cfg),error)
check('no_exception_without_spec',lambda:h.model_state(Tiny()),'NONFINITE_TENSOR:buffer:mask_full')
m=Tiny();del m.mask_short
check('missing_mask',lambda:h.model_state(m,cfg),'MISSING_ROOT_MASK_BUFFERS')
check('bad_input_shape',lambda:h.verify_trained_model(Tiny(),torch.ones((1,3)),3,cfg),'SYNTHETIC_INPUT_SHAPE')
check('degenerate_suffix',lambda:h.verify_trained_model(Tiny(),torch.ones((1,6)),3,cfg),'DEGENERATE_SUFFIX_FIXTURE')

def restore(saved=4,minimum=6,actual=6,toggles=None,default=.5,live=.5):
    payload={'model_config':{'sequence_len':saved}}
    if toggles is not None: payload['toggles']=toggles
    return h.restored_mask_spec(NS(config=NS(sequence_len=actual)),NS(EVAL_MIN_SEQ_LEN=minimum,_EVAL_TOGGLE_DEFAULTS={'SHORT_WINDOW_FRAC':default},SHORT_WINDOW_FRAC=live),payload)
for args,length,window in [({},6,3),({'saved':8,'actual':8},8,4),({'toggles':{'SHORT_WINDOW_FRAC':.25},'live':.25},6,1)]:
    got=check('restore_'+str(args),lambda:restore(**args))
    if got['sequence_len']!=length or got['short_window']!=window:raise RuntimeError('RESTORE_REFERENCE')
for args,error in [({'actual':4},'RESTORED_SEQUENCE_LENGTH'),({'live':.25},'RESTORED_SHORT_WINDOW_FRAC'),({'saved':True},'SAVED_SEQUENCE_LENGTH'),({'minimum':0},'EVAL_MIN_SEQUENCE_LENGTH'),({'toggles':[]},'SAVED_TOGGLES'),({'toggles':{'SHORT_WINDOW_FRAC':float('nan')}},'SAVED_SHORT_WINDOW_FRAC')]:
    check('bad_restore_'+str(args),lambda:restore(**args),error)
check('root_pin_before_io',lambda:h.run(Path('/tmp/wrong-root')),'EXACT_READONLY_R2_ROOT')
bundle=json.loads((root/'results-bundle.json').read_text())
for arm,(rel,digest) in h.EXPECTED_CHECKPOINTS.items():
    entry=json.loads(bundle['files'][arm+'/checkpoint-identity.json']['text'])
    if entry['path']!=str(h.OLD_ROOT/rel) or entry['sha256']!=digest:raise RuntimeError('CHECKPOINT_PIN:'+arm)
    req=json.loads(bundle['files'][arm+'/request.json']['text'])
    if req['source_sha256']!=h.EXPECTED_SOURCES[arm]:raise RuntimeError('SOURCE_PIN:'+arm)
cases.append({'case':'all_four_original_bundle_pins','passed':True})
if json.loads(bundle['files']['checkpoint-compare-exit.json']['text'])['exit_code']!=1 or 'RuntimeError: NONFINITE_TENSOR' not in bundle['files']['checkpoint-compare.stderr']['text']:raise RuntimeError('ORIGINAL_FAIL')
for receipt in ('tiny-normal.json','tiny-optimized.json'):
    supplied=json.loads((root/receipt).read_text())
    if supplied['helper_sha256']!=inputs['new_helper.py']:raise RuntimeError('TINY_RECEIPT_PIN')
print(json.dumps({'verdict':'PASS_HELPER_ONLY','cases':cases,'count':len(cases),'input_sha256':inputs,'test_sha256':sha(Path(__file__)),'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'torch':torch.__version__,'optimized':sys.flags.optimize,'positive':positive,'full_checkpoint_executed':False,'native_executed':False,'original_r2':'FAILED_PRESERVED'},indent=2))
