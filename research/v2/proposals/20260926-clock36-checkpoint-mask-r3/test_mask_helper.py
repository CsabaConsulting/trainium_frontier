"""Implementer-only tiny tests. No train.py/checkpoint/SDK/data/host imports."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import torch

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('mask_helper',HERE/'compare_checkpoints.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
CASES=[]


def oracle(length,window):
    # Scalar reference deliberately distinct from the helper's vector formula.
    return torch.tensor([[0.0 if col<=row and col>row-window else -float('inf')
                          for col in range(length)] for row in range(length)],dtype=torch.float32).reshape(1,1,length,length)


def positive(name,fn):
    result=fn();CASES.append({'case':name,'outcome':'PASS'});return result


def negative(name,fn,prefix):
    try:fn()
    except RuntimeError as error:
        h.require(str(error).startswith(prefix),'WRONG_FAILURE:'+name+':'+str(error))
        CASES.append({'case':name,'outcome':'REFUSED_AS_EXPECTED','error':str(error)})
    else:raise RuntimeError('FALSE_ACCEPT:'+name)


class Tiny(torch.nn.Module):
    def __init__(self,mode='good',length=8,window=4):
        super().__init__()
        self.config=SimpleNamespace(sequence_len=length)
        self.weight=torch.nn.Parameter(torch.tensor(.25))
        self.register_buffer('counter',torch.tensor(0.),persistent=False)
        self.register_buffer('cos',torch.ones(1,2),persistent=False)
        self.register_buffer('sin',torch.zeros(1,2),persistent=False)
        self.register_buffer('mask_full',oracle(length,length),persistent=False)
        self.register_buffer('mask_short',oracle(length,window),persistent=False)
        self.mode=mode;self.calls=0;self.scratch=torch.zeros(1,8,4);self.eval()
    def forward(self,x):
        self.calls+=1
        if self.mode=='parameter':self.weight.add_(.125)
        if self.mode=='parameter_bit':self.weight.zero_().copysign_(torch.tensor(-1.))
        if self.mode=='buffer':self.counter.add_(1.)
        if self.mode=='mask':self.mask_full[0,0,0,0]=-float('inf')
        if self.mode=='mask_bit':self.mask_full[0,0,0,0]=-0.0
        if self.mode=='input':x.add_(1)
        if self.mode=='training':self.train()
        out=torch.nn.functional.one_hot(x%4,4).float()+self.weight
        if self.mode in ('future','alias_future'):out=out+x[:,-1:].float().unsqueeze(-1)
        if self.mode=='nondeterministic':out=out+self.calls
        if self.mode=='nonfinite':out=out*float('nan')
        if self.mode=='shape':out=out[:,:-1]
        if self.mode=='signed_zero':out=torch.zeros_like(out).copysign(torch.tensor(-1. if self.calls%2==0 else 1.))
        if self.mode=='alias_future':self.scratch.copy_(out);return self.scratch
        return out


def main():
    module=SimpleNamespace(EVAL_MIN_SEQ_LEN=8,SHORT_WINDOW_FRAC=.5,_EVAL_TOGGLE_DEFAULTS={'SHORT_WINDOW_FRAC':.5})
    payload={'model_config':{'sequence_len':4},'toggles':{'SHORT_WINDOW_FRAC':.5}}
    model=Tiny()
    mask_spec=positive('derive_saved_restored_configuration',lambda:h.restored_mask_spec(model,module,payload))
    h.require(mask_spec['sequence_len']==8 and mask_spec['short_window']==4,'RESTORED_CONFIG')
    positive('saved_default_fraction_restoration',lambda:h.restored_mask_spec(model,module,{'model_config':{'sequence_len':8},'toggles':{}}))
    for length in (1,4,8):
        for window in (1,max(1,length//2),length):
            m=oracle(length,window)
            result=positive('exact_mask_'+str(length)+'_'+str(window),lambda:h.mask_record(m,length,window,'buffer:mask_short'))
            h.require(result['negative_infinity_count']==int(torch.isneginf(m).sum()),'COUNT')
            h.require(result['sha256']==h.raw_record(m)['sha256'],'RAW_SHA')
    # The old helper failure is preserved, not reclassified.
    old=HERE.parent/'20260926-clock36-recursive-exact-cpu-r2-trn22/compare_checkpoints.py'
    h.require(hashlib.sha256(old.read_bytes()).hexdigest()=='a3d9d767b33967311bc8d8fceb852fb771775aeeac0a82433ed2dd9560fa1b6d','OLD_HELPER_PIN')
    oldspec=importlib.util.spec_from_file_location('old_helper',old)
    oldh=importlib.util.module_from_spec(oldspec);oldspec.loader.exec_module(oldh)
    negative('original_legitimate_mask_failure',lambda:oldh.tensor_record(model.mask_full),'NONFINITE_TENSOR')
    negative('no_unvalidated_mask_exemption',lambda:h.model_state(model),'NONFINITE_TENSOR:buffer:mask_full')
    x=torch.tensor([[1,2,3,0,1,2,3,0]])
    qualified=positive('ordinary_tiny_causal_repeat_state',lambda:h.verify_trained_model(model,x.clone(),4,mask_spec))
    h.require(qualified['state_tensor_count']==6 and qualified['causality']['prefix_exact'],'ALL_STATE')
    h.require(qualified['named_parameters_and_all_buffers']['buffer:mask_full']['zero_count']==36,'FULL_MASK_COUNT')
    def corrupt_mask(kind):
        m=oracle(8,4)
        if kind=='nan':m[0,0,0,0]=float('nan')
        if kind=='positive_infinity':m[0,0,0,1]=float('inf')
        if kind=='hidden_zero':m[0,0,0,1]=0.0
        if kind=='visible_neginf':m[0,0,0,0]=-float('inf')
        if kind=='negative_zero':m[0,0,0,0]=-0.0
        return h.mask_record(m,8,4,'buffer:mask_short')
    for name,prefix in (('nan','MASK_NAN'),('positive_infinity','MASK_POSINF'),('hidden_zero','MASK_VALUES'),('visible_neginf','MASK_VALUES'),('negative_zero','MASK_VALUES')):
        negative(name,lambda n=name:corrupt_mask(n),prefix+':buffer:mask_short')
    negative('wrong_shape',lambda:h.mask_record(oracle(4,4),8,4,'buffer:mask_short'),'MASK_SHAPE:buffer:mask_short')
    negative('wrong_dtype',lambda:h.mask_record(oracle(8,4).double(),8,4,'buffer:mask_short'),'MASK_DTYPE_DEVICE_GRAD:buffer:mask_short')
    negative('wrong_window',lambda:h.mask_record(oracle(8,8),8,4,'buffer:mask_short'),'MASK_VALUES:buffer:mask_short')
    negative('nonpositive_window',lambda:h.mask_record(oracle(8,4),8,0,'buffer:mask_short'),'MASK_CONFIG:buffer:mask_short')
    negative('wrong_restored_length',lambda:h.restored_mask_spec(Tiny(length=4),module,payload),'RESTORED_SEQUENCE_LENGTH')
    wrong=SimpleNamespace(EVAL_MIN_SEQ_LEN=8,SHORT_WINDOW_FRAC=.25,_EVAL_TOGGLE_DEFAULTS={'SHORT_WINDOW_FRAC':.5})
    negative('wrong_restored_fraction',lambda:h.restored_mask_spec(model,wrong,payload),'RESTORED_SHORT_WINDOW_FRAC')
    bad_payload={'model_config':{'sequence_len':4},'toggles':{'SHORT_WINDOW_FRAC':float('nan')}}
    negative('nan_saved_fraction',lambda:h.restored_mask_spec(model,module,bad_payload),'SAVED_SHORT_WINDOW_FRAC')
    for bad,label in ((float('nan'),'nan'),(float('inf'),'posinf'),(-float('inf'),'neginf')):
        p=torch.nn.Module();p.register_parameter('mask_full',torch.nn.Parameter(torch.tensor(bad)))
        negative('parameter_named_mask_'+label,lambda p=p:h.model_state(p,mask_spec),'NONFINITE_TENSOR:parameter:mask_full')
    bad_buffer=Tiny();bad_buffer.cos[0,0]=-float('inf')
    negative('other_buffer_nonfinite',lambda:h.model_state(bad_buffer,mask_spec),'NONFINITE_TENSOR:buffer:cos')
    nested=Tiny();nested.child=torch.nn.Module();nested.child.register_buffer('mask_full',oracle(2,2))
    negative('nested_mask_not_exempt',lambda:h.model_state(nested,mask_spec),'NONFINITE_TENSOR:buffer:child.mask_full')
    missing=Tiny();del missing.mask_short
    negative('missing_root_mask',lambda:h.model_state(missing,mask_spec),'MISSING_ROOT_MASK_BUFFERS')
    expected={'future':'SYNTHETIC_PREFIX_CAUSALITY','alias_future':'SYNTHETIC_PREFIX_CAUSALITY',
              'parameter':'MODEL_STATE_CHANGED:parameter:weight','parameter_bit':'MODEL_STATE_CHANGED:parameter:weight',
              'buffer':'MODEL_STATE_CHANGED:buffer:counter','mask':'MASK_VALUES:buffer:mask_full','mask_bit':'MASK_VALUES:buffer:mask_full',
              'input':'INPUT_CHANGED','training':'MODEL_MODE_CHANGED','nondeterministic':'REPEATED_LOGITS_BYTES',
              'nonfinite':'NONFINITE_TENSOR:output','shape':'OUTPUT_SHAPE','signed_zero':'REPEATED_LOGITS_BYTES'}
    for mode,prefix in expected.items():
        m=Tiny(mode)
        if mode=='parameter_bit':m.weight.data.zero_()
        negative(mode,lambda m=m:h.verify_trained_model(m,x.clone(),4,mask_spec),prefix)
    print(json.dumps({'status':'IMPLEMENTER_TINY_MASK_HELPER_TESTS_ONLY','torch':torch.__version__,
        'threads':torch.get_num_threads(),'maximum_test_tensor_elements':64,'cases':CASES,'count':len(CASES),
        'positive_qualification':qualified,'helper_sha256':hashlib.sha256((HERE/'compare_checkpoints.py').read_bytes()).hexdigest(),
        'source_imported':False,'trained_checkpoint_loaded':False,'native_executed':False,
        'original_r2_qualification':'FAILED_PRESERVED','cold_authorized':False},indent=2,allow_nan=False))


if __name__=='__main__':main()
