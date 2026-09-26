"""Own tiny one-thread helper tests; no train.py/model/native/data imports."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('checkpoint_helper_under_test',HERE/'compare_checkpoints.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


class Tiny(torch.nn.Module):
    def __init__(self,mode='good'):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(.25))
        self.register_buffer('counter',torch.tensor(0.),persistent=False)
        self.mode=mode;self.calls=0
        self.scratch=torch.zeros(1,8,4)
        self.eval()
    def forward(self,x):
        self.calls+=1
        if self.mode=='parameter': self.weight.add_(.125)
        if self.mode=='buffer': self.counter.add_(1.)
        if self.mode=='input': x.add_(1)
        if self.mode=='training': self.train()
        out=torch.nn.functional.one_hot(x%4,4).float()+self.weight
        if self.mode in ('future','alias_future'): out=out+x[:,-1:].float().unsqueeze(-1)
        if self.mode=='nondeterministic':out=out+self.calls
        if self.mode=='nonfinite':out=out*float('nan')
        if self.mode=='shape':out=out[:,:-1]
        if self.mode=='signed_zero':out=torch.zeros_like(out).copysign(torch.tensor(-1. if self.calls%2==0 else 1.))
        if self.mode=='alias_future':self.scratch.copy_(out);return self.scratch
        return out


def main():
    x=torch.tensor([[1,2,3,0,1,2,3,0]])
    result=helper.verify_trained_model(Tiny(),x.clone(),4)
    helper.require(result['causality']['passed'] and result['causality']['prefix_exact'],'positive')
    helper.require(result['state_tensor_count']==2,'nonpersistent buffer included')
    cases=[]
    expected={'future':'SYNTHETIC_PREFIX_CAUSALITY','alias_future':'SYNTHETIC_PREFIX_CAUSALITY',
              'parameter':'MODEL_STATE_CHANGED','buffer':'MODEL_STATE_CHANGED','input':'INPUT_CHANGED',
              'training':'MODEL_MODE_CHANGED','nondeterministic':'REPEATED_LOGITS_BYTES',
              'nonfinite':'NONFINITE_TENSOR','shape':'OUTPUT_SHAPE','signed_zero':'REPEATED_LOGITS_BYTES'}
    for mode,error in expected.items():
        try: helper.verify_trained_model(Tiny(mode),x.clone(),4)
        except RuntimeError as caught:
            helper.require(str(caught)==error,mode+':wrong failure '+str(caught))
            cases.append({'mode':mode,'refused':str(caught)})
        else: raise RuntimeError('FALSE_ACCEPT:'+mode)
    for bad in (float('nan'),float('inf'),-float('inf')):
        try: helper.tensor_record(torch.tensor([bad]))
        except RuntimeError as caught: helper.require(str(caught)=='NONFINITE_TENSOR','finite error')
        else: raise RuntimeError('nonfinite record accepted')
    print(json.dumps({'status':'IMPLEMENTER_TINY_HELPER_ONLY','torch':torch.__version__,
                      'threads':torch.get_num_threads(),'max_tensor_elements':32,
                      'helper_sha256':hashlib.sha256((HERE/'compare_checkpoints.py').read_bytes()).hexdigest(),
                      'positive':result,'negative_controls':cases,'nonfinite_records_refused':3,
                      'fullsize_or_native_executed':False,'train_source_imported':False},indent=2,allow_nan=False))


if __name__=='__main__':main()
