"""CPU-only strict retained model-state and ordinary loader-logit comparison.

One diagnosed checkpoint-only retry. Original r2 qualification remains FAILED.
No native training, no real data and no evaluator modifications.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import torch


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


EXPECTED_SOURCES = {
    'reference': '36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb',
    'default': '4bfaec501c2e0e499803eaafbf7526819f248936369e7d8cbbc973a58cf0b1ea',
    'off': '4bfaec501c2e0e499803eaafbf7526819f248936369e7d8cbbc973a58cf0b1ea',
    'on': '4bfaec501c2e0e499803eaafbf7526819f248936369e7d8cbbc973a58cf0b1ea',
}


OLD_ROOT = Path('/home/ubuntu/trainium_clock36_recursive_cpu_20260926_trn22_r2')
EXPECTED_CHECKPOINTS = {
    'reference': ('reference/tmp/g2-t650_0z_/out/final.pt', '623577bb7e6f60d183a1944b28a5624cd55767c16fcf03a9248fd4573f6a6762'),
    'default': ('default/tmp/g2-vi7085z9/out/final.pt', 'b5b4c3ab997f451f25608f4d55a42ca6680069320068162214bfd293b5bc41a7'),
    'off': ('off/tmp/g2-j8pqv5eh/out/final.pt', 'b5b4c3ab997f451f25608f4d55a42ca6680069320068162214bfd293b5bc41a7'),
    'on': ('on/tmp/g2-jwnj1q6g/out/final.pt', 'b006932d75820d40038068335c93fd1690788c444de42f877422ef6f0bac0f17'),
}


def raw_record(t):
    value=t.detach().cpu().contiguous()
    raw=value.reshape(-1).view(torch.uint8).numpy()
    return {'shape':list(value.shape),'dtype':str(value.dtype),
            'sha256':hashlib.sha256(memoryview(raw)).hexdigest(),
            'requires_grad':t.requires_grad}


def tensor_record(t,name='tensor'):
    require(isinstance(t,torch.Tensor),'TENSOR_REQUIRED:'+name)
    require(bool(torch.isfinite(t).all()),'NONFINITE_TENSOR:'+name)
    return raw_record(t)


def restored_mask_spec(model,module,payload):
    saved_length=payload['model_config']['sequence_len']
    minimum=module.EVAL_MIN_SEQ_LEN
    require(type(saved_length) is int and saved_length>0,'SAVED_SEQUENCE_LENGTH')
    require(type(minimum) is int and minimum>0,'EVAL_MIN_SEQUENCE_LENGTH')
    length=max(saved_length,minimum)
    require(type(model.config.sequence_len) is int and model.config.sequence_len==length,'RESTORED_SEQUENCE_LENGTH')
    saved=payload.get('toggles',{})
    require(isinstance(saved,dict),'SAVED_TOGGLES')
    fraction=saved.get('SHORT_WINDOW_FRAC',module._EVAL_TOGGLE_DEFAULTS['SHORT_WINDOW_FRAC'])
    require(type(fraction) in (float,int) and math.isfinite(fraction),'SAVED_SHORT_WINDOW_FRAC')
    require(type(module.SHORT_WINDOW_FRAC) in (float,int) and module.SHORT_WINDOW_FRAC==fraction,'RESTORED_SHORT_WINDOW_FRAC')
    window=max(1,int(length*fraction))
    return {'sequence_len':length,'full_window':length,'short_window':window,
            'saved_sequence_len':saved_length,'eval_min_sequence_len':minimum,
            'restored_short_window_frac':fraction}


def mask_record(t,length,window,name):
    # This exemption is called ONLY for named root buffers, never parameters.
    require(isinstance(t,torch.Tensor),'TENSOR_REQUIRED:'+name)
    require(type(length) is int and length>0 and type(window) is int and window>0,'MASK_CONFIG:'+name)
    require(t.device.type=='cpu' and t.dtype==torch.float32 and not t.requires_grad,'MASK_DTYPE_DEVICE_GRAD:'+name)
    require(list(t.shape)==[1,1,length,length],'MASK_SHAPE:'+name)
    zero_count=0
    j=torch.arange(length).reshape(1,-1)
    # Bound scratch even for the loader's widened8192 tables. This is an
    # independent causal formula, not the candidate's mask construction call.
    for start in range(0,length,64):
        stop=min(start+64,length)
        i=torch.arange(start,stop).reshape(-1,1)
        visible=(j<=i)&(j>i-window)
        expected=torch.zeros((stop-start,length),dtype=torch.float32)
        expected.masked_fill_(~visible,-float('inf'))
        actual=t[0,0,start:stop].contiguous()
        require(not bool(torch.isnan(actual).any()),'MASK_NAN:'+name)
        require(not bool(torch.isposinf(actual).any()),'MASK_POSINF:'+name)
        require(torch.equal(actual.reshape(-1).view(torch.uint8),expected.reshape(-1).view(torch.uint8)),
                'MASK_VALUES:'+name+':rows'+str(start)+':'+str(stop))
        zero_count+=int(visible.sum())
    record=raw_record(t)
    record.update({'mask_semantics_validated':True,'window':window,'zero_count':zero_count,
                   'negative_infinity_count':length*length-zero_count,'nan_count':0,'positive_infinity_count':0})
    return record


def model_state(model,mask_spec=None):
    # Includes nonpersistent buffers. Parameters never receive the exception,
    # even if maliciously named mask_full or mask_short.
    records={}
    for name,value in model.named_parameters():
        records['parameter:'+name]=tensor_record(value,'parameter:'+name)
    buffers=dict(model.named_buffers())
    if mask_spec is not None:
        require({'mask_full','mask_short'}<=buffers.keys(),'MISSING_ROOT_MASK_BUFFERS')
    for name,value in buffers.items():
        key='buffer:'+name
        if mask_spec is not None and name in ('mask_full','mask_short'):
            window=mask_spec['full_window' if name=='mask_full' else 'short_window']
            records[key]=mask_record(value,mask_spec['sequence_len'],window,key)
        else:
            records[key]=tensor_record(value,key)
    return records


def require_state_equal(actual,expected):
    changed=sorted(k for k in actual.keys()|expected.keys() if actual.get(k)!=expected.get(k))
    require(not changed,'MODEL_STATE_CHANGED:'+','.join(changed))


def verify_trained_model(model,x,vocab_size,mask_spec=None):
    require(not model.training,'MODEL_NOT_EVAL')
    require(x.ndim==2 and x.shape[0]==1 and x.shape[1]>=4,'SYNTHETIC_INPUT_SHAPE')
    before=model_state(model,mask_spec)
    input_before=tensor_record(x,'input')
    calls=[]
    def forward(inp):
        expected_input=tensor_record(inp,'call_input')
        with torch.no_grad():
            out=model(inp)
        require(isinstance(out,torch.Tensor) and list(out.shape)==[1,x.shape[1],vocab_size],'OUTPUT_SHAPE')
        output_record=tensor_record(out,'output')
        require(not model.training,'MODEL_MODE_CHANGED')
        require_state_equal(model_state(model,mask_spec),before)
        require(tensor_record(inp)==expected_input and tensor_record(x)==input_before,'INPUT_CHANGED')
        calls.append(output_record)
        return out
    # Own the observations: an output tensor may alias a model's scratch storage.
    a=forward(x).detach().clone()
    b=forward(x).detach().clone()
    require(torch.equal(a.reshape(-1).view(torch.uint8),b.reshape(-1).view(torch.uint8)),'REPEATED_LOGITS_BYTES')
    cut=max(2,x.shape[1]//2)
    suffix=x.clone()
    suffix[:,cut:]=torch.flip(suffix[:,cut:],dims=[1])
    require(not torch.equal(x[:,cut:],suffix[:,cut:]),'DEGENERATE_SUFFIX_FIXTURE')
    changed=forward(suffix)
    # Pinned prepare.py:663-684 arithmetic, but synthetic input, no data loader.
    prefix=a[:,:cut-1].float()
    modified=changed[:,:cut-1].float()
    error=float((prefix-modified).abs().max().item())
    require(error<=1e-2,'SYNTHETIC_PREFIX_CAUSALITY')
    return {'route':'ordinary_unpatched_cpu_load_for_eval','input':input_before,
            'outputs':calls,'repeated_bytes_equal':True,'model_state_equal_after_each_call':True,
            'mask_configuration':mask_spec,'named_parameters_and_all_buffers':before,'state_tensor_count':len(before),
            'causality':{'synthetic_only':True,'cut':cut,'prefix_stop':cut-1,'max_abs_diff':error,
                         'tolerance':1e-2,'passed':True,'prefix_exact':torch.equal(prefix,modified)},
            'organizer_causality_check_called':False}


def run(root):
    require(root==OLD_ROOT,'EXACT_READONLY_R2_ROOT')
    torch.set_num_threads(1)
    records=[]
    states=[]
    logits=[]
    x=torch.randint(1,8192,(1,256),generator=torch.Generator().manual_seed(7))
    for name in ('reference','default','off','on'):
        request=json.loads((root/name/'request.json').read_text())
        require(request['source_sha256']==EXPECTED_SOURCES[name],'EXPECTED_SOURCE')
        checkpoint=json.loads((root/name/'checkpoint-identity.json').read_text())
        path=Path(checkpoint['path'])
        relative,pin=EXPECTED_CHECKPOINTS[name]
        require(path==OLD_ROOT/relative and checkpoint['sha256']==pin,'EXACT_RETAINED_CHECKPOINT:'+name)
        with path.open('rb') as stream:
            require(hashlib.file_digest(stream,'sha256').hexdigest() == checkpoint['sha256'],'CHECKPOINT_SHA')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        require(payload['model_config']['n_layer'] == 8 and payload['model_config']['n_embd'] == 1024,'FULL_CONFIG')
        require(len(payload['model_state'])==65 and sum(t.numel() for t in payload['model_state'].values())==113256724,'FULL_CHECKPOINT_STATE')
        if name!='on':
            states.append(payload['model_state'])
        source=root/name/'train.py'
        require(hashlib.sha256(source.read_bytes()).hexdigest() == request['source_sha256'],'SOURCE_SHA')
        spec=importlib.util.spec_from_file_location('checkpoint_'+name,source)
        mod=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=mod
        spec.loader.exec_module(mod)
        # New r2 qualification route has no source/dtype/global/forward override.
        ordinary=mod.load_for_eval(str(path),torch.device('cpu'))
        loaded=ordinary.state_dict()
        require(loaded.keys()==payload['model_state'].keys(),'LOADED_STATE_KEYS')
        for key,value in loaded.items():
            require(tensor_record(value,'loaded_parameter:'+key)==tensor_record(payload['model_state'][key],'saved_parameter:'+key),'ORDINARY_LOADED_STATE:'+key)
        mask_spec=restored_mask_spec(ordinary,mod,payload)
        qualification=verify_trained_model(ordinary,x,8192,mask_spec)
        del loaded,ordinary
        records.append({'name':name,'checkpoint_sha256':checkpoint['sha256'],
                        'source_sha256':request['source_sha256'],'trained_model_qualification':qualification})
        if name=='on':
            del mod,payload
            continue
        # Retain the pre-existing comparison route; its BF16 CPU shim is explicit
        # and never used by the separate ordinary-loader r2 qualification above.
        mod.compute_dtype_for=lambda device:torch.bfloat16
        model=mod.load_for_eval(str(path),torch.device('cpu'))
        with torch.no_grad():
            a,b=model(x),model(x)
        require(torch.equal(a,b) and torch.isfinite(a).all(),'DETERMINISTIC_FINITE_LOGITS')
        logits.append(a)
        del model,mod,payload
    require(all(states[0].keys() == state.keys() for state in states) and len(states[0]) == 65,'ALL65_STATE_KEYS')
    require(sum(t.numel() for t in states[0].values()) == 113256724,'FULL_STATE_SIZE')
    for state in states[1:]:
        for key in states[0]:
            a,b=states[0][key],state[key]
            require(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.reshape(-1).view(torch.uint8),b.reshape(-1).view(torch.uint8)), 'STATE_BYTES:'+key)
            require(torch.isfinite(a).all(),'FINITE_STATE:'+key)
    require(all(torch.equal(logits[0],value) for value in logits[1:]),'RELOADED_B1_LOGITS_EXACT')
    require([r['name'] for r in records]==['reference','default','off','on'],'ALL_FOUR_QUALIFIED')
    return {'status':'FULL_PARAMETER_CPU_CHECKPOINT_AND_LOGIT_IDENTITY','states':65,'parameters':113256724,'logits_shape':list(logits[0].shape),'records':records,'torch':torch.__version__,'compute_dtype_shim':'CPU bfloat16 only for preserved three-arm comparison; new four-arm qualification unpatched','synthetic_input':[1,256],'native_correctness':False,'quality_evidence':False,'original_r2_qualification':'FAILED_PRESERVED','checkpoint_only_retry':True,'on_off_equivalence_claimed':False,'trained_model_causality_and_state_scope':'four ordinary CPU loaders,synthetic B1T256,not native or organizer data'}


if __name__=='__main__':
    print(json.dumps(run(Path(sys.argv[1])),indent=2,allow_nan=False))
