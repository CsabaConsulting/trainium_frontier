"""CPU-only strict retained model-state and ordinary loader-logit comparison.

Future Operator child only, after both full-parameter G2 receipts pass.
No native training, no real data and no evaluator modifications.
"""
import hashlib
import importlib.util
import json
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


def tensor_record(t):
    require(isinstance(t,torch.Tensor),'TENSOR_REQUIRED')
    value=t.detach().cpu().contiguous()
    require(bool(torch.isfinite(value).all()),'NONFINITE_TENSOR')
    raw=value.reshape(-1).view(torch.uint8).numpy()
    return {'shape':list(value.shape),'dtype':str(value.dtype),
            'sha256':hashlib.sha256(memoryview(raw)).hexdigest(),
            'requires_grad':t.requires_grad}


def model_state(model):
    # Includes nonpersistent registered buffers, not only checkpoint state_dict.
    return {kind+':'+name:tensor_record(value) for kind,items in (
        ('parameter',model.named_parameters()),('buffer',model.named_buffers())) for name,value in items}


def verify_trained_model(model,x,vocab_size):
    require(not model.training,'MODEL_NOT_EVAL')
    require(x.ndim==2 and x.shape[0]==1 and x.shape[1]>=4,'SYNTHETIC_INPUT_SHAPE')
    before=model_state(model)
    input_before=tensor_record(x)
    calls=[]
    def forward(inp):
        expected_input=tensor_record(inp)
        with torch.no_grad():
            out=model(inp)
        require(isinstance(out,torch.Tensor) and list(out.shape)==[1,x.shape[1],vocab_size],'OUTPUT_SHAPE')
        output_record=tensor_record(out)
        require(not model.training,'MODEL_MODE_CHANGED')
        require(model_state(model)==before,'MODEL_STATE_CHANGED')
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
            'named_parameters_and_all_buffers':before,'state_tensor_count':len(before),
            'causality':{'synthetic_only':True,'cut':cut,'prefix_stop':cut-1,'max_abs_diff':error,
                         'tolerance':1e-2,'passed':True,'prefix_exact':torch.equal(prefix,modified)},
            'organizer_causality_check_called':False}


def run(root):
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
            require(tensor_record(value)==tensor_record(payload['model_state'][key]),'ORDINARY_LOADED_STATE:'+key)
        qualification=verify_trained_model(ordinary,x,8192)
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
    return {'status':'FULL_PARAMETER_CPU_CHECKPOINT_AND_LOGIT_IDENTITY','states':65,'parameters':113256724,'logits_shape':list(logits[0].shape),'records':records,'torch':torch.__version__,'compute_dtype_shim':'CPU bfloat16 only for preserved three-arm comparison; new four-arm qualification unpatched','synthetic_input':[1,256],'native_correctness':False,'quality_evidence':False,'on_off_equivalence_claimed':False,'trained_model_causality_and_state_scope':'four ordinary CPU loaders,synthetic B1T256,not native or organizer data'}


if __name__=='__main__':
    print(json.dumps(run(Path(sys.argv[1])),indent=2,allow_nan=False))
