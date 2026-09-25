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


def run(root):
    torch.set_num_threads(1)
    records=[]
    states=[]
    logits=[]
    x=torch.randint(1,8192,(1,256),generator=torch.Generator().manual_seed(7))
    for name in ('reference','candidate'):
        request=json.loads((root/name/'request.json').read_text())
        checkpoint=json.loads((root/name/'checkpoint-identity.json').read_text())
        path=Path(checkpoint['path'])
        with path.open('rb') as stream:
            require(hashlib.file_digest(stream,'sha256').hexdigest() == checkpoint['sha256'],'CHECKPOINT_SHA')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        require(payload['model_config']['n_layer'] == 8 and payload['model_config']['n_embd'] == 1024,'FULL_CONFIG')
        states.append(payload['model_state'])
        source=root/name/'train.py'
        require(hashlib.sha256(source.read_bytes()).hexdigest() == request['source_sha256'],'SOURCE_SHA')
        spec=importlib.util.spec_from_file_location('checkpoint_'+name,source)
        mod=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=mod
        spec.loader.exec_module(mod)
        mod.compute_dtype_for=lambda device:torch.bfloat16
        model=mod.load_for_eval(str(path),torch.device('cpu'))
        if name == 'candidate':
            require(type(model) is mod.GPT,'ORDINARY_GPT_LOADER')
        with torch.no_grad():
            a,b=model(x),model(x)
        require(torch.equal(a,b) and torch.isfinite(a).all(),'DETERMINISTIC_FINITE_LOGITS')
        logits.append(a)
        records.append({'name':name,'checkpoint_sha256':checkpoint['sha256'],'source_sha256':request['source_sha256']})
        del model,mod,payload
    require(states[0].keys() == states[1].keys() and len(states[0]) == 65,'ALL65_STATE_KEYS')
    require(sum(t.numel() for t in states[0].values()) == 113256724,'FULL_STATE_SIZE')
    for key in states[0]:
        a,b=states[0][key],states[1][key]
        require(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.uint8),b.view(torch.uint8)), 'STATE_BYTES:'+key)
        require(torch.isfinite(a).all(),'FINITE_STATE:'+key)
    require(torch.equal(logits[0],logits[1]),'RELOADED_B1_LOGITS_EXACT')
    return {'status':'FULL_PARAMETER_CPU_CHECKPOINT_AND_LOGIT_IDENTITY','states':65,'parameters':113256724,'logits_shape':list(logits[0].shape),'records':records,'torch':torch.__version__,'compute_dtype_shim':'CPU bfloat16','synthetic_input':[1,256],'native_correctness':False,'quality_evidence':False}


if __name__=='__main__':
    print(json.dumps(run(Path(sys.argv[1])),indent=2,allow_nan=False))
