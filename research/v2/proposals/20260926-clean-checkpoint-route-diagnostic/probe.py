"""Disabled retained-checkpoint diagnostic; no training, data reads, hooks or source changes."""
import argparse
import gc
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

CANDIDATE_SHA='3f5035da293d984e15df915e2348258aea7f399bbf4d0e47aeb5c9896c35a47f'
REFERENCE_SHA='3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c'
CHECKPOINT_SHA='068749c2559f7c68f6acc76d558caff9d3806d85273fa8562483d87bbad50d25'
PREPARE_SHA='5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb'
T=1024
V=8192

def require(ok,message):
    if not ok: raise RuntimeError(message)

def file_sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

def finite_metric(value):
    return float(value) if math.isfinite(float(value)) else None

def verdict(cases):
    return 'BEHAVIOURAL_DIAGNOSTIC_EXACT_ON_TESTED_SCOPE' if cases and all(c.get('passed') is True for c in cases) else 'HOLD_NUMERIC_OR_STATE_DIFFERENCE'

def write_json(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def run(args,receipt):
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    if args.backend=='neuron':
        import torch_neuronx
        require(os.environ.get('WORLD_SIZE')=='4' and os.environ.get('NEURON_LOGICAL_NC_CONFIG')=='2','NATIVE_FOUR_RANK_LNC2')
        dist.init_process_group('neuron')
        require(hasattr(torch,'neuron') and hasattr(torch.neuron,'synchronize'),'NATIVE_SYNC_MISSING')
    device=torch.device(args.backend)
    receipt['torch']=torch.__version__
    receipt['package_versions']={}
    for package in ('torch-neuronx','neuronx-distributed','nki'):
        try:receipt['package_versions'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:receipt['package_versions'][package]='UNAVAILABLE'
    receipt['world_size']=dist.get_world_size() if dist.is_initialized() else 1
    receipt['environment']={k:v for k,v in os.environ.items() if k.startswith(('NEURON_','TORCH_NEURONX_')) or k in ('RANK','LOCAL_RANK','WORLD_SIZE','OMP_NUM_THREADS','MKL_NUM_THREADS','TRN_EXP_LABEL')}
    # Fixed synthetic sequences, no tokenizer/dataset/public/private examples.
    generator=torch.Generator().manual_seed(42)
    inputs={k:torch.randint(1,V,(1,T),generator=generator,dtype=torch.long) for k in ('A','B','C')}
    def raw(t):return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    def tensor_record(t):
        t=t.detach().cpu()
        return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':hashlib.sha256(raw(t)).hexdigest(),'finite':bool(torch.isfinite(t).all())}
    receipt['inputs']={k:{**tensor_record(v),'tokens':v.tolist()} for k,v in inputs.items()}
    def snapshot(model):
        tensors={};scalars={}
        for name,mod in model.named_modules():
            require(not mod.training,'EVAL_MODE:'+name)
            for hook in ('_forward_hooks','_forward_pre_hooks','_backward_hooks','_backward_pre_hooks'):
                require(not getattr(mod,hook,{}),'MODEL_HOOKS:'+name+':'+hook)
            scalars[name+'.training']=bool(mod.training)
            for key,value in vars(mod).items():
                if torch.is_tensor(value):tensors[name+'.attr.'+key]=tensor_record(value)
            for key,value in mod._parameters.items():
                if value is not None:tensors[name+'.parameter.'+key]={**tensor_record(value),'requires_grad':bool(value.requires_grad)}
            for key,value in mod._buffers.items():
                if value is not None:tensors[name+'.buffer.'+key]=tensor_record(value)
        return {'tensors':tensors,'training_flags':scalars}
    def call(model,x):
        with torch.no_grad():out=model(x.to(device))
        if args.backend=='neuron':torch.neuron.synchronize()
        out=out.detach().cpu()
        require(tuple(out.shape)==(x.shape[0],T,V),'LOGIT_SHAPE')
        return out
    def compare(name,a,b,causal=False):
        ar,br=tensor_record(a),tensor_record(b)
        compatible=a.shape==b.shape and a.dtype==b.dtype
        exact=compatible and raw(a)==raw(b)
        row={'name':name,'criterion':'prepare synthetic suffix causal1e-2' if causal else 'same-backend identical dtype/shape/bytes; any difference HOLD',
             'actual':ar,'expected':br,'exact_bytes':bool(exact),'passed':False}
        if compatible:
            af,bf=a.float(),b.float();difference=(af-bf).abs()
            maximum=difference.max().item();finite=ar['finite'] and br['finite'] and math.isfinite(maximum)
            relative=(difference/bf.abs().clamp_min(1e-12)).max().item()
            rms=(difference.square().mean().sqrt()/bf.square().mean().sqrt().clamp_min(1e-12)).item()
            flat=int(difference.reshape(-1).argmax().item()) if finite else None
            coordinate=list(torch.unravel_index(torch.tensor(flat),difference.shape)) if flat is not None else []
            row.update(element_equal=bool(torch.equal(a,b)),max_abs=finite_metric(maximum),max_relative=finite_metric(relative),
               relative_denominator='max(abs(expected),1e-12)',relative_rms=finite_metric(rms),
               rms_denominator='max(sqrt(mean(expected**2)),1e-12)',element_mismatch_count=int((a!=b).sum()),
               byte_mismatch_count=int((a.contiguous().reshape(-1).view(torch.uint8)!=b.contiguous().reshape(-1).view(torch.uint8)).sum()),
               max_error_coordinate=[int(v) for v in coordinate],
               actual_at_max=finite_metric(af.reshape(-1)[flat].item()) if flat is not None else None,
               expected_at_max=finite_metric(bf.reshape(-1)[flat].item()) if flat is not None else None,
               metric_nonfinite=not (math.isfinite(maximum) and math.isfinite(relative) and math.isfinite(rms)))
            row['passed']=bool(finite and (maximum<=1e-2 if causal else exact))
            if causal:row.update(tolerance=1e-2,cut=max(2,T//2),compared_prefix=max(2,T//2)-1)
        receipt['cases'].append(row)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    checkpoint_state={k:tensor_record(v) for k,v in payload['model_state'].items()}
    require(len(checkpoint_state)==65 and sum(v.numel() for v in payload['model_state'].values())==113256724,'CHECKPOINT_STATE')
    receipt['checkpoint_state']=checkpoint_state
    receipt['checkpoint_model_config']=payload['model_config']
    del payload
    cut=max(2,T//2)
    ab=torch.cat([inputs['A'],inputs['B']])
    suffix=ab.clone();suffix[:,cut:]=torch.flip(suffix[:,cut:],dims=[1])
    matrix=[('A',inputs['A']),('B',inputs['B']),('AB',ab),
            ('BA',ab.flip(0)),('AC',torch.cat([inputs['A'],inputs['C']])),
            ('AA',torch.cat([inputs['A'],inputs['A']])),('suffix_AB',suffix),
            ('AB_repeat',ab),('A_repeat',inputs['A'])]
    receipt['call_order']=[key for key,_ in matrix]
    reference={}
    for name,source in [('reference',args.reference),('candidate',args.candidate)]:
        mod=load_module('route_'+name,source)
        model=mod.load_for_eval(str(args.checkpoint),device)
        require(not model.training and all(not p.requires_grad for p in model.parameters()),'FROZEN_EVAL_MODEL')
        require(sum(p.numel() for p in model.parameters())==113256724,'PARAMETER_COUNT')
        require(len(list(model.named_parameters()))==65,'PARAMETER_TENSORS')
        if name=='candidate':require(type(model) is mod.GPT,'ORDINARY_CANDIDATE_GPT')
        loaded_state={k:tensor_record(v) for k,v in model.state_dict().items()}
        require(loaded_state==checkpoint_state,'LOADED_STATE_CHECKPOINT_BYTES')
        before=snapshot(model)
        from dataclasses import asdict
        receipt.setdefault('models',{})[name]={'class':type(model).__name__,'compute_dtype':str(mod.COMPUTE_DTYPE),
             'effective_config':asdict(model.config),'parameters':113256724,'parameter_tensors':65,
             'loaded_state':loaded_state,'state_before':before}
        outputs={}
        for key,value in matrix:
            outputs[key]=call(model,value)
            receipt.setdefault('outputs',{}).setdefault(name,{})[key]=tensor_record(outputs[key])
        compare(name+'_B2_vs_independent_rows',outputs['AB'],torch.cat([outputs['A'],outputs['B']]))
        compare(name+'_permutation',outputs['BA'].flip(0),outputs['AB'])
        compare(name+'_partner_content_A',outputs['AC'][:1],outputs['A'])
        compare(name+'_duplicate_AA',outputs['AA'],torch.cat([outputs['A'],outputs['A']]))
        compare(name+'_repeat_AB',outputs['AB_repeat'],outputs['AB'])
        compare(name+'_repeat_A',outputs['A_repeat'],outputs['A'])
        compare(name+'_synthetic_suffix_causality',outputs['suffix_AB'][:,:cut-1].float(),outputs['AB'][:,:cut-1].float(),causal=True)
        if name=='reference':reference=outputs
        else:
            for key,_ in matrix:compare('source_parity_'+key,outputs[key],reference[key])
        after=snapshot(model)
        receipt['models'][name]['state_after']=after
        receipt['cases'].append({'name':name+'_state_invariance','passed':before==after,
          'criterion':'all registered parameters/buffers and direct tensor attributes plus training flags unchanged; arbitrary nested Python state not asserted'})
        del model,mod,outputs
        gc.collect()
    require(file_sha(args.checkpoint)==CHECKPOINT_SHA,'CHECKPOINT_CHANGED')
    require(receipt['inputs']=={k:{**tensor_record(v),'tokens':v.tolist()} for k,v in inputs.items()},'INPUTS_MUTATED')
    receipt['coverage_limit']='Fixed synthetic T1024 and specified B1/B2 calls only; not all possible inputs/batch sizes, scorer acceptance, fresh fullrun or upload eligibility.'
    receipt['status']=verdict(receipt['cases'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--enable-diagnostic',action='store_true')
    p.add_argument('--backend',choices=['cpu','neuron'])
    p.add_argument('--label')
    for key in ('candidate','reference','checkpoint','prepare','out-dir'):p.add_argument('--'+key,type=Path)
    args=p.parse_args()
    if not args.enable_diagnostic:
        print('{"status":"DISABLED","model_imported":false}');return 0
    require(all(getattr(args,k) is not None for k in ('backend','label','candidate','reference','checkpoint','prepare','out_dir')),'MISSING_BINDING')
    rank=int(os.environ.get('RANK','0'))
    args.out_dir.mkdir(exist_ok=True)
    receipt={'scope':'CHECKPOINT_ONLY_SYNTHETIC_ROUTE_DIAGNOSTIC','status':'UNRESOLVED_EXECUTION_OR_EVIDENCE','backend':args.backend,'rank':rank,'label':args.label,
      'candidate_sha256':CANDIDATE_SHA,'reference_sha256':REFERENCE_SHA,'checkpoint_sha256':CHECKPOINT_SHA,'prepare_sha256':PREPARE_SHA,
      'argv':sys.argv[1:],'cases':[],'source_review':'REFUSE_UNCHANGED','full_training':'HOLD','upload':'HOLD','source_edits':False,
      'quality_evidence':False,'launch_authorized':False,'started_monotonic':time.monotonic()}
    try:
        for path,pin in [(args.candidate,CANDIDATE_SHA),(args.reference,REFERENCE_SHA),(args.checkpoint,CHECKPOINT_SHA),(args.prepare,PREPARE_SHA)]:
            require(file_sha(path)==pin,'INPUT_SHA:'+str(path))
        require(Path(__file__).parent.resolve()==args.candidate.parent.resolve()==args.reference.parent.resolve(),'SOURCE_PACKET_ROOT')
        # Real, pinned prepare import; no stubs, replacing globals, data loading or scorer edits.
        sys.path.insert(0,str(args.prepare.parent))
        run(args,receipt)
    except Exception as exc:
        receipt.update(status='UNRESOLVED_EXECUTION_OR_EVIDENCE',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        try:
            if 'torch.distributed' in sys.modules:
                dist=sys.modules['torch.distributed']
                if dist.is_initialized():dist.destroy_process_group()
        except Exception as exc:
            receipt['cleanup_error']=type(exc).__name__+': '+str(exc);receipt['status']='UNRESOLVED_EXECUTION_OR_EVIDENCE'
        receipt['elapsed_seconds']=time.monotonic()-receipt['started_monotonic']
        serialized=json.dumps(receipt,allow_nan=False)
        if len(serialized.encode())>16*1024**2:
            receipt={'status':'UNRESOLVED_EXECUTION_OR_EVIDENCE','error':'RECEIPT_SIZE_CAP','rank':rank,'label':args.label}
        write_json(args.out_dir/('rank'+str(rank)+'.json'),receipt)
    return 0 if receipt['status']=='BEHAVIOURAL_DIAGNOSTIC_EXACT_ON_TESTED_SCOPE' else 2
if __name__=='__main__':raise SystemExit(main())
