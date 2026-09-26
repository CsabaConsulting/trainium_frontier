"""Disabled two-arm ordinary-loader CPU checkpoint qualification.

No launcher, source patch, dtype shim, data reader or native execution.
Real INPUTS must be separately frozen and supplied by exact SHA before run().
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import sys
import torch

ROOT = Path('/home/ubuntu/trainium_confirmed36_clean_cpu_20260926_trn21_r1')
PREPARE = Path('/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2/prepare.py')
PREPARE_SHA = '5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb'
SOURCES = {
    'reference':'36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb',
    'candidate':'e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43',
}
REQUESTS = {
    'reference':'38160400738af45274aad1c96f9a54e0c5ec1343ab9eb87f1279b46e4f5ba354',
    'candidate':'64d0b756784eb8a8306d1fcd739c09c355d7611479f8da9f64bf2fb7ffc4eca3',
}
G2_SHA = '6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8'
SHAPES = ((1,256),(2,256),(3,256),(1,257),(2,257),(3,257))


class CheckFailure(RuntimeError):
    def __init__(self,message,details=None):
        super().__init__(message)
        self.details=details


def require(ok,message):
    if not ok:
        raise CheckFailure(message)


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



def comparison(actual,expected):
    a=tensor_record(actual,'actual')
    b=tensor_record(expected,'expected')
    same=a['shape']==b['shape'] and a['dtype']==b['dtype']
    result={'actual':a,'expected':b,'exact_bytes':False,
            'max_abs_diff':None,'relative_rms':None}
    if same:
        result['exact_bytes']=bool(torch.equal(actual.contiguous().reshape(-1).view(torch.uint8),
                                              expected.contiguous().reshape(-1).view(torch.uint8)))
        delta=actual.double()-expected.double()
        result['max_abs_diff']=float(delta.abs().max().item())
        denominator=float(expected.double().square().mean().sqrt().item())
        numerator=float(delta.square().mean().sqrt().item())
        result['relative_rms']=numerator/denominator if denominator else (0.0 if numerator==0 else 'UNDEFINED_ZERO_REFERENCE')
    return result


def strict_equal(actual,expected,label):
    observed=comparison(actual,expected)
    if not observed['exact_bytes']:
        raise CheckFailure(label,observed)
    return observed


def check_modes_and_hooks(model):
    for name,module in model.named_modules():
        require(not module.training,'MODEL_MODE_CHANGED:'+name)
        for hook in ('_forward_hooks','_forward_pre_hooks','_backward_hooks','_backward_pre_hooks'):
            require(not getattr(module,hook,{}),'MODEL_HOOK:'+name+':'+hook)


def verify_shape(model,x,vocab_size,mask_spec=None):
    require(x.dtype==torch.int64 and x.device.type=='cpu' and x.ndim==2,'SYNTHETIC_INPUT_TYPE')
    batch,length=x.shape
    require(batch in (1,2,3) and length>=4,'SYNTHETIC_INPUT_SHAPE')
    require(bool(((x>=1)&(x<vocab_size)).all()),'SYNTHETIC_INPUT_RANGE')
    check_modes_and_hooks(model)
    before=model_state(model,mask_spec)
    initial_input=tensor_record(x,'input')
    calls=[]
    def forward(inp,label):
        inp_before=tensor_record(inp,'call_input')
        check_modes_and_hooks(model)
        with torch.no_grad():
            out=model(inp)
        require(isinstance(out,torch.Tensor) and list(out.shape)==[inp.shape[0],length,vocab_size],'OUTPUT_SHAPE:'+label)
        rec=tensor_record(out,'output:'+label)
        check_modes_and_hooks(model)
        require_state_equal(model_state(model,mask_spec),before)
        require(tensor_record(inp)==inp_before and tensor_record(x)==initial_input,'INPUT_CHANGED:'+label)
        calls.append({'call':label,'input':inp_before,'output':rec})
        # Snapshot now: later calls may alias and overwrite returned scratch.
        return out.detach().clone()
    original=forward(x,'original')
    repeat=forward(x,'repeat')
    repeated=strict_equal(repeat,original,'REPEATED_LOGITS_BYTES')
    rows=torch.cat([forward(x[i:i+1].clone(),'row'+str(i)) for i in range(batch)],dim=0)
    rowwise=strict_equal(rows,original,'ROWWISE_LOGITS_BYTES')
    cut=max(2,length//2)
    suffix=x.clone()
    suffix[:,cut:]=torch.flip(suffix[:,cut:],dims=[1])
    require(all(not torch.equal(x[i,cut:],suffix[i,cut:]) for i in range(batch)),'DEGENERATE_SUFFIX_FIXTURE')
    altered=forward(suffix,'reversed_suffix')
    prefix=original[:,:cut-1].float()
    changed=altered[:,:cut-1].float()
    error=float((prefix-changed).abs().max().item())
    causal={'synthetic_only':True,'cut':cut,'prefix_stop':cut-1,'max_abs_diff':error,
            'tolerance':1e-2,'prefix_exact':bool(torch.equal(prefix.contiguous().view(torch.uint8),
                                                           changed.contiguous().view(torch.uint8)))}
    if error>1e-2:
        raise CheckFailure('SYNTHETIC_PREFIX_CAUSALITY',causal)
    return original,{'shape':[batch,length],'input':initial_input,'calls':calls,'repeat':repeated,
                     'rowwise':rowwise,'causality':causal,'state_before':before,
                     'state_unchanged_after_every_forward':True,'ordinary_loader':True}


def full_payload(payload):
    require(isinstance(payload,dict),'PAYLOAD_DICT')
    toggles=payload.get('toggles',{})
    require(isinstance(toggles,dict) and toggles.get('USE_DOC_MASK',False) is False,'NO_DOCUMENT_DATA_ROUTE')
    config=payload.get('model_config')
    require(isinstance(config,dict),'MODEL_CONFIG')
    require(all(type(config.get(k)) is int and config[k]==v for k,v in
                (('n_layer',8),('n_embd',1024),('vocab_size',8192))),'FULL_CONFIG')
    require(type(config.get('sequence_len')) is int and config['sequence_len']>=257,'SAVED_CONTEXT')
    state=payload.get('model_state')
    require(isinstance(state,dict) and len(state)==65 and all(isinstance(k,str) and isinstance(t,torch.Tensor) for k,t in state.items()),'FULL_STATE_KEYS')
    require(sum(t.numel() for t in state.values())==113256724,'FULL_PARAMETER_COUNT')
    return {k:tensor_record(v,'saved_parameter:'+k) for k,v in state.items()}


def loaded_state(model,payload):
    actual=model.state_dict()
    require(actual.keys()==payload['model_state'].keys(),'LOADED_STATE_KEYS')
    for key in actual:
        strict_equal(actual[key],payload['model_state'][key],'LOADED_STATE_BYTES:'+key)


def mask_configuration(name,model,module,payload):
    if name=='reference':
        return restored_mask_spec(model,module,payload)
    require(name=='candidate' and type(model) is module.GPT,'CANDIDATE_NOT_ORDINARY_GPT')
    saved=payload['model_config']['sequence_len']
    minimum=module.EVAL_MIN_SEQ_LEN
    require(type(saved) is int and type(minimum) is int and saved>0 and minimum>0,'CANDIDATE_SEQUENCE_CONFIG')
    require(model.config.sequence_len==max(saved,minimum),'CANDIDATE_RESTORED_SEQUENCE_LENGTH')
    require(set(dict(model.named_buffers()))=={'cos','sin'},'CANDIDATE_BUFFER_LAYOUT')
    # Exact clean source contains no root mask/toggle globals. No exemption is
    # granted: ordinary model_state requires both clean rotary buffers finite.
    return None


def no_symlink(path):
    require(path.is_absolute() and '..' not in path.parts,'ABSOLUTE_CANONICAL_PATH')
    for part in (path,*path.parents):
        require(not part.is_symlink(),'SYMLINK:'+str(part))
    require(path.is_file(),'MISSING_FILE:'+str(path))


def file_record(path):
    no_symlink(path)
    before=path.stat()
    with path.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    after=path.stat()
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
            (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'FILE_CHANGED_DURING_HASH:'+str(path))
    return {'sha256':digest,'bytes':after.st_size}


def valid_pin(pin):
    return isinstance(pin,dict) and set(pin)=={'sha256','bytes'} and isinstance(pin['sha256'],str) and re.fullmatch('[a-f0-9]{64}',pin['sha256']) and type(pin['bytes']) is int and pin['bytes']>0


def authenticate_files(files):
    require(isinstance(files,dict) and bool(files),'FILES_REQUIRED')
    for path,pin in files.items():
        require(valid_pin(pin),'FILE_PIN:'+str(path))
        require(file_record(Path(path))==pin,'FILE_IDENTITY:'+str(path))
    return files


def validate_manifest(inputs):
    require(isinstance(inputs,dict) and inputs.get('status')=='FROZEN_REAL_INPUTS','REAL_INPUTS_PENDING')
    require(inputs.get('root')==str(ROOT),'EXACT_PAIR_ROOT')
    require(set(inputs.get('arms',{}))==set(SOURCES),'EXACT_TWO_ARMS')
    paths={str(PREPARE),str(ROOT/'preflight_dryrun.py'),str(ROOT/'standard-compare.json'),
           str(ROOT/'standard-compare-exit.json'),str(ROOT/'terminal.json')}
    for name in SOURCES:
        arm=inputs['arms'][name]
        require(set(arm)=={'checkpoint_path','checkpoint_sha256','checkpoint_bytes'},'CHECKPOINT_FIELDS:'+name)
        path=Path(arm['checkpoint_path'])
        try:relative=path.relative_to(ROOT/name)
        except ValueError:raise CheckFailure('CHECKPOINT_OUTSIDE_ARM:'+name)
        parts=relative.parts
        require(len(parts)==4 and parts[0]=='tmp' and re.fullmatch(r'g2-[A-Za-z0-9_-]+',parts[1]) and parts[2:]==('out','final.pt'),'CHECKPOINT_PRIVATE_PATH:'+name)
        pin={'sha256':arm['checkpoint_sha256'],'bytes':arm['checkpoint_bytes']}
        require(valid_pin(pin),'CHECKPOINT_PIN_PENDING:'+name)
        paths.add(str(path))
        for f in ('request.json','g2.json','terminal.json','checkpoint-identity.json','train.py'):
            paths.add(str(ROOT/name/f))
    files=inputs.get('files')
    require(isinstance(files,dict) and set(files)==paths,'EXACT_INPUT_FILE_SET')
    for pin in files.values():require(valid_pin(pin),'INPUT_PIN_PENDING')
    require(files[str(PREPARE)]['sha256']==PREPARE_SHA,'PREPARE_PIN')
    require(files[str(ROOT/'preflight_dryrun.py')]['sha256']==G2_SHA,'G2_TOOL_PIN')
    for name in SOURCES:
        arm=inputs['arms'][name]
        require(files[str(ROOT/name/'train.py')]['sha256']==SOURCES[name],'SOURCE_PIN:'+name)
        require(files[str(ROOT/name/'request.json')]['sha256']==REQUESTS[name],'REQUEST_PIN:'+name)
        require(files[arm['checkpoint_path']]=={'sha256':arm['checkpoint_sha256'],'bytes':arm['checkpoint_bytes']},'CHECKPOINT_MANIFEST_PIN:'+name)
    return files


def read_json(path):
    return json.loads(path.read_text())


def qualify_g2(receipt,request):
    require(receipt.get('verdict')=='PASS' and receipt.get('timed_out') is False and type(receipt.get('torchrun_rc')) is int and receipt['torchrun_rc']==0,'G2_PASS_REQUIRED')
    require(receipt.get('train_py_sha256')==request['source_sha256'] and receipt.get('argv')==request['argv'],'G2_SOURCE_ARGV')
    require([receipt.get(k) for k in ('steps','micro_batch','nproc')]==[3,[2,256],4],'G2_PROFILE')
    require(receipt.get('prepare',{}).get('sha256')==PREPARE_SHA and receipt['prepare'].get('stub') is False,'G2_REAL_PREPARE')
    ranks=receipt.get('ranks',[])
    require(len(ranks)==4 and all(type(r.get('rank')) is int for r in ranks) and {r.get('rank') for r in ranks}==set(range(4)),'G2_FOUR_RANKS')
    require(isinstance(receipt.get('plan_sha256'),str) and re.fullmatch('[a-f0-9]{64}',receipt['plan_sha256']),'G2_PLAN')
    for rank in ranks:
        require(rank.get('status')=='PASS' and rank.get('n_params')==65 and rank.get('cross_rank_mismatches')==[],'G2_RANK_STATUS')
        require(rank.get('plan_sha256')==receipt['plan_sha256'],'G2_RANK_PLAN')
        losses=rank.get('losses')
        require(isinstance(losses,list) and len(losses)==3 and all(type(v) in (int,float) and math.isfinite(v) for v in losses),'G2_FINITE_THREE_LOSSES')
    r0=next(r for r in ranks if r['rank']==0)
    require(r0.get('checkpoint_check',{}).get('ok') is True,'G2_CHECKPOINT')
    digests=r0.get('final_digests',{})
    # The unchanged standard G2 intentionally records 16-hex digest prefixes.
    # This is provenance only; full saved tensors are compared below by bytes.
    require(len(digests)==65 and all(isinstance(k,str) and isinstance(v,str) and re.fullmatch('[a-f0-9]{16}',v) for k,v in digests.items()),'G2_65_DIGESTS')
    return {'plan_sha256':receipt['plan_sha256'],'losses':{r['rank']:r['losses'] for r in ranks},'digests':digests}


def qualify_pair(inputs):
    records={}
    for name in SOURCES:
        request=read_json(ROOT/name/'request.json')
        require(request['source_sha256']==SOURCES[name],'REQUEST_SOURCE')
        terminal=read_json(ROOT/name/'terminal.json')
        require(type(terminal.get('exit_code')) is int and terminal['exit_code']==0 and terminal.get('verdict')=='PASS' and terminal.get('source_sha256')==SOURCES[name],'ARM_TERMINAL:'+name)
        identity=read_json(ROOT/name/'checkpoint-identity.json')
        arm=inputs['arms'][name]
        require(identity.get('path')==arm['checkpoint_path'] and identity.get('sha256')==arm['checkpoint_sha256'] and identity.get('bytes')==arm['checkpoint_bytes'],'CHECKPOINT_IDENTITY:'+name)
        records[name]=qualify_g2(read_json(ROOT/name/'g2.json'),request)
    require(records['reference']==records['candidate'],'G2_PAIR_NOT_IDENTICAL')
    comparison=read_json(ROOT/'standard-compare.json')
    for key in ('identical','plan_equal','losses_identical'):require(comparison.get(key) is True,'STANDARD_COMPARE:'+key)
    require(comparison.get('n_params_compared')==65 and all(comparison.get(k)==[] for k in ('state_keys_only_in_one','param_digest_differences')) and comparison.get('loss_differences')=={},'STANDARD_COMPARE_DETAILS')
    require(comparison.get('a_sha256')==SOURCES['reference'] and comparison.get('b_sha256')==SOURCES['candidate'],'STANDARD_COMPARE_SOURCES')
    comparison_exit=read_json(ROOT/'standard-compare-exit.json').get('exit_code')
    require(type(comparison_exit) is int and comparison_exit==0,'STANDARD_COMPARE_EXIT')
    terminal=read_json(ROOT/'terminal.json')
    require(type(terminal.get('exit_code')) is int and terminal['exit_code']==0 and terminal.get('status')=='CPU_PAIR_COMPLETE_PENDING_INDEPENDENT_REVIEW','PAIR_TERMINAL')
    return records


def load_module(name):
    spec=importlib.util.spec_from_file_location('checkpoint_clean_port_'+name,ROOT/name/'train.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def run(inputs_path,inputs_sha256,report):
    require(isinstance(inputs_sha256,str) and re.fullmatch('[a-f0-9]{64}',inputs_sha256),'INPUTS_SHA_REQUIRED')
    input_identity=file_record(inputs_path)
    require(input_identity['sha256']==inputs_sha256,'INPUTS_SHA')
    inputs=read_json(inputs_path)
    files=validate_manifest(inputs)
    report['input_manifest']={'path':str(inputs_path),**input_identity}
    authenticate_files(files)
    report['input_hashes_pending_until_frozen_receipt']=False
    try:
        report['pair']=qualify_pair(inputs)
        torch.set_num_threads(1)
        # Do not inject a prepare shim/path. The future reviewed execution
        # environment must already make this exact original module available.
        import prepare
        require(Path(prepare.__file__).absolute()==PREPARE,'ACTUAL_PREPARE_ORIGIN')
        require(file_record(PREPARE)['sha256']==PREPARE_SHA,'ACTUAL_PREPARE_SHA')
        observations={}
        saved_records={}
        for name in SOURCES:
            path=Path(inputs['arms'][name]['checkpoint_path'])
            payload=torch.load(path,map_location='cpu',weights_only=True)
            saved_records[name]=full_payload(payload)
            if name=='candidate':require(saved_records[name]==saved_records['reference'],'SAVED_STATE_BYTES')
            module=load_module(name)
            model=module.load_for_eval(str(path),torch.device('cpu'))
            require(isinstance(model,module.GPT),'LOADER_GPT')
            require(all(not t.requires_grad for t in model.parameters()),'LOADER_NOT_FROZEN')
            require(sum(t.numel() for t in model.parameters())==113256724 and len(list(model.parameters()))==65,'LOADED_FULL_MODEL')
            loaded_state(model,payload)
            masks=mask_configuration(name,model,module,payload)
            arm_record={'source_sha256':SOURCES[name],'checkpoint':inputs['arms'][name],
                        'loader_class':type(model).__name__,'dtype_override':False,'mask_configuration':masks,'shapes':[]}
            report['arms'][name]=arm_record
            state_initial=model_state(model,masks)
            for batch,length in SHAPES:
                x=torch.randint(1,8192,(batch,length),generator=torch.Generator().manual_seed(7))
                output,shape_record=verify_shape(model,x,8192,masks)
                require_state_equal(model_state(model,masks),state_initial)
                arm_record['shapes'].append(shape_record)
                if name=='reference':observations[(batch,length)]=output
                else:shape_record['cross_source']=strict_equal(output,observations.pop((batch,length)),'CROSS_SOURCE_LOGITS_BYTES:'+str((batch,length)))
            del model,module,payload
            gc.collect()
        require(not observations,'ALL_SIX_CROSS_SOURCE_SHAPES')
        report['status']='CPU_CHECKPOINT_SYNTHETIC_QUALIFICATION_COMPLETE'
    except Exception as error:
        report['model_or_pair_error']={'type':type(error).__name__,'message':str(error),
                                      'comparison':getattr(error,'details',None)}
        raise
    finally:
        # Immutable sources, requests, G2 receipts and both retained checkpoints
        # are rehashed on successful and failed model checks alike.
        authenticate_files(files)
        require(file_record(inputs_path)==input_identity,'INPUTS_CHANGED')
        report['inputs_unchanged_after']=True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--inputs-sha256',required=True)
    args=parser.parse_args()
    report={'status':'REFUSED_OR_INCOMPLETE','arms':{},'python':platform.python_version(),
            'torch':torch.__version__,'threads':1,'native_correctness':False,
            'organizer_causality_check_called':False,'quality_or_submission_claim':False,
            'source_changed':False,'input_hashes_pending_until_frozen_receipt':True}
    try:
        run(args.inputs,args.inputs_sha256,report)
    except Exception as error:
        report['status']='REFUSED_OR_INCOMPLETE'
        report['error']={'type':type(error).__name__,'message':str(error),
                         'comparison':getattr(error,'details',None)}
        print(json.dumps(report,indent=2,allow_nan=False))
        return 1
    print(json.dumps(report,indent=2,allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
