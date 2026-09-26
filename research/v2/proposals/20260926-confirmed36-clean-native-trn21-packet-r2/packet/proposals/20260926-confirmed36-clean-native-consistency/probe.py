"""Disabled native synthetic diagnostic. No launcher or authority is supplied here.

Reuses the byte-pinned accepted CPU helper, without patching its globals or any
scientific module. Native observations always synchronize then copy to CPU.
"""
import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

HELPER_SHA = '65e7eb1a21b0e880ee2362461ff3664c4dbd52933816e5ec0684645c92cd75e8'
INPUTS_SHA = '0cc0569936b0b1653bcc946edcdedd1bf91d3f671fabbc205a0a7b883bf9a15e'
LABEL = 'fs-20260926-confirmed36-clean-native-consistency-trn21-r2'
CAP = 16 * 1024**2
SHAPES = ((1,256),(2,256),(3,256),(1,257),(2,257),(3,257))


class RankReceipt:
    """Durable latest progress; a final receipt is published once, exclusively.

    A killed writer can leave its private temporary file beside the last intact
    partial receipt. Final publication briefly adds a third name. Bounding each
    file to CAP//3 bounds all three names together, including the trailing LF.
    Nothing depends on a signal handler or Python finally running on timeout.
    """
    def __init__(self, h, directory, report, start):
        self.report, self.start, self.revision = report, start, 0
        self.directory = directory.absolute()
        h.require('..' not in self.directory.parts and
                  all(not part.is_symlink() for part in (self.directory,*self.directory.parents)), 'RECEIPT_DIRECTORY_SYMLINK')
        self.directory.mkdir(exist_ok=True)
        stem = 'rank'+str(report['rank'])
        self.partial = self.directory/(stem+'.partial.json')
        self.final = self.directory/(stem+'.json')
        self.prefix = '.'+stem+'.'
        self.finished = False
        h.require(not os.path.lexists(self.partial) and not os.path.lexists(self.final)
                  and not list(self.directory.glob(self.prefix+'*.tmp')), 'RECEIPT_ALREADY_EXISTS')
        self.checkpoint('starting')

    def sync_directory(self):
        fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def publish(self, final=False):
        if self.finished:
            raise RuntimeError('RECEIPT_ALREADY_FINAL')
        self.report['receipt_state'] = 'FINAL' if final else 'PARTIAL'
        self.report['receipt_revision'] = self.revision+1
        self.report['elapsed_seconds'] = time.monotonic()-self.start
        fd, name = tempfile.mkstemp(prefix=self.prefix, suffix='.tmp', dir=self.directory)
        temporary = Path(name)
        try:
            with os.fdopen(fd, 'wb') as stream:
                size = 0
                chunks = json.JSONEncoder(allow_nan=False).iterencode(self.report)
                for chunk in chunks:
                    data = chunk.encode('utf-8')
                    size += len(data)
                    if size+1 > CAP//3:
                        raise RuntimeError('RECEIPT_CAP')
                    stream.write(data)
                stream.write(b'\n')
                stream.flush()
                os.fsync(stream.fileno())
            if final:
                os.link(temporary, self.final)  # Exclusive: never replace a final receipt.
            elif self.revision == 0:
                os.link(temporary, self.partial)  # Reserve this rank exclusively.
            else:
                os.replace(temporary, self.partial)
            self.sync_directory()
            self.revision += 1
        finally:
            temporary.unlink(missing_ok=True)
        if final:
            # The final link is already durable. Keep the partial until every
            # publication operation above succeeds, so a failed final write
            # cannot look complete to the offline validator. Crash-resurrected
            # cleanup names only cause conservative refusal; no evidence loss.
            self.partial.unlink()
            self.finished = True

    def checkpoint(self, stage, **location):
        self.report['progress'] = {'stage':stage, **location}
        self.publish()

    def finish(self):
        self.publish(final=True)


def foundation():
    path = Path(__file__).parent.parent / '20260926-confirmed36-clean-checkpoint-cpu/compare_checkpoints.py'
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != HELPER_SHA:
        raise RuntimeError('CPU_HELPER_PIN')
    spec = importlib.util.spec_from_file_location('accepted_checkpoint_foundation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.no_symlink(path.absolute())
    return module


def snapshot(h, model, sync, masks=None, validated_masks=None):
    sync()
    h.check_modes_and_hooks(model)
    records = {}
    buffers = dict(model.named_buffers())
    if masks is not None:
        h.require({'mask_full','mask_short'} <= buffers.keys(), 'MISSING_ROOT_MASK_BUFFERS')
    for name, value in model.named_parameters():
        h.require(not value.requires_grad, 'PARAMETER_REQUIRES_GRAD:'+name)
        records['parameter:'+name] = h.tensor_record(value.detach().cpu(), name)
        records['parameter:'+name]['requires_grad'] = value.requires_grad
    for name, value in buffers.items():
        value_cpu = value.detach().cpu()
        key = 'buffer:'+name
        if masks is not None and name in ('mask_full','mask_short'):
            window = masks['full_window' if name == 'mask_full' else 'short_window']
            if validated_masks is None:
                h.mask_record(value_cpu, masks['sequence_len'], window, name)
            records[key] = h.raw_record(value_cpu)
            if validated_masks is not None:
                h.require(records[key] == validated_masks[key], 'MASK_CHANGED:'+name)
        else:
            records[key] = h.tensor_record(value_cpu, key)
        records[key]['requires_grad'] = value.requires_grad
    for name, module in model.named_modules():
        for key, value in vars(module).items():
            if isinstance(value, h.torch.Tensor):
                rec = h.tensor_record(value.detach().cpu(), name+'.'+key)
                rec['requires_grad'] = value.requires_grad
                records['attr:'+name+'.'+key] = rec
    return records


def exact(h, a, b, label):
    result = h.comparison(a, b)
    if a.shape == b.shape and a.dtype == b.dtype:
        unequal = a.contiguous().reshape(-1).view(h.torch.uint8) != b.contiguous().reshape(-1).view(h.torch.uint8)
        indices = unequal.nonzero()
        result['first_different_byte'] = int(indices[0]) if indices.numel() else None
    if not result['exact_bytes']:
        raise h.CheckFailure(label, result)
    return result


def verify_shape(h, model, x, vocab, device, sync, baseline, masks=None, *, case=None, progress=None):
    torch = h.torch
    h.require(x.dtype == torch.int64 and x.device.type == 'cpu' and x.ndim == 2, 'INPUT_TYPE')
    b, t = x.shape
    h.require((b,t) in SHAPES and bool(((x >= 1) & (x < vocab)).all()), 'INPUT_RANGE_SHAPE')
    initial = h.tensor_record(x)
    case = {} if case is None else case
    progress = progress if progress is not None else lambda *args, **kwargs: None
    case.update(shape=[b,t], tokens=x.tolist(), input=initial, calls=[], status='IN_PROGRESS')
    calls = case['calls']
    progress('case_started')
    def call(inp, label):
        case['current_call'] = label
        record = {'call':label,'status':'IN_PROGRESS'}
        calls.append(record)
        progress('call_started', call=label)
        try:
            h.require_state_equal(snapshot(h,model,sync,masks,baseline), baseline)
            host_before = h.tensor_record(inp)
            actual = inp.to(device).clone()
            sync()
            device_before = h.tensor_record(actual.detach().cpu())
            record.update(input=host_before, device_input=device_before)
            progress('call_ready', call=label)
            with torch.no_grad():
                out = model(actual)
            sync()
            h.require(isinstance(out,torch.Tensor) and list(out.shape) == [len(inp),t,vocab], 'OUTPUT_SHAPE')
            # Clone before the next forward, even if cpu() returned the same storage.
            copied = out.detach().cpu().clone()
            record['output'] = h.tensor_record(copied)
            progress('call_output', call=label)
            h.require(h.tensor_record(actual.detach().cpu()) == device_before, 'DEVICE_INPUT_CHANGED')
            h.require(h.tensor_record(inp) == host_before and h.tensor_record(x) == initial, 'CPU_INPUT_CHANGED')
            h.require_state_equal(snapshot(h,model,sync,masks,baseline), baseline)
            record['status'] = 'COMPLETE'
            progress('call_complete', call=label)
            return copied
        except Exception as error:
            record.update(status='REFUSE',error=str(error),details=getattr(error,'details',None))
            raise
    try:
        original = call(x,'original')
        case['repeat'] = exact(h,call(x,'repeat'),original,'REPEAT_BYTES')
        rows = torch.cat([call(x[i:i+1].clone(),'row'+str(i)) for i in range(b)])
        case['rowwise'] = exact(h,rows,original,'ROWWISE_BYTES')
        cut = max(2,t//2)
        suffix = x.clone()
        suffix[:,cut:] = suffix[:,cut:].flip(1)
        h.require(all(not torch.equal(x[i,cut:],suffix[i,cut:]) for i in range(b)), 'DEGENERATE_SUFFIX_FIXTURE')
        altered = call(suffix,'reversed_suffix')
        left, right = original[:,:cut-1].float(), altered[:,:cut-1].float()
        delta = (left-right).abs()
        error = float(delta.max())
        finite = math.isfinite(error)
        # Keep the original FP32 subtraction/max and threshold as the causal gate.
        # Extra CPU-only diagnostics do not turn this into a byte-equality gate.
        maximum = int(delta.reshape(-1).argmax())
        metrics = h.comparison(right,left) if bool(torch.isfinite(left).all() and torch.isfinite(right).all()) else None
        causal = {'shape':[b,t], 'max_abs_diff':error if finite else str(error),
            'finite':finite, 'tolerance':.01, 'cut':cut, 'prefix_start':0, 'prefix_stop':cut-1,
            'prefix_exact':h.raw_record(left)==h.raw_record(right),
            'expected':h.raw_record(left), 'actual':h.raw_record(right),
            'relative_rms':metrics['relative_rms'] if metrics else 'UNDEFINED_NONFINITE_PREFIX',
            'max_abs_index':[maximum//((cut-1)*vocab),(maximum//vocab)%(cut-1),maximum%vocab],
            'original_input':initial, 'altered_input':calls[-1]['input']}
        case['causality'] = causal
        progress('causality_measured', call='reversed_suffix')
        if not (finite and error <= .01):
            raise h.CheckFailure('CAUSAL_PREFIX',causal)
        case['status'] = 'COMPLETE'
        progress('case_complete', call='reversed_suffix')
        return original, case
    except Exception as error:
        case.update(status='REFUSE',error=str(error),details=getattr(error,'details',None))
        progress('case_refused', call=case.get('current_call'))
        raise


def validate_cases(h, cases):
    h.require([c['shape'] for c in cases] == [list(s) for s in SHAPES], 'EXACT_SIX_SHAPES')
    for case in cases:
        b,t = case['shape']
        h.require(case.get('status')=='COMPLETE', 'CASE_COMPLETE')
        h.require([c['call'] for c in case['calls']] == ['original','repeat'] + ['row'+str(i) for i in range(b)] + ['reversed_suffix'], 'EXACT_CALL_SET')
        for c in case['calls']:
            h.require(c.get('status')=='COMPLETE', 'CALL_COMPLETE')
            expected_b = 1 if c['call'].startswith('row') else b
            h.require(c['output']['shape'] == [expected_b,t,8192], 'CALL_OUTPUT_SHAPE')


def signatures(cases):
    return [{'shape':c['shape'],'input':c['input'],'calls':c['calls']} for c in cases]


def validate_ranks(h, ranks):
    h.require(len(ranks)==4 and all(type(r['rank']) is int for r in ranks) and {r['rank'] for r in ranks}==set(range(4)), 'EXACT_FOUR_RANKS')
    for r in ranks:
        h.require(r.get('status') == 'LOCAL_COMPLETE' and set(r['arms'])=={'reference','candidate'}, 'RANK_COMPLETE')
        for arm in r['arms'].values():
            validate_cases(h,arm['shapes'])
        h.require(signatures(r['arms']['reference']['shapes']) == signatures(r['arms']['candidate']['shapes']), 'CROSS_SOURCE_CALL_HASHES')
    first = ranks[0]['arms']
    for r in ranks[1:]:
        h.require(r['arms']==first, 'CROSS_RANK_RECORDS')


def restored_route(h, name, model, module, payload, device):
    h.require(device.type=='neuron' and module._HAS_NKI is True, 'NATIVE_NKI_REQUIRED')
    h.require(module.COMPUTE_DTYPE == h.torch.bfloat16, 'ORDINARY_COMPUTE_DTYPE')
    h.require(module.EVAL_MIN_SEQ_LEN==8192 and model.config.sequence_len==8192, 'EVAL_FLOOR')
    blocks = model.transformer.h
    h.require(len(blocks)==8 and model.config.n_embd==1024 and model.config.vocab_size==8192, 'LOADED_ARCHITECTURE')
    h.require([i for i,b in enumerate(blocks) if b.attn is not None]==[0,2,4,6,7], 'ATTENTION_LAYERS')
    h.require([i for i,b in enumerate(blocks) if b.local_conv is not None]==[0,2,4,6,7], 'CONV_LAYERS')
    h.require(all(b.local_conv.kernel_size==2 and b.local_conv.n_embd==1024 for b in blocks if b.local_conv is not None), 'CONV_NATIVE_DIMENSIONS')
    h.require(all(p.device.type=='neuron' for p in model.parameters()), 'FULL_NATIVE_MODEL')
    conv_source = inspect.getsource(module.CausalConv.forward)
    lane_source = inspect.getsource(module._Conv2NKI.forward)
    h.require('x.shape[0] % 2 == 0' in lane_source, 'CONV_LANE_PREDICATE')
    h.require(len(model.unet_skip_gates)==4 and all(p.numel()==1 for p in model.unet_skip_gates), 'SCALAR_SKIPS')
    h.require(list(model.value_embeds.keys())==['7'], 'VE_LAYER')
    flags = {}
    if name=='reference':
        h.require(type(model) is module._SubmissionEvalGPT, 'ORDINARY_REFERENCE_CLASS')
        expected = {'USE_UNET_SKIP':True,'USE_PARTIAL_ATTN_8':True,'USE_QK_SHIFT':False,
            'USE_NKI_LOCAL_CONV':True,'USE_BF16_NORM_OUTPUT':True,'COMPILE_SDPA_DIRECT':True,'SHORT_WINDOW_FRAC':.5}
        for key, value in expected.items():
            h.require(key in payload.get('toggles',{}) and payload['toggles'][key]==value and getattr(module,key)==value, 'RESTORED_TOGGLE:'+key)
            flags[key] = getattr(module,key)
    masks = h.mask_configuration(name,model,module,payload)
    if masks is not None:
        h.require(masks['short_window']==4096, 'SHORT_WINDOW')
    return masks, {'device':str(device),'has_nki':module._HAS_NKI,'compute_dtype':str(module.COMPUTE_DTYPE),
        'restored_flags':flags,'loader_class':type(model).__name__,
        'conv_forward_source':conv_source,'conv_lane_source':lane_source,
        'predicate_coverage':[{'shape':[b,t],'base_batch':1 if name=='reference' else b,
            'conv_native_predicate':t%128==0,'conv_lanes':(2 if b%2==0 else 1) if name=='candidate' and t%128==0 else (1 if t%128==0 else None),
            'rope_length_predicate':t%128==0} for b,t in SHAPES],
        'coverage_limit':'Pinned source predicates and actual restored native configuration; not instrumented kernel execution proof.'}


def run(h, args, report, progress):
    torch = h.torch
    import torch.distributed as dist
    h.require(args.inputs_sha256 == INPUTS_SHA, 'FIXED_INPUTS_PIN')
    identity = h.file_record(args.inputs)
    h.require(identity['sha256']==INPUTS_SHA, 'INPUTS_SHA')
    inputs = h.read_json(args.inputs)
    files = h.validate_manifest(inputs)
    try:
        h.authenticate_files(files)
        report['pair'] = h.qualify_pair(inputs)
        saved = {}
        # Both complete saved states qualify before importing/initializing native.
        for name in h.SOURCES:
            payload = torch.load(inputs['arms'][name]['checkpoint_path'],map_location='cpu',weights_only=True)
            saved[name] = h.full_payload(payload)
            del payload
        h.require(saved['reference']==saved['candidate'], 'SAVED_STATE_BYTES')
        report['saved_parameters'] = saved['reference']
        progress('saved_states_qualified')
        import prepare
        h.require(Path(prepare.__file__).absolute()==h.PREPARE and h.file_record(h.PREPARE)['sha256']==h.PREPARE_SHA, 'ACTUAL_PREPARE')
        h.require(os.environ.get('WORLD_SIZE')=='4' and os.environ.get('NEURON_LOGICAL_NC_CONFIG')=='2', 'FOUR_RANK_LNC2')
        h.require('--optlevel=1' in os.environ.get('NEURON_CC_FLAGS','').split(), 'OPTLEVEL1')
        import torch_neuronx
        dist.init_process_group('neuron')
        h.require(dist.get_world_size()==4 and dist.get_rank()==report['rank'], 'PROCESS_GROUP')
        h.require(callable(getattr(getattr(torch,'neuron',None),'synchronize',None)), 'SYNC_REQUIRED')
        sync = torch.neuron.synchronize
        device = torch.device('neuron')
        originals = {}
        for name in h.SOURCES:
            arm = {'status':'LOADING','shapes':[]}
            report['arms'][name] = arm
            progress('arm_loading', arm=name)
            checkpoint = inputs['arms'][name]['checkpoint_path']
            payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
            module = h.load_module(name)
            model = module.load_for_eval(checkpoint,device)
            sync()
            actual = model.state_dict()
            h.require(actual.keys()==payload['model_state'].keys(), 'LOADED_KEYS')
            for key, tensor in actual.items():
                exact(h,tensor.detach().cpu(),payload['model_state'][key], 'LOADED_BYTES:'+key)
            del actual, tensor
            masks, route = restored_route(h,name,model,module,payload,device)
            baseline = snapshot(h,model,sync,masks)
            arm.update(route=route,state=baseline,status='IN_PROGRESS')
            progress('arm_ready', arm=name)
            for b,t in SHAPES:
                x = torch.randint(1,8192,(b,t),generator=torch.Generator().manual_seed(42))
                case = {'shape':[b,t],'status':'IN_PROGRESS','calls':[]}
                arm['shapes'].append(case)
                progress('case_preparing', arm=name, shape=[b,t])
                output, case = verify_shape(h,model,x,8192,device,sync,baseline,masks,case=case,
                    progress=lambda stage, **location: progress(stage,arm=name,shape=[b,t],**location))
                if name=='reference':
                    originals[b,t] = output
                else:
                    exact(h,output,originals.pop((b,t)), 'CROSS_SOURCE_ORIGINAL')
                progress('case_checked', arm=name, shape=[b,t])
            validate_cases(h,arm['shapes'])
            arm['status'] = 'COMPLETE'
            progress('arm_teardown', arm=name)
            del model, module, payload, output, baseline, x
            gc.collect()
            sync()
            progress('arm_released', arm=name)
        h.require(not originals, 'ALL_ORIGINALS_COMPARED')
        h.require(signatures(report['arms']['reference']['shapes'])==signatures(report['arms']['candidate']['shapes']), 'CROSS_SOURCE_CALL_HASHES')
    finally:
        # Attempt each immutable pin even when another one fails.
        failures = []
        for path,pin in files.items():
            try:
                h.require(h.file_record(Path(path))==pin, 'INPUT_CHANGED:'+path)
            except Exception as error:
                failures.append(str(error))
        try:
            h.require(h.file_record(args.inputs)==identity, 'MANIFEST_CHANGED')
        except Exception as error:
            failures.append(str(error))
        report['post_input_failures'] = failures
        progress('immutable_inputs_rechecked',last_progress=report.get('progress'))
        h.require(not failures, 'IMMUTABLE_INPUT_RECHECK_FAILED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enable-native',action='store_true')
    parser.add_argument('--label')
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--inputs-sha256')
    parser.add_argument('--out-dir',type=Path)
    parser.add_argument('--validate-receipts',type=Path,help='Offline validation only; never initializes native')
    args = parser.parse_args()
    if args.validate_receipts is not None:
        h = foundation()
        h.require(not args.enable_native, 'NO_COMBINED_VALIDATE_NATIVE')
        paths = sorted(args.validate_receipts.glob('rank*.json'))
        h.require([path.name for path in paths]==['rank0.json','rank1.json','rank2.json','rank3.json'], 'EXACT_RANK_FILES')
        rows = []
        for path in paths:
            h.no_symlink(path.absolute())
            h.require(path.stat().st_size<=CAP, 'RECEIPT_CAP')
            rows.append(json.loads(path.read_text()))
        p = {r['rank']:r for r in rows}
        h.require(len(p)==4 and all(p[i]['label']==LABEL and p[i].get('receipt_state')=='FINAL' and p[i].get('post_input_failures')==[] and 'cleanup_error' not in p[i] for i in range(4)), 'TERMINAL_RECEIPTS')
        validate_ranks(h,rows)
        print('{"status":"FOUR_RANK_SYNTHETIC_RECORDS_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE","quality_evidence":false}')
        return 0
    if not args.enable_native:
        print('{"status":"DISABLED","model_imported":false}')
        return 0
    h = foundation()
    h.require(args.label==LABEL and args.inputs is not None and args.out_dir is not None, 'EXACT_BINDING_REQUIRED')
    h.require(os.environ.get('RANK') in ('0','1','2','3'), 'RANK_REQUIRED')
    rank = int(os.environ['RANK'])
    report = {'rank':rank,'label':LABEL,'status':'UNRESOLVED','arms':{},'launch_authorized':False,
        'quality_evidence':False,'native_result_independently_accepted':False}
    start = time.monotonic()
    receipt = RankReceipt(h,args.out_dir,report,start)
    completed = False
    try:
        h.torch.set_num_threads(1)
        h.torch.set_num_interop_threads(1)
        run(h,args,report,receipt.checkpoint)
        completed = True
    except Exception as error:
        report.update(status='REFUSE',error=str(error),details=getattr(error,'details',None))
    finally:
        # Persist failures before potentially blocking process-group teardown.
        try:
            receipt.checkpoint('process_group_cleanup', last_progress=report.get('progress'))
        except Exception as error:
            report.update(status='UNRESOLVED',receipt_error=str(error))
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception as error:
            report.update(status='UNRESOLVED',cleanup_error=str(error))
        if completed and 'cleanup_error' not in report and 'receipt_error' not in report:
            report['status'] = 'LOCAL_COMPLETE'
        try:
            receipt.finish()
        except Exception as error:
            # Keep the last intact partial if the new record cannot be published.
            report['status'] = 'UNRESOLVED'
            print('RECEIPT_FINAL_WRITE_FAILED:'+str(error)[:1024],file=sys.stderr)
    # A separate offline validator must check all four complete rank receipts.
    return 0 if report['status']=='LOCAL_COMPLETE' else 2


if __name__=='__main__':
    raise SystemExit(main())
