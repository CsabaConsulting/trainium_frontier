"""Implementer self-check, explicitly reusing the old review's tiny CPU probe.

This is NOT an independent review. No real data, distributed run or native kernel.
The reused probe enforces <=64 MiB live tensor storage, D128/V256, one thread.
Test-only import stubs and nonpersistent eval floor32 are fully disclosed there.
"""
import ast
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD_TEST = ROOT / 'research/v2/verification/20260926-fresh-astra-architecture-clean/session/reviewer-output/independent_test.py'
REFERENCE = subprocess.check_output(['git', 'show', '7419927c7417b5666a984ec9c2b849feadabaaa9:workflow/proposals/research-v2/train.py'], cwd=ROOT).decode()
CANDIDATE = (HERE / 'train.py').read_text()
VEHICLE = (ROOT / '05-train-submission-clean-116188.py').read_text()
assert hashlib.sha256(REFERENCE.encode()).hexdigest() == '36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb'
assert hashlib.sha256(VEHICLE.encode()).hexdigest() == '116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d'
assert hashlib.sha256((HERE / 'launch-command.txt').read_bytes()).hexdigest() == 'b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd'
def definition(text, name):
    return next(n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
def tree(text, name):
    return ast.dump(definition(text, name), include_attributes=False)
assert tree(CANDIDATE, 'startup_adjusted_schedule_elapsed') == tree(REFERENCE, 'startup_adjusted_schedule_elapsed')
assert tree(CANDIDATE, 'load_for_eval') == tree(VEHICLE, 'load_for_eval')
assert '_SubmissionEvalGPT' not in CANDIDATE
# Exact scope relative to the preserved old draft: helper + two consumers only.
old = (ROOT / 'workflow/candidates/architecture3e79-116188/train.py').read_text()
helper = ast.get_source_segment(CANDIDATE, definition(CANDIDATE, 'startup_adjusted_schedule_elapsed'))
reverted = CANDIDATE.replace(helper + '\n\n', '').replace('        schedule_elapsed = startup_adjusted_schedule_elapsed(step, elapsed, budget_started)\n', '').replace('elapsed=schedule_elapsed', 'elapsed=elapsed').replace('schedule_elapsed / max', 'elapsed / max')
assert reverted == old
# Independent algebraic oracle with exact operation order across startup boundary.
ns = dict(TRAIN_STARTUP_STEPS_EXCLUDED=5, TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS=900, STARTUP_LAUNCH_MARGIN_SECONDS=30.0, _PROCESS_STARTED=100.0)
exec(compile(ast.Module(body=[definition(CANDIDATE, 'startup_adjusted_schedule_elapsed')], type_ignores=[]), '<clock>', 'exec'), ns)
count = 0
for step in range(7):
    for started in [100., 103., 869., 970., 1000., 1700.]:
        for elapsed in [0., 1e-9, 5., 29.999999, 30., 30.000001, 450., 870., 900., 1799.]:
            expected = elapsed if step >= 5 else max(0., elapsed + (started - 100.) + 30. - 900.)
            assert ns['startup_adjusted_schedule_elapsed'](step, elapsed, started) == expected
            count += 1
assert ns['startup_adjusted_schedule_elapsed'](0, 900., 100.) != 900.  # wrong-clock control
print('Source/command/ordinary-loader/scope and', count, 'clock cases PASS', flush=True)

# Reuse prior bounded numerical probe as Implementer tests, with explicit changes:
# actual confirmed36 source; actual organizer 5/900 constants; three prescribed
# step-start samples routed through helper; temp artifact directory. No production
# source transformations; the prior probe's stated CPU-only substitutions remain.
output = Path(tempfile.mkdtemp(prefix='confirmed36-tiny-'))
probe = OLD_TEST.read_text()
probe = probe.replace("W=Path('/work')", 'W=Path(' + repr(str(output)) + ')')
probe = probe.replace('TRAIN_STARTUP_STEPS_EXCLUDED=2,TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS=600', 'TRAIN_STARTUP_STEPS_EXCLUDED=5,TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS=900')
probe = probe.replace("--initial-cache-refresh'.split()", "--initial-cache-refresh --startup-adjusted-schedule'.split()")
probe = probe.replace("spec.loader.exec_module(m)", "exec(compile(REFERENCE if name=='scored_reference' else CANDIDATE, str(HERE/'train.py'), 'exec'), m.__dict__)")
probe = probe.replace("W/(name+'.py')", "HERE/'train.py'")
probe = probe.replace("  elapsed=[0.,900.,1764.][step]", "  m._PROCESS_STARTED=100.\n  elapsed=m.startup_adjusted_schedule_elapsed(step,[0.,900.,1764.][step],103.)")
# Retain optimizer states and cache tensors for all three steps too.
probe = probe.replace("  save(name+f'_step{step}_after_update',model.state_dict())", "  save(name+f'_step{step}_after_update',model.state_dict())\n  save(name+f'_step{step}_optimizer',opt.state_dict())\n  save(name+f'_step{step}_cache',{n:x._cached_weight for n,x in model.named_modules() if isinstance(x,m.Linear)})")
probe = probe.replace(' sys.exit(bool(failures))', " assert not failures, failures")
exec(compile(probe, str(OLD_TEST), 'exec'), globals())
# Compare all retained optimizer and cache tensor trees; no digest-only shortcut.
def equal_tree(a,b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal_tree(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal_tree(x,y)
    else: assert a==b
for step in range(3):
    for kind in ['optimizer','cache']:
        equal_tree(torch.load(A/f'candidate_step{step}_{kind}.pt',weights_only=True),torch.load(A/f'scored_reference_step{step}_{kind}.pt',weights_only=True))
print('All 3 optimizer/cache trees exactly equal. Artifacts:', output, flush=True)
# Native-eligible convolution sequence length and odd/even arithmetic batches,
# still CPU fallback only. Ordinary trained GPT and ordinary loaded candidate
# are both checked; research loader row splitting remains explicitly visible.
batch_outputs = {}
for name in ['candidate', 'scored_reference']:
    m = load(name)
    m.COMPUTE_DTYPE = torch.bfloat16
    m.EVAL_MIN_SEQ_LEN = 256
    loaded = m.load_for_eval(str(A/f'{name}_trained.pt'), torch.device('cpu'))
    routes = []
    handle = loaded.transformer.h[0].register_forward_pre_hook(lambda module,args: routes.append(list(args[0].shape)))
    with torch.no_grad():
        for batch in [1,2,3]:
            tokens = (torch.arange(batch*256).reshape(batch,256)*13+11)%256
            routes.clear()
            logits = loaded(tokens)
            batch_outputs[name,batch] = logits.clone()
            record(f'{name}_eligible_conv_batch{batch}_route', passed=True, shapes=list(routes), native_executed=False)
            rows = torch.cat([loaded(row[None]) for row in tokens])
            compare(f'{name}_eligible_conv_batch{batch}_rows',logits,rows)
            changed=tokens.clone();changed[:,128:]=(changed[:,128:]+17)%256
            compare(f'{name}_eligible_conv_batch{batch}_causal', logits[:,:128], loaded(changed)[:,:128])
    handle.remove()
    if name=='candidate':
        trained=m.GPT(m.GPTConfig(sequence_len=256,vocab_size=256,n_layer=8,n_head=1,n_embd=128))
        trained.init_weights()
        trained.load_state_dict(torch.load(A/'candidate_trained.pt',weights_only=True)['model_state'])
        refresh(m,trained)
        with torch.no_grad():
            for batch in [1,2,3]:
                tokens=(torch.arange(batch*256).reshape(batch,256)*13+11)%256
                compare(f'ordinary_train_vs_loaded_batch{batch}',trained(tokens),batch_outputs[name,batch])
        del trained
    del loaded,m
    gc.collect()
for batch in [1,2,3]:
    compare(f'eligible_conv_reference_batch{batch}',batch_outputs['candidate',batch],batch_outputs['scored_reference',batch])
assert all(r['passed'] for r in RESULTS)
print('Eligible-context CPU B1/B2/B3, train/load and causality PASS; native pending.',flush=True)
