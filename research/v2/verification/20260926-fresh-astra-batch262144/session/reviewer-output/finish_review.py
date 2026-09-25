"""Finish independent receipt checks, summarize executed tests, and emit verdict."""
import hashlib
import json
import math
import re
from pathlib import Path

OUT=Path('/work')
def read(name): return json.loads((OUT/name).read_text())
audit=read('receipt_audit.json')
tests=read('reference_results.json')
checks={}
for name,arm in audit['arms'].items():
    gas=2 if name=='batch262144' else 1
    log=(Path('/packet/raw')/name/'torchrun.log').read_text()
    smooth=0.0
    expected=[]
    for step in range(3):
        smooth=0.9*smooth+0.1*arm['losses'][0][(step+1)*gas-1]
        expected.append(f'{smooth/(1-0.9**(step+1)):.4f}')
    actual=re.findall(r'^step \d+ \| loss ([\d.]+) ',log,re.M)
    owners=[]
    for line in log.splitlines():
        if line.startswith('SUBMISSION_OPTIMIZER_INTEGRATION '):
            owners.append(json.loads(line.split(' ',1)[1]))
    checks[name]={
        'embedded_ranks_equal_files':all(arm['embedded_ranks_equal_files']),
        'loss_count_matches_accumulation':all(len(row)==3*gas for row in arm['losses']),
        'losses_finite':all(math.isfinite(x) for row in arm['losses'] for x in row),
        'rank0_logged_smooth_last_micro_loss_matches':actual==expected,
        'expected_log_losses':expected,'actual_log_losses':actual,
        'owner_records_all_ranks':sorted(o['rank'] for o in owners)==list(range(4)),
        'owner_summaries_match_raw_ranks':all(all(o[k]==v for k,v in arm['plan_summary'].items()) for o in owners),
        'owner_element_sum':arm['parameter_count_sum']==117453580,
        'owner_padding':arm['plan_summary']['total_padded_bytes']==arm['plan_summary']['shard_elements']*4*4,
        'recorded_success':arm['torchrun_rc']==0 and not arm['timed_out'],
        'log_tail_equal':arm['recorded_log_tail_is_actual_tail'],
        'digest_count_and_width':len(arm['final_digests'])==62 and all(re.fullmatch('[0-9a-f]{16}',v) for v in arm['final_digests'].values()),
        'rank0_checkpoint_fields':arm['checkpoint_check'],
    }
    assert all(v for v in checks[name].values() if isinstance(v,bool)),checks[name]
assert audit['arms']['reference']['argv']==audit['arms']['default']['argv']
assert audit['arms']['batch262144']['argv']==audit['arms']['default']['argv']+['--total-batch-size','262144']
assert audit['default_loss_float_hex_equality_all_ranks']
assert audit['default_rank0_digest_map_exact_equality']
assert audit['all_plan_hashes_and_summaries_equal']
assert tests['verdict']=='PASS' and not tests['failures']
assert all(c['passed'] for c in tests['checks'])
assert read('reference_test_run1.command.json')['exit_code']==0
assert read('packet_audit.command.json')['exit_code']==0
errors={}
for dtype in ('float64','float32'):
    cases=[c for c in tests['cases'] if c['dtype']=='torch.'+dtype]
    errors[dtype]={key:max(s[key]['max_abs'] for c in cases for s in c['comparisons']) for key in ('loss','gradient')}
    for key in ('clip_error','weight_error'):
        errors[dtype][key]=max(c[key]['max_abs'] for c in tests['checks'] if key in c and c[key]['atol']==tests['criteria'][dtype]['atol'])
(OUT/'receipt_crosschecks.json').write_text(json.dumps(checks,indent=2,sort_keys=True)+'\n')
result={
    'verdict':'PASS','scope':'source/CPU-only review of total-batch CLI in the supplied packet',
    'default_was_REFUSE_until_required_evidence_complete':True,
    'independence':'single fresh reviewer; no other agents, supplied test code, history, repository, network, or native execution',
    'base_commit_metadata':'a4aa4351789115a2d0aa947d9ca00ff689f9bd0b',
    'candidate_commit_metadata':'4e361d15baef94f041501f558620422c55ae8c88',
    'commit_associations_independently_verified':False,
    'source_hashes':read('source_hashes.json'),
    'intended_tokens':{'device_batch':1,'pack':32,'context':1024,'world':4,'per_micro_global':131072,
                       'default_accumulation':1,'experimental_accumulation':2,'default':131072,'experimental':262144},
    'test':{'source':'reference_test.py','results':'reference_results.json','checks':len(tests['checks']),
            'cases':len(tests['cases']),'exit_code':0,'criteria':tests['criteria'],'max_absolute_errors':errors,
            'settings':tests['settings'],'bounds':tests['bounds'],
            'exact_claim':'identical full recorded toy histories (tensor values and SHA256 of raw tensor bytes) across base, candidate omitted flag, explicit 131072 in FP32 and FP64; no tolerance',
            'mocks':['serial four-rank deterministic AVG','toy embedding/tanh/linear model','SGD with momentum instead of MuonAdamW/OwnedOptimizer','no-op debug hook'],
            'omitted':['module main','full production model','real process group','NKI/BF16/native compilation','dataset/tokenizer','clock/timing','checkpoint binary/save-load execution','dynamic overflow','batch ramp ON','deferred Adam and flat-buffer branches']},
    'receipt_evidence':{'source':'receipt_audit.json','crosschecks':'receipt_crosschecks.json',
                        'default_losses_exact':True,'rank0_equal_parameter_digest_count':62,
                        'all_owner_plan_summaries_equal':True,'changed_arm_different_parameter_digest_count':62,
                        'digest_limitation':'only 16 hex characters per final tensor; only rank0 supplies the maps; no initial tensors or checkpoint binaries'},
    'findings':[
        {'severity':'informational','location':'candidate.py:3616-3620','finding':'Positive integer CLI preserves default; fractional, malformed, missing, zero and negative values reject.'},
        {'severity':'low-existing','location':'candidate.py:3833-3872','finding':'Divisibility rejects after runtime/model setup; error wording reverses divides/divisible-by. Predicate is correct; neither location nor message was introduced by patch.'},
        {'severity':'low-existing','location':'candidate.py:3983,4050,4075-4082','finding':'Logged loss is smoothed rank0 last-microbatch loss, not full-update/global mean. Gradient uses every microbatch.'},
        {'severity':'scope-existing','location':'candidate.py:3284-3297','finding':'Explicit batch ramp overrides accumulation and can scale LR; no-scaling conclusion requires intended batch-ramp OFF.'},
        {'severity':'scope-existing','location':'candidate.py:1395-1428','finding':'Optional byte-weight loss averages separately normalized microbatches, not a single global byte-weighted ratio; intended ordinary-token mean unaffected.'},
        {'severity':'scope-existing','location':'candidate.py:3158-3208,4099','finding':'Weight-only checkpoint omits total batch and optimizer/RNG/loader state; load is warm start, not exact resume. Patch preserves schema.'}
    ],
    'blocking_findings_in_assigned_scope':[],
    'refusals':[
        {'verdict':'REFUSE','claim':'native correctness, startup, compilation, throughput or >=5% speedup','missing_evidence':'actual native production-microshape diagnostic; CPU receipts slice 2x256 and fake time'},
        {'verdict':'REFUSE','claim':'quality, scientific adoption, changed default, full run, staging, hardware launch or merge authorization','missing_evidence':'separate user assignment/authority and applicable native/accounting/quality evidence'},
        {'verdict':'REFUSE','claim':'independent verification of supplied checkpoints and cross-rank tensor identity','missing_evidence':'checkpoint binaries, non-rank0 digest maps and supplied harness are absent'}
    ],
    'warnings_preserved':['PyTorch emitted: Error in cpuinfo: failed to parse processor information from /proc/cpuinfo; test exit 0, all checks passed.'],
    'artifacts':{'report':'REVIEW.md','manifest':'artifact_manifest.json','input_hashes':'input_hashes.json','test_source':'reference_test.py','raw_stdout':'reference_test_run1.stdout.txt','raw_stderr':'reference_test_run1.stderr.txt','command':'reference_test_run1.command.json'}
}
(OUT/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'verdict':result['verdict'],'scope':result['scope'],'errors':errors,'receipt_crosschecks':checks},indent=2))
