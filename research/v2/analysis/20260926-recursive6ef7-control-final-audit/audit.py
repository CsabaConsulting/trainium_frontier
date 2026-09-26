"""Exact one-control archive audit; no arguments means PENDING, never a score."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

REPO = Path(__file__).resolve().parents[4]
P = 'research/v2/proposals/20260926-recursive6ef7-control-trn22-r2/'
OP = 'research/v2/operator/20260926-trn22-recursive6ef7-control/'
PACKET = '9889c4e07de37fd4c0c84c2efc4a014e9ff82b26'
LABEL = 'fs-20260926-recursive6ef7-control-trn22-r2'
SOURCE = '6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3'
KNOWN_TRAIN_PIDS = {3218053,3218054,3218055,3218072,3218073,3218087,3218102,3218103,3218104,3218105}
KNOWN_COMPILER_PIDS = {3218647,3218649,3218671}
KNOWN_PIDS = KNOWN_TRAIN_PIDS | KNOWN_COMPILER_PIDS
def git(ref,path):
    return subprocess.check_output(['git','-C',str(REPO),'show',ref+':'+path])
def sha(raw):return hashlib.sha256(raw).hexdigest()

# Reuse reviewed parsing/accounting/postflight machinery verbatim, pinned to Git.
# Its main/pins/selector paths are NOT invoked; it has no host or file-write API.
helper = git('ae725297','research/v2/analysis/20260926-basebcbd-final-audit-preparation/audit.py')
if sha(helper) != '1ec138987f32d050448d2318702f06632f6ca8e1d531c9c3e1142812730dc770':
    raise RuntimeError('HELPER_PIN_MISMATCH')
ns = {'__name__':'immutable_archive_helper'}
exec(compile(helper,'ae725297:archive-audit.py','exec'),ns)
checks = ns['checks']
require,parse,decode,closed_result,timing,evaluation = [ns[n] for n in ('require','parse','decode','closed_result','timing','evaluation')]

def intended(b):
    require(b['label']==LABEL and b['host']=='trn22' and b['source_sha256']==SOURCE,'exact intended source/host/label')
    require(b['kind']=='control' and b['base_sha256']==SOURCE,'genuine control/base6ef7 only')
    require(b['source_commit']=='a4aa4351789115a2d0aa947d9ca00ff689f9bd0b','exact source commit')
    cmd=b['training_command']
    require('--recursive-schedule' in cmd and all(x not in cmd for x in ('--initial-cache-refresh','--startup-adjusted-schedule','--unet-8x1024')),'no extra scientific factor')
    require(b['training_environment']['NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS']=='1800','fixed1800 declaration')
    require(b['lock_identity']['expected_inode']==832589 and b['lock_identity']['expected_device']==66305,'original lock832589')

def pins():
    raw=git(PACKET,P+'NATIVE_BINDING.disabled.json')
    require(sha(raw)=='dab176a8424bc750e7352877cf0c2703d855fd36b1e96b3af0bdd935446be77a','frozen control binding')
    b=parse(raw);intended(b)
    require(sha(git('cac7cfdc','research/v2/verification/20260926-recursive6ef7-control-trn22-assignment.md'))=='9bdc6e5308d9d2e518da10525bde2dba89447b7fb4a585d0fe6d8c08396c47e1','exact scoped assignment')
    require(sha(git(b['source_commit'],b['source_git_path']))==SOURCE,'scientific source Git bytes')
    require(sha(git(PACKET,P+'host_driver.py'))==b['driver_sha256'],'driver bytes')
    require(sha(git(PACKET,'prepare.py'))==b['prepare_sha256'] and sha(git('00f2ca7f','research/scratch/submission_rehearsal.py'))==b['harness_sha256'],'prepare and observer unchanged')
    for name,key in [('idle_guard.py','idle_guard_sha256'),('dispatch_driver.py','dispatch_driver_sha256')]:
        require(sha(git(PACKET,'research/v2/operator/20260926-common-host-idle-guard/'+name))==b[key],'shared support '+name)
    manifest=parse(git(PACKET,P+'MANIFEST.sha256.json'))
    for name,pin in manifest['files'].items():require(sha(git(PACKET,P+name))==pin,'packet manifest '+name)
    live,lf=decode(git('2397f0aa',OP+'launch/first-native-evidence.json'),'044f44063ca2ae7eb053ed097fbd2f5aca1eefdc05e8837a5b26c465f1e308ba')
    five,ff=decode(git('fc24dc8d',OP+'progress/first-five-raw.json'),'4a4ae47fc4196dd73eb54e524a74ef4fc7edb4543cd96eb979b3cd7762de61f4')
    for observed,files in [(live,lf),(five,ff)]:
        require(observed['root']==b['root'] and files['NATIVE_BINDING.disabled.json']==raw,'actual launch/progress binding')
        require({int(x.split()[0]) for x in observed['job_lines']}==KNOWN_TRAIN_PIDS,'observed ten training PIDs')
        require(len(observed['workers'])==4 and {w['environment']['RANK'] for w in observed['workers']}=={'0','1','2','3'},'four actual workers')
    raw_comp=git('5d91a0d2',OP+'progress/0045-readonly.json')
    require(sha(raw_comp)=='6a65a461c4ac1c9df15d33ce6e9bd2bcd1f72f855f840b67d85874011893dcc1','actual compiler observation')
    comp=parse(raw_comp)
    require({int(x.split()[0]) for x in comp['job_lines']}==KNOWN_PIDS,'all13 observed launcher/worker/compiler identities')
    require(comp['parsed']['training-launch.json']==parse(lf['training-launch.json']),'compiler observation same bound run')
    require(len(ff['train/steps.jsonl'].splitlines())>=5,'first-five archive present, not final result')
    return b,ff

def known_postflight(post):
    require(KNOWN_PIDS <= set(post['observed_pids']),'all known compiler/worker PIDs in final explicit census')
    require(not any(re.search('neuronx?-cc|neuronx-compile|walrus_driver',r['comm']) for r in post.get('all_process_identity_rows',[])),'no observed compiler family including walrus in final census')

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-commit');ap.add_argument('--bundle-sha');a=ap.parse_args(argv)
    b,first_five=pins()
    if a.raw_commit is None and a.bundle_sha is None:
        return dict(status='ONLY_PENDING_REAL_RAW',label=LABEL,source_sha256=SOURCE,kind='control',base_sha256=SOURCE,public_20m=None,selector_verdict=None,known_observed_pids=sorted(KNOWN_PIDS),required=['immutable final40hex commit and exact64hex bundle SHA','complete train+fresh20M logs and all4 graph CSVs','explicit-process-postflight.json including all earlier and final eval/compiler PIDs','post-eval-checkpoint-hash.json'],checks=len(checks),ledger_written=False,source_gate_replaced=False)
    require(bool(re.fullmatch('[0-9a-f]{40}',a.raw_commit or '')) and bool(re.fullmatch('[0-9a-f]{64}',a.bundle_sha or '')),'both immutable final pins required')
    root=OP+'completed/'
    bundle,f=decode(git(a.raw_commit,root+'results-bundle.json'),a.bundle_sha)
    require(f['NATIVE_BINDING.disabled.json']==git(PACKET,P+'NATIVE_BINDING.disabled.json') and f['MANIFEST.sha256.json']==git(PACKET,P+'MANIFEST.sha256.json'),'actual final frozen binding/manifest bytes')
    require(f['training.log'].startswith(first_five['training.log']) and f['train/steps.jsonl'].startswith(first_five['train/steps.jsonl']),'final logs preserve exact observed first-five prefixes')
    pr=git(a.raw_commit,root+'explicit-process-postflight.json');cr=git(a.raw_commit,root+'post-eval-checkpoint-hash.json')
    post=parse(pr);known_postflight(post)
    facts=closed_result(bundle,f,b,post,parse(cr),832589,KNOWN_PIDS)
    row=dict(schema='trn-run-record-v2',label=LABEL,arm_id='recursive6ef7-control-trn22-r2',kind='control',run_purpose='quality',host='trn22',timestamp_utc=facts['terminal_utc'],base_sha256=SOURCE,train_py_sha256=SOURCE,source_commit=b['source_commit'],status='evaluated',git_dirty=False,git_dirty_scope='Authenticated exact scientific bytes, not a globally clean remote checkout',argv=b['training_command'][4:],env=b['training_environment'],training_budget_seconds=1800,over_budget=False,steps=facts['steps'],charged_through_exit_seconds=facts['charged_through_exit_seconds'],startup_measured_seconds=facts['fifth_seconds'],cold_startup_proven=False,causality_passed=None,eval_tokens=20971520,eval_seq_len=1024,public_20m=facts['evaluation']['val_bpb'],checkpoint_sha256=facts['checkpoint']['sha256'],submission_eligible=False)
    return dict(status='AUTHENTICATED_GENUINE_CONTROL_RESULT_ONLY',facts=facts,ledger_candidate_only=row,ledger_written=False,historical_schedule_arm_relabelled=False,selector_verdict=None,source_gate_replaced=False,source_adopted=False,submission_authorized=False,raw_commit=a.raw_commit,bundle_sha256=a.bundle_sha,postflight_sha256=sha(pr),posteval_checkpoint_receipt_sha256=sha(cr),checks=len(checks))

if __name__=='__main__':
    try:result=main()
    except Exception as e:
        print(json.dumps(dict(status='HOLD_MISSING_OR_INVALID_FINAL_EVIDENCE',error=type(e).__name__+': '+str(e),checks=len(checks),ledger_written=False),allow_nan=False));raise SystemExit(2)
    print(json.dumps(result,indent=2,allow_nan=False))
