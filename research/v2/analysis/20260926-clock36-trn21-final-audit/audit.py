"""One exact36 replica, local immutable archives only; no arguments => PENDING."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

REPO = Path(__file__).resolve().parents[4]
P = 'research/v2/proposals/20260926-startup-adjusted-full-trn21-replication/'
OP = 'research/v2/operator/20260926-trn21-startup-clock-replication/'
PACKET = 'f4fe5d8e557f3d0646d1a2d6a47858768e29954d'
LABEL = 'fs-20260926-arch36a8-startup-clock-trn21-r2-full'
SOURCE = '36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb'
BASE = '6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3'
KNOWN_TRAIN_PIDS = {3127536,3127537,3127538,3127554,3127555,3127569,3127584,3127585,3127586,3127587}
KNOWN_COMPILER_PIDS = {3128094,3128096,3128141,3128697,3128700,3128703,3128704,3128706,3128707}
KNOWN_PIDS = KNOWN_TRAIN_PIDS | KNOWN_COMPILER_PIDS
OBSERVATIONS = [
    ('d421376c', 'launch/first-native-evidence.json', '3346c10a49242a5b1d92f1c6303c22c86e0eab92f6d6a81bacb30c3c767bb1b3'),
    ('ea66038c1367db1e44d87491c557752a1bd84282', 'progress/0113-readonly.json', '518d7bc8f46de8a3c933614f61a70fa22e40ac22db3d14a55b15f4d08e840e62'),
    ('a8231ff04fb598b8258ad3a076603025188e970e', 'progress/early-startup-raw.json', 'db456939449ac977195b5fe736ce1f44d3657afa00049eb0e57f1f8bbe783b0f'),
    ('a2069c9f', 'progress/0870-startup-boundary.json', '856b36d0837fb06e70aa96fff39c39b194a9cec04368cd56368ee52a5f1708d5'),
    ('f1315e07', 'progress/0139-readonly.json', '250f5ccbcd4816ef0262edd27902f9b7da40271c35f9a15ed01eb250ef2a54e2'),
]
def git(ref, path):
    return subprocess.check_output(['git','-C',str(REPO),'show',ref+':'+path])
def sha(raw): return hashlib.sha256(raw).hexdigest()

# Reuse reviewed closed_result/accounting/JSON/hash/lineage checks verbatim.
# Do not invoke its main(), source gates, host APIs or selector.
helper = git('ae725297','research/v2/analysis/20260926-basebcbd-final-audit-preparation/audit.py')
if sha(helper) != '1ec138987f32d050448d2318702f06632f6ca8e1d531c9c3e1142812730dc770':
    raise RuntimeError('HELPER_PIN_MISMATCH')
ns = {'__name__':'immutable_closed_result_helper'}
exec(compile(helper,'ae725297:archive-helper','exec'),ns)
checks = ns['checks']
require,parse,decode,closed_result,timing,evaluation = [ns[n] for n in ('require','parse','decode','closed_result','timing','evaluation')]

def intended(b):
    require(b['label']==LABEL and b['host']=='trn21' and b['source_sha256']==SOURCE,'exact36 host/source/label')
    require(b['source_commit']=='7419927c7417b5666a984ec9c2b849feadabaaa9','exact source commit')
    cmd=b['training_command']
    require(all(x in cmd for x in ('--startup-adjusted-schedule','--initial-cache-refresh','--unet-8x1024')) and '--recursive-schedule' not in cmd,'exact clock architecture, no Recursive')
    require(b['training_environment']['NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS']=='1800','fixed1800')
    require(b['original_lock']['inode']==596823 and b['original_lock']['device']==66305,'original596823 lock')

def pins():
    raw=git(PACKET,P+'NATIVE_BINDING.disabled.json')
    require(sha(raw)=='26497d6dc9d5346779313fb437672f975501d5667b2ea21690344fb020d20ee0','frozen binding')
    b=parse(raw);intended(b)
    require(sha(git('18727a55','research/v2/verification/20260926-clock36-trn21-replication-assignment.md'))=='3438f12a11f0b73ffa2fb51c374bfff068c2f44fd21e500506f24ad23dd249e5','exact scoped assignment')
    require(sha(git(b['source_commit'],b['source_git_path']))==SOURCE,'scientific source bytes')
    require(sha(git(PACKET,P+'host_driver.py'))==b['driver_sha256'],'driver bytes')
    require(sha(git(PACKET,'prepare.py'))==b['prepare_sha256'] and sha(git('00f2ca7f','research/scratch/submission_rehearsal.py'))==b['harness_sha256'],'unchanged prepare/harness')
    for name,key in [('idle_guard.py','idle_guard_sha256'),('dispatch_driver.py','dispatch_driver_sha256')]:
        require(sha(git(PACKET,'research/v2/operator/20260926-common-host-idle-guard/'+name))==b[key],'shared support '+name)
    for name,pin in parse(git(PACKET,P+'MANIFEST.sha256.json'))['files'].items():
        require(sha(git(PACKET,P+name))==pin,'packet manifest '+name)
    observed=set(); prefixes=[]
    for ref,path,pin in OBSERVATIONS:
        obs,files=decode(git(ref,OP+path),pin)
        require(obs['root']==b['root'] and files['NATIVE_BINDING.disabled.json']==raw,'actual binding '+path)
        ids={int(x.split()[0]) for x in obs['job_lines']};observed|=ids
        require(KNOWN_TRAIN_PIDS<=ids,'actual training lineage '+path)
        require(len(obs['workers'])==4 and {w['environment']['RANK'] for w in obs['workers']}=={'0','1','2','3'},'four actual workers '+path)
        for w in obs['workers']:
            require(w['cwd']==b['sandbox'] and all(w['environment'][k]==v for k,v in b['training_environment'].items() if k in w['environment']),'observed worker environment '+str(w['pid']))
        prefixes.append(files)
    require(observed==KNOWN_PIDS,'all19 observed training/compiler identities authenticated')
    require(len(prefixes[-1]['train/steps.jsonl'].splitlines())>=5,'actual first-five present, not final result')
    ledger=git('4037a818','research/v2/ledger.jsonl')
    require(sha(ledger)=='51822b2f60cdcfd1eecc6ec70c3eb2987db9bccc03c76b2283d43b050c9da44e','immutable comparator ledger snapshot')
    names=[('fs-20260925-l3-base-repeat-trn21-r2-full','0f6b507896a008adca065aabbd8cfaf60f1720add3f6ee1632272601a93d5cc7'),('fs-20260926-recursive6ef7-control-trn22-r2','2aa775361056034bc4ba9fec018f57c285770fe3a97d984c0a2030e4b77acb58'),('fs-20260926-arch36a8-startup-clock-trn22-r1-full','e9544b798fcaa60602323ee74c4d8ba3b4bdc8a5f41ad5c9dfe50efa0b15692f')]
    rows=[]
    for name,pin in names:
        matches=[x for x in ledger.splitlines(keepends=True) if parse(x).get('label')==name]
        require(len(matches)==1 and sha(matches[0])==pin,'unchanged comparator row '+name);rows.append(parse(matches[0]))
    require(rows[0]['kind']==rows[1]['kind']=='control' and rows[0]['public_20m']==0.9899528600562257 and rows[1]['public_20m']==0.9912010195941572 and rows[2]['public_20m']==0.9883821509811542,'genuine controls and replica counterpart, no relabel')
    return b,prefixes,rows

def eval_identities(obs,files,b):
    """Require actual live evaluation observation, not guessed final census IDs."""
    require(obs['root']==b['root'] and parse(files['NATIVE_BINDING.disabled.json'])==b,'eval observation belongs to exact run')
    launch=parse(files['fresh-public20m-launch.json'])
    require(launch['command']==b['evaluation_command'] and launch['source_sha256']==b['source_sha256'] and launch['environment']==b['evaluation_environment'],'observed exact eval launch')
    rows=[]
    for line in obs['job_lines']:
        parts=line.split(None,6)
        require(len(parts)==7 and all(x.isdigit() for x in parts[:5]),'bounded ps identity row format')
        rows.append(dict(zip(('pid','ppid','pgid','sid'),map(int,parts[:4])),comm=parts[5],argv=shlex.split(parts[6])))
    require(rows and len({r['pid'] for r in rows})==len(rows),'unique observed eval identities')
    expected=[b['evaluation_command'][5],'prepare.py','eval-public','--train-py','train.py','--checkpoint',b['sandbox']+'/out/final.pt','--device','neuron','--batch-size','1','--seq-len','1024','--eval-tokens','20971520','--timeout-seconds','1200']
    selected=[]
    for cmd in (b['evaluation_command'],b['evaluation_command'][5:],expected):
        found=[r for r in rows if r['argv']==cmd]
        require(len(found)==1,'one actual timeout/harness/evaluator command');selected.append(found[0])
    timeout,harness,evaluator=selected
    require(harness['ppid']==timeout['pid'] and evaluator['ppid']==harness['pid'],'actual evaluation process chain')
    require(obs['lock_inode']==596823 and obs['lock_device']==66305 and obs['lock_competitor_rc']==1,'original lock held during eval observation')
    require(obs['neuron_ls_rc']==0 and not obs['neuron_ls_stderr'],'successful actual evaluation device observation')
    devices=parse(obs['neuron_ls'])
    require(any(p['pid']==evaluator['pid'] for d in devices for p in d['neuron_processes']),'actual evaluator reported by device')
    require(datetime.datetime.fromisoformat(launch['utc'])<=datetime.datetime.fromisoformat(obs['utc']),'eval observation after launch')
    return {r['pid'] for r in rows}

def known_postflight(post,required):
    require(required<=set(post['observed_pids']),'all earlier and actual evaluation/compiler IDs in final explicit census')
    require(not any(re.search('neuronx?-cc|neuronx-compile|walrus_driver',r['comm']) for r in post.get('all_process_identity_rows',[])),'no compiler/walrus process in final census')

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-commit');ap.add_argument('--bundle-sha')
    ap.add_argument('--eval-observation',nargs=3,metavar=('COMMIT','PATH','SHA256'))
    a=ap.parse_args(argv);b,prefixes,comparators=pins()
    if a.raw_commit is None and a.bundle_sha is None and a.eval_observation is None:
        return dict(status='ONLY_PENDING_REAL_RAW',label=LABEL,source_sha256=SOURCE,public_20m=None,selector_verdict=None,known_observed_pids=sorted(KNOWN_PIDS),evaluation_pids=None,required=['immutable final40hex commit and exact64hex bundle SHA','immutable actual in-evaluation observation commit/path/hash; no invented evaluator IDs','complete train+fresh20M, checkpoint pre/post hash, all4 graph CSVs and explicit final process/device/original-lock quiescence'],checks=len(checks),ledger_written=False,source_gate_replaced=False)
    require(bool(re.fullmatch('[0-9a-f]{40}',a.raw_commit or '')) and bool(re.fullmatch('[0-9a-f]{64}',a.bundle_sha or '')) and a.eval_observation is not None,'all final and actual evaluation pins required')
    ref,path,pin=a.eval_observation
    require(bool(re.fullmatch('[0-9a-f]{40}',ref)) and path.startswith(OP) and bool(re.fullmatch('[0-9a-f]{64}',pin)),'immutable run-scoped evaluation observation pins')
    obs,of=decode(git(ref,path),pin);actual_ids=eval_identities(obs,of,b)
    root=OP+'completed/';bundle,f=decode(git(a.raw_commit,root+'results-bundle.json'),a.bundle_sha)
    require(f['NATIVE_BINDING.disabled.json']==git(PACKET,P+'NATIVE_BINDING.disabled.json') and f['MANIFEST.sha256.json']==git(PACKET,P+'MANIFEST.sha256.json'),'final frozen binding/manifest bytes')
    for prefix in prefixes+[of]:
        for name in ('training.log','train/steps.jsonl','fresh-public20m.log'):
            if name in prefix:require(f[name].startswith(prefix[name]),'final preserves actual observed prefix '+name)
    require(f['fresh-public20m-launch.json']==of['fresh-public20m-launch.json'],'final equals actually observed evaluation launch')
    require(datetime.datetime.fromisoformat(obs['utc'])<=datetime.datetime.fromisoformat(parse(f['terminal.json'])['utc']),'live evaluation observed before final terminal')
    pr=git(a.raw_commit,root+'explicit-process-postflight.json');cr=git(a.raw_commit,root+'post-eval-checkpoint-hash.json')
    required=KNOWN_PIDS|actual_ids;post=parse(pr);known_postflight(post,required)
    facts=closed_result(bundle,f,b,post,parse(cr),596823,required)
    row=dict(schema='trn-run-record-v2',label=LABEL,arm_id='L2-architecture-startup-clock-36a862',kind='architecture_startup_clock_arm',run_purpose='quality',host='trn21',timestamp_utc=facts['terminal_utc'],base_sha256=BASE,train_py_sha256=SOURCE,source_commit=b['source_commit'],status='evaluated',git_dirty=False,git_dirty_scope='Authenticated frozen scientific bytes, not globally clean host checkout',argv=b['training_command'][4:],env=b['training_environment'],training_budget_seconds=1800,over_budget=False,steps=facts['steps'],charged_through_exit_seconds=facts['charged_through_exit_seconds'],startup_measured_seconds=facts['fifth_seconds'],cold_startup_proven=False,causality_passed=None,eval_tokens=20971520,eval_seq_len=1024,public_20m=facts['evaluation']['val_bpb'],checkpoint_sha256=facts['checkpoint']['sha256'],raw_archive_commit=a.raw_commit,raw_archive_sha256=a.bundle_sha,submission_eligible=False)
    return dict(status='AUTHENTICATED_REPLICA_RESULT_ONLY_NOT_ADOPTED',facts=facts,ledger_candidate_only=row,comparison_control=comparators[0],delta_vs_genuine_same_host_control=row['public_20m']-comparators[0]['public_20m'],trn22_comparators=comparators[1:],selector_verdict=None,ledger_written=False,historical_rows_relabelled=False,source_gate_replaced=False,source_adopted=False,submission_authorized=False,raw_commit=a.raw_commit,bundle_sha256=a.bundle_sha,evaluation_observation=dict(commit=ref,path=path,sha256=pin,observed_pids=sorted(actual_ids)),postflight_sha256=sha(pr),posteval_checkpoint_receipt_sha256=sha(cr),checks=len(checks))

if __name__=='__main__':
    try:result=main()
    except Exception as e:
        print(json.dumps(dict(status='HOLD_MISSING_OR_INVALID_FINAL_EVIDENCE',error=type(e).__name__+': '+str(e),checks=len(checks),ledger_written=False),allow_nan=False));raise SystemExit(2)
    print(json.dumps(result,indent=2,allow_nan=False))
