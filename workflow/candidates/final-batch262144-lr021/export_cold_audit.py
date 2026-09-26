"""Export exactly five audit attachments only after authenticated cold eval.

Creates one new directory; never uploads or overwrites older Downloads files.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OP = REPO/'research/v2/operator/20260926-operator-recovery'
SOURCE_SHA = 'cd0c4aa59754f46705355d964fad4528cf1de77324469b0829bcc0de149b757d'
PARENT_SHA = 'e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43'
COMMAND_SHA = 'b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd'


def sha(raw): return hashlib.sha256(raw).hexdigest()


def require(value, why):
    if not value: raise ValueError(why)


def verified_inputs():
    source = (HERE/'train.py').read_bytes()
    parent = (HERE.parent/'confirmed-clock36-116188/train.py').read_bytes()
    command = (HERE/'launch-command.txt').read_bytes()
    require(sha(source)==SOURCE_SHA and sha(parent)==PARENT_SHA, 'SOURCE_OR_PARENT_DRIFT')
    require(sha(command)==COMMAND_SHA, 'ACCEPTED_LAUNCH_COMMAND_DRIFT')
    require(command==(HERE.parent/'confirmed-clock36-116188/launch-command.txt').read_bytes(), 'PARENT_COMMAND_MISMATCH')
    bundle_path = OP/'manual-jobs/trn21-finalcold/results-bundle.json'
    require(bundle_path.is_file(), 'ACTUAL_COLD_BUNDLE_MISSING_NO_EXPORT')
    # Existing strict analyzer verifies hashes, exact source/argv, actual empty
    # caches, runtime accounting, fresh20M, checkpoint and process quiescence.
    subprocess.run([sys.executable,str(OP/'analyze_current.py'),'trn21-finalcold'],check=True)
    raw = bundle_path.read_bytes()
    bundle = json.loads(raw)
    row = json.loads((bundle_path.parent/'ledger-candidate.json').read_bytes())
    require(row['raw_archive_sha256']==sha(raw), 'COLD_LEDGER_BINDING_DRIFT')
    require(row['train_py_sha256']==SOURCE_SHA and row['seed']==42, 'WRONG_COLD_CANDIDATE')
    require(row['cold_startup_proven'] is True and row['over_budget'] is False, 'COLD_OR_BUDGET_UNPROVEN')
    require(row['eval_tokens']==20971520 and row['eval_seq_len']==1024, 'WRONG_EVAL_SCOPE')
    require(bundle['files']['package/train.py']['text'].encode()==source, 'TRAINED_SOURCE_DIFFERS')
    require(bundle['files']['package/launch-command.txt']['text'].encode()==command, 'TRAINED_COMMAND_DIFFERS')
    return source,parent,command,bundle,row


def build_files(source,parent,command,bundle,row):
    cpu = json.loads((HERE/'CPU_RECEIPT.json').read_bytes())
    require(cpu['status']=='PASS' and cpu['candidate_sha256']==sha(source)
            and cpu['vehicle_sha256']==sha(parent)
            and cpu['launch_command_sha256']==sha(command),'CPU_RECEIPT_SOURCE_BINDING_DRIFT')
    selected = ('operator-preflight.json','NATIVE_BINDING.json','terminal.json',
                'train/rehearsal.json','eval/rehearsal.json','fresh-public20m.log',
                'checkpoint.json','checkpoint-after-postflight.json')
    evidence = {'status':'AUDIT_HANDOFF_NOT_UPLOAD_APPROVAL','cold_result':row,
                'candidate_sha256':sha(source),'accepted_parent_sha256':sha(parent),
                'launch_command_sha256':sha(command),'cpu_scope_receipt':cpu,
                'cold_raw_files':{k:bundle['files'][k] for k in selected},
                'postflight':bundle['postflight'],
                'pending':['Claude source/compliance audit','selected-stack second-seed confirmation','Poojith upload decision'],
                'official_best_human_report':0.9878,'new_official_score':None}
    request = (HERE/'CLAUDE_REVIEW_REQUEST.md').read_text()
    # Historical request stays verbatim, explicitly superseded by measured block.
    brief = f'''# Claude audit handoff — NOT upload approval

This folder contains exactly five attachments. Attach all five to the reviewer.
Candidate train.py SHA256: {sha(source)}
Accepted parent SHA256: {sha(parent)}

## Actual cold result (supersedes pending-cold prose in historical request below)

Public20M/context1024: {row['public_20m']:.12f}; seed42; {row['steps']} steps.
Cold startup: {row['cold_startup_seconds']:.6f}s.
Startup excess: {row['startup_excess_seconds']:.6f}s.
Charged through process exit: {row['charged_through_exit_seconds']:.6f}s; over_budget=False.
Archive SHA256: {row['raw_archive_sha256']}
This is public-dev, NOT a new official score or an acceptance guarantee.
Historical confirmed research mean:0.986879270651192 (different research source).
The cold result must be used as measured, not replaced with that mean.

## Exact unchanged launch command

```bash
{command.decode().strip()}
```

Only upload train.py if Poojith approves after audit and final result checks.
Do NOT upload accepted_parent.py or cold-evidence.json. No automatic submission.
The ordinary batch/LR stack is included; no Recursive, momentum150, K2 or tables.
Second-seed confirmation and Claude audit remain pending at this handoff.

## Original scoped source-review request

{request}

The five attachments here supersede that request's older attachment list:
train.py, accepted_parent.py, README.md, AUDIT_BRIEF.md, cold-evidence.json.
The evidence JSON includes the CPU receipt and actual cold runtime receipts.
Do not follow historical launch or pending-state prose over the measured block.
'''
    return {'train.py':source,'accepted_parent.py':parent,'README.md':(REPO/'README.md').read_bytes(),
            'AUDIT_BRIEF.md':brief.encode(),'cold-evidence.json':(json.dumps(evidence,indent=2)+'\n').encode()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true',help='Authenticate and render without exporting')
    args=parser.parse_args()
    inputs=verified_inputs()
    files=build_files(*inputs)
    require(len(files)==5 and sha(files['train.py'])==SOURCE_SHA,'EXPORT_CONTENT_MISMATCH')
    if args.check:
        print(json.dumps({'status':'AUTHENTICATED_AUDIT_EXPORT_READY','files':list(files),'writes':False}))
        return
    root=Path('/home/poojith-devan/Downloads')
    require(root.is_dir() and not root.is_symlink(),'DOWNLOADS_DIRECTORY_UNAVAILABLE')
    name='Trainium-cold-AUDIT-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-cd0c4aa5'
    target=root/name
    target.mkdir(mode=0o700,exist_ok=False)
    for name,data in files.items():
        with (target/name).open('xb') as stream: stream.write(data)
        require((target/name).read_bytes()==data,'EXPORTED_BYTES_MISMATCH:'+name)
    print(json.dumps({'status':'AUDIT_ONLY_NOT_UPLOAD_APPROVAL','folder':str(target),
                      'source_sha256':SOURCE_SHA,'files':list(files),'older_files_untouched':True}))


if __name__=='__main__': main()
