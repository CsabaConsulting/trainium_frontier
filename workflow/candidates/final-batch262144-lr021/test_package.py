"""Bounded CPU packaging proof, reusing the exact existing tiny optimizer harness."""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import resource
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
import ast
import copy
import gc
import hashlib
import inspect
import json
import math
import random
import shlex
import sys
import types
from dataclasses import asdict
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PARENT=ROOT/'research/v2/proposals/20260926-sprint-p0-clean-batch/train.py'
VEHICLE=ROOT/'workflow/candidates/confirmed-clock36-116188/train.py'
CANDIDATE=HERE/'train.py'
HARNESS=ROOT/'research/v2/lanes/sprint-batch-lr021-recursive-stack/author_cpu.py'
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def require(ok,message):
    if not ok:raise AssertionError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()


vehicle,candidate=VEHICLE.read_bytes(),CANDIDATE.read_bytes()
require(sha(vehicle)=='e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43','VEHICLE_DRIFT')
require(sha(candidate)=='cd0c4aa59754f46705355d964fad4528cf1de77324469b0829bcc0de149b757d','CANDIDATE_DRIFT')
require(sha(PARENT.read_bytes())=='8c6d7af73e3237690c7f9a61c24544393b21224996d60484502ab379303c34ee','PARENT_DRIFT')
restored=candidate.replace(b'TOTAL_BATCH_SIZE = 262144',b'TOTAL_BATCH_SIZE = 131072').replace(b'MATRIX_LR = 0.021',b'MATRIX_LR = 0.015')
require(restored==vehicle,'MORE_THAN_TWO_CONSTANTS_CHANGED')
command=(HERE/'launch-command.txt').read_bytes()
require(sha(command)=='b75f844f413575f0cc35164972b4abf68de41e4b6f39e9b5423a6070de005abd','ACCEPTED_COMMAND_CHANGED')
require(command==(VEHICLE.parent/'launch-command.txt').read_bytes(),'COMMAND_NOT_IDENTICAL')
trees={p:ast.parse(p.read_text()) for p in (PARENT,CANDIDATE)}
needed={'named','dump','load','cli_prefix','route','digest','run'}
selected=[n for n in ast.parse(HARNESS.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in needed]
require({n.name for n in selected}==needed,'HARNESS_FUNCTION_MISSING')
exec(compile(ast.Module(body=selected,type_ignores=[]),str(HARNESS)+':selected-functions','exec'),globals())
a=load(PARENT,'package_research_reference')
b=load(CANDIDATE,'package_submission_candidate')
require((b.TOTAL_BATCH_SIZE,b.MATRIX_LR,b.WARMDOWN_RATIO,b.MUON_MOMENTUM_WARMUP_STEPS)==(262144,.021,.75,300),'BAKED_CONSTANTS')
argv=shlex.split(command.decode().split('train.py ',1)[1])
ra=cli_prefix(a,PARENT,argv+['--total-batch-size','262144','--matrix-lr','0.021'])
rb=cli_prefix(b,CANDIDATE,argv)
for name in ('TOTAL_BATCH_SIZE','MATRIX_LR','PACK_FACTOR','SEQ_LEN','N_EMBD'):
    require(getattr(a,name)==getattr(b,name),'CLI_GLOBAL:'+name)
for name in vars(rb):require(getattr(ra,name)==getattr(rb,name),'CLI_ARG:'+name)
require((rb.seed,rb.max_train_seconds)==(42,1800),'SEED_OR_BUDGET')
require(b.TOTAL_BATCH_SIZE//(b.DEVICE_BATCH_SIZE*b.PACK_FACTOR*b.SEQ_LEN*4)==2,'ACCUMULATION')
ar=run(a,262144,.021,2,False)
br=run(b,262144,.021,2,False)
require(ar==br,'ACTUAL_MODEL_OPTIMIZER_3STEP_IDENTITY')
receipt={'status':'PASS','candidate_sha256':sha(candidate),'research_source_sha256':sha(PARENT.read_bytes()),
         'vehicle_sha256':sha(vehicle),'launch_command_sha256':sha(command),
         'only_baked_constants':{'TOTAL_BATCH_SIZE':262144,'MATRIX_LR':.021},
         'argv_and_effective_globals_match':True,'two_accumulation_microsteps':True,
         'actual_model_optimizer_three_step_bit_identity':br,
         'harness_sha256':sha(HARNESS.read_bytes()),
         'scope':'Synthetic reduced8-layer128wide, BF16 CPU fallback, actual Muon/AdamW, no distributed/native/full-size claim',
         'native_passed':False,'cold_replay_passed':False,'claude_audit_passed':False,'submission_authorized':False}
(HERE/'CPU_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='actual_model_optimizer_three_step_bit_identity'}))
