"""Strict scalar-receipt validator; no tensors/native imports."""
import json
from pathlib import Path
ORDER=['A','B','AB','BA','AC','AA','suffix_AB','AB_repeat','A_repeat']
CASE_NAMES={source+'_'+suffix for source in ('reference','candidate') for suffix in ('B2_vs_independent_rows','permutation','partner_content_A','duplicate_AA','repeat_AB','repeat_A','synthetic_suffix_causality','state_invariance')} | {'source_parity_'+key for key in ORDER}
EXACT='BEHAVIOURAL_DIAGNOSTIC_EXACT_ON_TESTED_SCOPE'
CANDIDATE='3f5035da293d984e15df915e2348258aea7f399bbf4d0e47aeb5c9896c35a47f'
REFERENCE='3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c'
CHECKPOINT='068749c2559f7c68f6acc76d558caff9d3806d85273fa8562483d87bbad50d25'
def require(ok,message):
 if not ok:raise RuntimeError(message)
def validate(root,label,backend):
 root=Path(root); expected=4 if backend=='neuron' else 1
 paths=sorted((root/'receipts').glob('rank*.json'));require(len(paths)==expected,'RANK_RECEIPT_COUNT')
 receipts=[json.loads(p.read_text()) for p in paths]
 for rank,r in enumerate(receipts):
  require(r.get('rank')==rank and r.get('world_size')==expected,'RANK_WORLD')
  require(r.get('status')==EXACT and r.get('backend')==backend and r.get('label')==label,'RECEIPT_STATUS')
  require(r.get('candidate_sha256')==CANDIDATE and r.get('reference_sha256')==REFERENCE and r.get('checkpoint_sha256')==CHECKPOINT,'INPUT_PINS')
  require(r.get('call_order')==ORDER,'CALL_MATRIX')
  require(len(r.get('cases',[]))==25 and {c['name'] for c in r['cases']}==CASE_NAMES and all(c.get('passed') is True for c in r['cases']),'ALL25_CASES')
  require(set(r.get('outputs',{}))=={'reference','candidate'},'SOURCE_OUTPUTS')
  for outputs in r['outputs'].values():require(list(outputs)==sorted(ORDER) or set(outputs)==set(ORDER),'OUTPUT_MATRIX')
  require(r['outputs']==receipts[0]['outputs'] and r['inputs']==receipts[0]['inputs'],'CROSS_RANK_OUTPUT_AND_INPUT_BYTES')
  require(r.get('source_review')=='REFUSE_UNCHANGED' and r.get('upload')=='HOLD','PRESERVED_HOLD')
 size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and 'caches' not in p.relative_to(root).parts)
 require(size<=256*1024**2,'ARTIFACT_CAP')
 return {'status':EXACT,'ranks':expected,'cases_per_rank':25,'source_review':'REFUSE_UNCHANGED','full_training':'HOLD','upload':'HOLD','artifact_bytes_excluding_sdk_caches':size,'quality_evidence':False}
