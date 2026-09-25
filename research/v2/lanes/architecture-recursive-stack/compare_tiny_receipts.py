"""Strict small G2 receipt summary; never imports candidate or Torch."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def require(ok,name):
 if not ok:raise RuntimeError(name)
def main():
 reports={name:json.loads((HERE/('tiny-g2-'+name+'.json')).read_text()) for name in ('reference','default','off','on')}
 base=reports['reference']
 for name,r in reports.items():
  require(r['verdict']=='PASS' and r['torchrun_rc']==0 and not r['timed_out'],name+':runtime')
  require(r['steps']==3 and r['nproc']==4 and r['micro_batch']==[1,16],name+':scope')
  require([v['rank'] for v in r['ranks']]==[0,1,2,3],name+':ranks')
  require(all(v['status']=='PASS' and v['n_params']==65 and v['n_changed']==65 and not v['cross_rank_mismatches'] for v in r['ranks']),name+':updates')
  require(r['ranks'][0]['checkpoint_check']['ok'] and r['ranks'][0]['checkpoint_check']['deterministic'],name+':checkpoint')
  require(len(r['ranks'][0]['final_digests'])==65,name+':digests')
  require(r['plan_sha256']==base['plan_sha256'],name+':ownership')
  if name in ('default','off'):
   require([v['losses'] for v in r['ranks']]==[v['losses'] for v in base['ranks']],name+':all12losses')
   require(r['ranks'][0]['final_digests']==base['ranks'][0]['final_digests'],name+':all65digests')
 changed=[key for key,digest in reports['on']['ranks'][0]['final_digests'].items() if digest!=base['ranks'][0]['final_digests'][key]]
 print(json.dumps({'status':'PASS_TINY_G2_ONLY','default_and_explicit_off_bit_identity':True,'all12_losses_exact':True,'all65_final_digests_exact':True,'ownership_plan_sha256':base['plan_sha256'],'on_vs_off_changed_tensors':changed,'on_finiteness_rank_consistency_reload':True,'source_hashes':{k:r['train_py_sha256'] for k,r in reports.items()},'fullsize':False,'native':False,'quality_evidence':False,'serialized_checkpoint_file_identity_not_claimed':True},indent=2))
if __name__=='__main__':main()
