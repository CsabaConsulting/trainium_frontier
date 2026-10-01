#!/usr/bin/env python3
"""Compare cleaned deliveries to their scored sources on CPU, including gradients."""
import dataclasses,hashlib,json,sys
from pathlib import Path
import torch
from check_candidates import module,config,ROOT


def main():
 records=[]
 for item in json.loads((ROOT/'deadline_submission/cleaning-receipt.json').read_text()):
  name=item['variant'];source=ROOT/item['source'];target=ROOT/item['delivery']
  assert hashlib.sha256(source.read_bytes()).hexdigest()==item['source_sha256']
  assert hashlib.sha256(target.read_bytes()).hexdigest()==item['delivery_sha256']
  a=module('delivery_original_'+name,source);b=module('delivery_clean_'+name,target)
  torch.manual_seed(42);original=a.GPT(config(a));original.init_weights();rng=torch.get_rng_state().clone()
  torch.manual_seed(42);clean=b.GPT(config(b));clean.init_weights()
  assert torch.equal(rng,torch.get_rng_state())
  assert dataclasses.asdict(original.config)==dataclasses.asdict(clean.config)
  state=original.state_dict()
  for key,value in state.items():assert torch.equal(value,clean.state_dict()[key]),(name,key,'initialization')
  with torch.no_grad():
   for block in original.transformer.h:block.mlp.c_proj.weight.normal_(std=.02)
  clean.load_state_dict(original.state_dict(),strict=True)
  x=torch.randint(0,256,(2,32));y=torch.randint(0,256,(2,32));w=torch.tensor([1.,.2])
  assert torch.equal(original(x),clean(x)),(name,'logits')
  la=original(x,y,mtp_weights=w);lb=clean(x,y,mtp_weights=w)
  assert torch.equal(la,lb),(name,'loss');la.backward();lb.backward()
  count=0
  for (key,p),(ckey,cp) in zip(original.named_parameters(),clean.named_parameters()):
   assert key==ckey and torch.equal(p.grad,cp.grad),(name,key,'gradient');count+=1
  oa=original.setup_optimizer();ob=clean.setup_optimizer()
  na={id(p):k for k,p in original.named_parameters()};nb={id(p):k for k,p in clean.named_parameters()}
  for ga,gb in zip(oa.param_groups,ob.param_groups):
   assert {k:v for k,v in ga.items() if k!='params'}=={k:v for k,v in gb.items() if k!='params'},(name,'optimizer settings')
   assert [na[id(p)] for p in ga['params']]==[nb[id(p)] for p in gb['params']],(name,'optimizer coverage')
  records.append(dict(variant=name,passed=True,initialization_rng_exact=True,nonzero_mlp_logits_exact=True,loss_exact=True,parameter_gradients_exact=count,optimizer_settings_and_coverage_exact=True,source_sha256=item['source_sha256'],delivery_sha256=item['delivery_sha256']))
 print('DELIVERY_CPU_PARITY '+json.dumps(dict(passed=True,receipts=records)),flush=True)

if __name__=='__main__':main()
