#!/usr/bin/env python3
"""CPU gates including optimizer coverage, causality and unchanged backbone RNG."""
import dataclasses
import hashlib
import json
import torch
from check_candidates import module, config, ROOT


def main():
    manifest = json.loads((ROOT/'deadline/candidate-manifest.json').read_text())
    m = module('head_base', ROOT/'submission/train.py')
    torch.manual_seed(42); base = m.GPT(config(m)); base.init_weights()
    reference_rng = torch.get_rng_state().clone()
    state = base.state_dict()
    x, y = torch.randint(0, 256, (2, 32)), torch.randint(0, 256, (2, 32))
    receipts = []
    for name in ['headbias', 'dualhead']:
        path = ROOT/'candidates'/name/'train.py'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['candidates'][name]['sha256']
        m = module('head_'+name, path)
        torch.manual_seed(42); model = m.GPT(config(m)); model.init_weights()
        assert torch.equal(reference_rng, torch.get_rng_state()), (name, 'RNG')
        for key, value in state.items():
            assert torch.equal(value, model.state_dict()[key]), (name, 'backbone init', key)
        if name == 'headbias':
            assert torch.equal(base(x), model(x)), 'Zero bias must recover parent logits exactly'
        # Exercise nonzero feedforward residuals rather than only zero init.
        with torch.no_grad():
            for block in model.transformer.h:
                block.mlp.c_proj.weight.normal_(std=.02)
        out = model(x); assert torch.isfinite(out).all()
        for cut in [1, 2, 3, 7, 16, 31]:
            changed=x.clone(); changed[:,cut:] = (changed[:,cut:]+37) % 256
            assert torch.equal(out[:,:cut],model(changed)[:,:cut]), (name,'causality',cut)
        changed=x.clone();changed[1]=torch.flip(changed[1],dims=[0])
        assert torch.equal(out[0],model(changed)[0])
        loss=model(x,y,mtp_weights=torch.tensor([1.,.2]));assert torch.isfinite(loss);loss.backward()
        for key,p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), (name,key)
        opt=model.setup_optimizer()
        params=[p for g in opt.param_groups for p in g['params']]
        assert len({id(p) for p in params})==len(params)
        assert {id(p) for p in params}=={id(p) for p in model.parameters()}
        reload=m.GPT(m.GPTConfig(**dataclasses.asdict(model.config)))
        reload.load_state_dict(model.state_dict(),strict=True)
        assert torch.equal(out,reload(x))
        receipts.append(dict(variant=name,passed=True,backbone_init_and_rng_unchanged=True,
            causal_prefix_exact=True,batch_independence=True,checkpoint_exact=True,
            finite_parameter_gradients=len(list(model.parameters())),optimizer_coverage_exact=True))
    a=torch.linspace(-100,100,1000);b=torch.flip(a,dims=[0])
    got=torch.maximum(a,b)+torch.log1p(torch.exp(-(a-b).abs()))
    assert torch.allclose(got,torch.logaddexp(a,b),atol=2e-6,rtol=2e-6)
    print('HEAD_CPU_CHECK '+json.dumps(dict(passed=True,receipts=receipts)),flush=True)


if __name__ == '__main__':
    main()
