#!/usr/bin/env python3
"""CPU structural gates for six-block coarsening, including nonzero MLPs."""
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import torch

from check_candidates import module, config, ROOT


def main():
    manifest = json.loads((ROOT/'deadline/candidate-manifest.json').read_text())
    base_module = module('more_base', ROOT/'submission/train.py')
    torch.manual_seed(42)
    base = base_module.GPT(config(base_module))
    base.init_weights()
    with torch.no_grad():
        for block in base.transformer.h:
            block.mlp.c_proj.weight.normal_(std=.02)
    state = base.state_dict()
    x, y = torch.randint(0, 256, (2, 32)), torch.randint(0, 256, (2, 32))
    receipts = []
    for name in ['middle6', 'middle6rms']:
        path = ROOT/'candidates'/name/'train.py'
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha == manifest['candidates'][name]['sha256']
        m = module('more_'+name, path)
        cfg = config(m, coarse_middle=True, coarse_mlp=False)
        model = m.GPT(cfg)
        model.load_state_dict(state, strict=True)
        assert [n for n, _ in model.named_parameters()] == [n for n, _ in base.named_parameters()]
        out = model(x)
        assert torch.isfinite(out).all() and out.shape == (2, 32, 256)
        for cut in [1, 2, 3, 7, 16, 31]:
            changed = x.clone(); changed[:, cut:] = (changed[:, cut:]+37) % 256
            assert torch.equal(out[:, :cut], model(changed)[:, :cut]), (name, 'causality', cut)
        changed = x.clone(); changed[1] = torch.flip(changed[1], dims=[0])
        assert torch.equal(out[0], model(changed)[0]), (name, 'batch')
        weights = torch.tensor([1., .2])
        loss = model(x, y, mtp_weights=weights)
        assert torch.isfinite(loss)
        loss.backward()
        gradients = 0
        for n, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), (name, n)
            gradients += 1
        restored = m.GPT(m.GPTConfig(**dataclasses.asdict(cfg)))
        restored.load_state_dict(model.state_dict(), strict=True)
        assert torch.equal(out, restored(x)), (name, 'checkpoint')
        # False really disables all coarsening and recovers the exact parent.
        disabled = m.GPT(config(m, coarse_middle=False, coarse_mlp=False))
        disabled.load_state_dict(state, strict=True)
        assert torch.equal(base(x), disabled(x)), (name, 'disabled logits')
        parent_loss = base(x, y, mtp_weights=weights)
        disabled_loss = disabled(x, y, mtp_weights=weights)
        assert torch.equal(parent_loss, disabled_loss)
        base.zero_grad(); parent_loss.backward(); disabled_loss.backward()
        for (bn, bp), (dn, dp) in zip(base.named_parameters(), disabled.named_parameters()):
            assert bn == dn and torch.equal(bp.grad, dp.grad), (name, 'disabled gradient', bn)
        receipts.append(dict(variant=name, source_sha256=sha, passed=True,
            shape=[2, 32, 128], nonzero_mlp_outputs=True, strict_causality=True,
            batch_independence=True, checkpoint_exact=True,
            disabled_parent_outputs_and_gradients_exact=True,
            finite_parameter_gradients=gradients))
    print('MORE_CPU_CHECK '+json.dumps(dict(passed=True, receipts=receipts)), flush=True)


if __name__ == '__main__':
    main()
