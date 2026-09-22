"""torchrun worker: optimizer-ownership equivalence between two train.py modules on CPU/gloo."""
import importlib.util
import os
import sys

import torch
import torch.distributed as dist


def load(path, name, frozen=None):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    # put the ORIGINAL module into the launch configuration (its import-time defaults differ,
    # e.g. USE_NGRAM_VE=True would add an RMSProp group the ownership adapter rejects)
    for k, v in (frozen or {}).items():
        if hasattr(mod, k) and not isinstance(v, (list, dict)):
            setattr(mod, k, tuple(v) if isinstance(getattr(mod, k), tuple) and isinstance(v, list) else v)
    return mod


def run(mod, install_kwargs):
    torch.manual_seed(0)
    cfg = mod.GPTConfig(sequence_len=64, vocab_size=mod.TOKENIZER_VOCAB_SIZE, n_layer=2, n_head=1, n_embd=128)
    model = mod.GPT(cfg)
    model.init_weights()
    opt = model.setup_optimizer()
    mod._submission_optimizer_install(opt, mod, **install_kwargs)
    gen = torch.Generator().manual_seed(1234)
    for step in range(3):
        for p in model.parameters():
            p.grad = torch.randn(p.shape, generator=gen, dtype=p.dtype) * 1e-2
        for g in opt.param_groups:
            g["lr"] = g["initial_lr"] * 0.5
            if g["kind"] == "muon":
                g["momentum"] = 0.9
                g["weight_decay"] = 0.1
                g["beta2"] = 0.9
        opt.step(update_adamw=True)
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def main():
    orig_path, lean_path, report_path = sys.argv[1], sys.argv[2], sys.argv[3]
    import json
    frozen = json.load(open(report_path))["globals_frozen"]
    dist.init_process_group("gloo")
    orig = load(orig_path, "train_orig", frozen)
    lean = load(lean_path, "train_lean")
    a = run(orig, dict(pad_singleton_muon=True))
    b = run(lean, {})
    bad = [n for n in a if not torch.equal(a[n], b[n])]
    if dist.get_rank() == 0:
        print(f"OWNED_OPTIMIZER_EQUIV rank0 params={len(a)} mismatched={bad}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
