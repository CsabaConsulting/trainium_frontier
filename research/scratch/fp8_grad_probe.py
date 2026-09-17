"""FP8 grad-flow probe: are backward grads alive under compiler auto-cast?

Runs a tiny GPT (1 layer) for 3 AdamW steps on synthetic data on Neuron and
prints per-group grad norms + update norms. Run twice and compare:
  NEURON_CC_FLAGS="--optlevel=1" ./venv/bin/python research/scratch/fp8_grad_probe.py  # control
  NEURON_CC_FLAGS="--optlevel=1 --auto-cast matmult --auto-cast-type fp8_e4m3" ...       # test
Verdict table:
  grads healthy in both -> backward fine; suspect update application (Muon path)
  grads ~0 only under fp8 -> fp8 grad underflow; needs loss scaling (or kill)
  grads None           -> backward graph broken by the cast (dump NEFF, kill)
CPU cannot reproduce this (needs neuronx-cc); runs on the Trn2 boxes only.
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train as T


def main() -> None:
    torch.manual_seed(1234)
    T.DEPTH, T.N_EMBD, T.SEQ_LEN = 1, 128, 256
    T.USE_NGRAM_VE = False
    T.USE_VE = False
    T.USE_HEAD_GATE = False
    T.USE_OUT_BIGRAM = False
    T.USE_OUT_POOL = False
    T.USE_ENGRAM = False
    T.USE_QK_SHIFT = False
    T.USE_X0_GATE = False
    T.USE_BLOCK_NUDGE = False
    T.USE_LOCAL_CONV = False
    device = torch.device("neuron")
    cfg = T.build_config()
    model = T.GPT(cfg)
    model.init_weights()
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    B, Tm, V = 1, 256, cfg.vocab_size
    for step in range(3):
        x = torch.randint(0, V, (B, Tm), dtype=torch.int32).to(device)
        y = torch.randint(0, V, (B, Tm)).to(device)
        opt.zero_grad(set_to_none=True)
        loss = model(x, y)
        loss.backward()
        total_sq, n_none, n_zero = 0.0, 0, 0
        worst = []
        for n, p in model.named_parameters():
            g = p.grad
            if g is None:
                n_none += 1
                worst.append((n, "None"))
            else:
                gn = float(g.float().norm().item())
                total_sq += gn * gn
                if gn == 0.0:
                    n_zero += 1
        import math
        print(f"step {step}: loss={float(loss.detach().float().item()):.4f} "
              f"grad_norm={math.sqrt(total_sq):.6f} none={n_none} zero={n_zero}",
              flush=True)
        opt.step()
    print("PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
