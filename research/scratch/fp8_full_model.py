"""Maximum real training-path evidence obtainable given the measured fp8 GEMM cost:
ONE full-model forward+backward of the real GPT at the locked shape, fp8 ON vs OFF,
on real hardware. Reports the real loss, a NaN/Inf audit over every gradient, and the
real per-micro-step wall time (which is what makes a multi-step run infeasible)."""
import os, sys, time
os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
os.environ["NEURON_CC_FLAGS"] = "--target trn2 --lnc 1"
sys.path.insert(0, os.path.expanduser("~/trainium_frontier"))
import torch
import train as T

dev = torch.device("neuron")
T.COMPUTE_DTYPE = torch.bfloat16
cfg = T.GPTConfig(sequence_len=T.SEQ_LEN, vocab_size=50257, n_layer=T.DEPTH,
                  n_head=T.DEPTH * T.ASPECT_RATIO // 128, n_embd=T.DEPTH * T.ASPECT_RATIO)
print(f"cfg n_embd={cfg.n_embd} n_head={cfg.n_head} n_layer={cfg.n_layer} seq_len={cfg.sequence_len}")

torch.manual_seed(1234)
x = torch.randint(0, cfg.vocab_size, (T.DEVICE_BATCH_SIZE * T.PACK_FACTOR, T.SEQ_LEN), dtype=torch.int32)
y = torch.randint(0, cfg.vocab_size, (T.DEVICE_BATCH_SIZE * T.PACK_FACTOR, T.SEQ_LEN), dtype=torch.long)

results = {}
for flag in (False, True):
    T.USE_FP8_LINEAR = flag
    torch.manual_seed(777)
    model = T.GPT(cfg)
    model.init_weights()
    n_par = sum(p.numel() for p in model.parameters())
    fc_cls = type(model.transformer.h[0].mlp.c_fc).__name__
    model = model.to(dev)
    xd, yd = x.to(dev), y.to(dev)
    model.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    loss = model(xd, yd)
    lv = float(loss.detach().cpu())
    fwd = time.perf_counter() - t0
    t1 = time.perf_counter()
    loss.backward()
    # NaN/Inf audit over EVERY gradient in the model
    bad = []
    gmax = 0.0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().cpu().float()
        if torch.isnan(g).any() or torch.isinf(g).any():
            bad.append(n)
        gmax = max(gmax, float(g.abs().max()))
    bwd = time.perf_counter() - t1
    label = "ON " if flag else "OFF"
    print(f"[{label}] c_fc class={fc_cls} params={n_par/1e6:.1f}M", flush=True)
    print(f"[{label}] loss={lv:.6f}  finite={lv == lv and abs(lv) != float('inf')}", flush=True)
    print(f"[{label}] fwd={fwd:8.2f}s  bwd={bwd:8.2f}s  micro_step_total={fwd+bwd:8.2f}s", flush=True)
    print(f"[{label}] grad NaN/Inf tensors={len(bad)} {bad[:5]}  grad_absmax={gmax:.6f}", flush=True)
    results[flag] = (lv, fwd + bwd)
    del model
    print(flush=True)

l_off, t_off = results[False]
l_on, t_on = results[True]
print("=== FULL MODEL, real hardware, identical init seed ===")
print(f"  loss  OFF={l_off:.6f}   ON={l_on:.6f}   delta={l_on-l_off:+.6f}")
print(f"  micro-step time OFF={t_off:.2f}s  ON={t_on:.2f}s  slowdown={t_on/t_off:.1f}x")
ga_1 = T.TOTAL_BATCH_SIZE // (T.DEVICE_BATCH_SIZE * T.PACK_FACTOR * T.SEQ_LEN * 1)
ga_4 = T.TOTAL_BATCH_SIZE // (T.DEVICE_BATCH_SIZE * T.PACK_FACTOR * T.SEQ_LEN * 4)
print(f"  grad_accum_steps: {ga_1} (1 rank) / {ga_4} (4 ranks)")
print(f"  => ONE optimizer step with fp8: {ga_1*t_on/3600:.2f} h (1 rank), {ga_4*t_on/3600:.2f} h (4 ranks)")
print(f"  => 100-step loss-trajectory run: {100*ga_4*t_on/86400:.1f} DAYS at 4 ranks")
