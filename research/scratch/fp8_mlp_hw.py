"""Real-hardware check of the INTEGRATED code path: train.py's own MLP class, at the
real locked shape, on the neuron device, USE_FP8_LINEAR True vs False. Reference is
computed on CPU (never on device) so only the real path generates NEFFs."""
import os, sys, time
os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
os.environ["NEURON_CC_FLAGS"] = "--target trn2 --lnc 1"
sys.path.insert(0, os.path.expanduser("~/trainium_frontier"))
import torch, torch.nn.functional as F
import train as T

dev = torch.device("neuron")
cfg = T.GPTConfig(sequence_len=T.SEQ_LEN, vocab_size=8192, n_layer=T.DEPTH,
                  n_head=5, n_embd=T.ASPECT_RATIO*T.DEPTH)
print(f"n_embd={cfg.n_embd} hidden={4*cfg.n_embd}")

def rel_l2(ref, ap):
    ref, ap = ref.float(), ap.float()
    return float((ref-ap).norm()/ref.norm().clamp_min(1e-12))

B, Tt = 2, 1024   # 2048 rows, the real per-rank token count
torch.manual_seed(0)
x_cpu = torch.randn(B, Tt, cfg.n_embd, dtype=torch.float32)

# ---- Build the bf16 arm (USE_FP8_LINEAR off) ----
T.USE_FP8_LINEAR = False
torch.manual_seed(42)
mlp_off = T.MLP(cfg)
print("OFF arm c_fc class:", type(mlp_off.c_fc).__name__)
sd = {k: v.clone() for k, v in mlp_off.state_dict().items()}

# ---- Build the fp8 arm with IDENTICAL weights ----
T.USE_FP8_LINEAR = True
torch.manual_seed(42)
mlp_on = T.MLP(cfg)
mlp_on.load_state_dict(sd, strict=True)   # proves state_dict compatibility both ways
print("ON  arm c_fc class:", type(mlp_on.c_fc).__name__, "| state_dict keys identical:",
      sorted(sd) == sorted(mlp_on.state_dict()))

# ---- CPU fp32 reference (NOT on device) ----
T.USE_FP8_LINEAR = False
y_ref_cpu = mlp_off(x_cpu)
print(f"CPU fp32 reference: absmax={float(y_ref_cpu.abs().max()):.4f}")

def run_on_device(mlp, fp8_flag, label):
    T.USE_FP8_LINEAR = fp8_flag
    m = mlp.to(torch.bfloat16).to(dev)
    xb = x_cpu.to(torch.bfloat16).to(dev).requires_grad_(True)
    t0 = time.perf_counter()
    y = m(xb)
    y_host = y.detach().cpu().float()
    fwd_ms = (time.perf_counter()-t0)*1e3
    t1 = time.perf_counter()
    y.float().pow(2).mean().backward()
    gx = xb.grad.detach().cpu().float()
    gw_fc = m.c_fc.weight.grad.detach().cpu().float()
    bwd_ms = (time.perf_counter()-t1)*1e3
    print(f"  [{label}] fwd={fwd_ms:9.1f} ms  bwd={bwd_ms:9.1f} ms")
    print(f"  [{label}] out: rel_l2 vs CPU-fp32-ref={rel_l2(y_ref_cpu, y_host):.5f} "
          f"absmax={float(y_host.abs().max()):.4f} "
          f"nan={bool(torch.isnan(y_host).any())} inf={bool(torch.isinf(y_host).any())}")
    print(f"  [{label}] grad_x  absmax={float(gx.abs().max()):.6f} nan={bool(torch.isnan(gx).any())}")
    print(f"  [{label}] grad_w_fc absmax={float(gw_fc.abs().max()):.6f} nan={bool(torch.isnan(gw_fc).any())}")
    return y_host, gx, gw_fc

print()
print("=== USE_FP8_LINEAR=False on device (the incumbent) ===", flush=True)
y_off, gx_off, gw_off = run_on_device(mlp_off, False, "OFF")

print()
print("=== USE_FP8_LINEAR=True on device (2 real fp8 GEMMs) ===", flush=True)
y_on, gx_on, gw_on = run_on_device(mlp_on, True, "ON ")

print()
print("=== ON vs OFF, real hardware, identical weights ===")
print(f"  forward   rel_l2 = {rel_l2(y_off, y_on):.5f}   cos = {float(F.cosine_similarity(y_off.flatten(), y_on.flatten(), dim=0)):.6f}")
print(f"  grad_x    rel_l2 = {rel_l2(gx_off, gx_on):.5f}")
print(f"  grad_w_fc rel_l2 = {rel_l2(gw_off, gw_on):.5f}")
print(f"  offline harness predicted whole-MLP forward rel_l2 = 0.0561 (CPU-simulated)")
