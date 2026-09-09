import os, time
os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
os.environ["NEURON_CC_FLAGS"] = "--target trn2 --lnc 1"
import torch, torch.nn.functional as F
import torch_neuronx  # noqa
dev = torch.device("neuron")
torch.manual_seed(0)
FP8MAX = 448.0
print("has torch.neuron:", hasattr(torch, "neuron"),
      "| synchronize:", hasattr(getattr(torch, "neuron", None), "synchronize"))

def sync():
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()

M, K, N = 2048, 640, 2560
xb = (torch.randn(M, K)).to(torch.bfloat16).to(dev)
wb = (torch.randn(N, K) * 0.05).to(torch.bfloat16).to(dev)
x8 = (torch.randn(M, K) * 100).to(torch.float8_e4m3fn).to(dev)
w8t = (torch.randn(K, N) * 5).to(torch.float8_e4m3fn).to(dev).contiguous()
sa = torch.tensor(0.01).to(dev)
sb = torch.tensor(0.2).to(dev)

# Methodology: force a REAL host-visible sync each iteration by pulling one
# scalar off the device. Identical for both arms -> apples-to-apples.
def timed_calls(name, fn, n=6):
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        y = fn()
        s = float(y.flatten()[0].cpu())   # forces completion, both arms alike
        sync()
        ts.append((time.perf_counter() - t0) * 1e3)
        print(f"    {name} call {i}: {ts[-1]:10.3f} ms  (sample={s:.4f})", flush=True)
    warm = ts[2:]
    print(f"  >> {name}: first={ts[0]:.3f} ms  steady(mean of {len(warm)})={sum(warm)/len(warm):.3f} ms", flush=True)
    return sum(warm)/len(warm)

print()
print("=== bf16 F.linear (the incumbent) ===", flush=True)
t_bf16 = timed_calls("bf16", lambda: F.linear(xb, wb))

print()
print("=== fp8 torch._scaled_mm, operands ALREADY cast (matmul cost only) ===", flush=True)
t_smm = timed_calls("_scaled_mm", lambda: torch._scaled_mm(x8, w8t, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))

print()
print(f"RATIO _scaled_mm / bf16 = {t_smm/t_bf16:.1f}x")

print()
print("=== control: is it _scaled_mm specifically, or any fp8 op? bf16->fp8 cast alone ===", flush=True)
t_cast = timed_calls("cast", lambda: (xb.float() * 7.0).to(torch.float8_e4m3fn).float())

print()
print("=== control: fp8 operands but a bf16 matmul (upcast fp8->bf16 then F.linear) ===", flush=True)
t_up = timed_calls("fp8->bf16 mm", lambda: F.linear(x8.to(torch.bfloat16), w8t.t().to(torch.bfloat16)))
