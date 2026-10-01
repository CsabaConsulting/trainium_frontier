"""CPU gates for onehot_embed.py.  Run: python test_onehot_embed.py"""
import torch
import torch.nn.functional as F
import onehot_embed as oe

torch.manual_seed(0)
V, D1, D2 = 8192, 64, 32


def ref_grads(idx, weights, gouts):
    ws = [w.detach().clone().requires_grad_() for w in weights]
    outs = [F.embedding(idx, w) for w in ws]
    torch.autograd.backward(outs, gouts)
    return [w.grad for w in ws]


def run(mode, idx, weights, gouts):
    ws = [w.detach().clone().requires_grad_() for w in weights]
    outs = oe.embed(idx, *ws, mode=mode)
    torch.autograd.backward(list(outs), gouts)
    return outs, [w.grad for w in ws]


def make(T_shape=(3, 7000)):  # 21000 tokens: 3 chunks incl. a ragged tail
    # Zipf-ish ids so some tokens repeat hundreds of times, like real text.
    ranks = torch.distributions.Exponential(0.01).sample(T_shape).long().clamp_max(V - 1)
    idx = ranks
    w1, w2 = torch.randn(V, D1), torch.randn(V, D2)
    g1, g2 = torch.randn(*T_shape, D1), torch.randn(*T_shape, D2)
    return idx, [w1, w2], [g1, g2]


idx, ws, gs = make()
print("max duplicate count:", torch.bincount(idx.reshape(-1)).max().item())
ref = ref_grads(idx, ws, gs)

# Gate 1: flag-off identity (bitwise).
outs0, g0 = run("0", idx, ws, gs)
assert all(torch.equal(a, b) for a, b in zip(g0, ref)), "flag-off not bitwise identical"
print("gate1 flag-off identity: OK (bitwise)")

# Gate 2: exactness with fp32 matmul (summation-order differences only).
oe.MM_DTYPE = torch.float32
for mode in ("fn", "op"):
    ws_in = ws if mode == "fn" else ws[:1]
    gs_in = gs if mode == "fn" else gs[:1]
    outs, g = run(mode, idx, ws_in, gs_in)
    assert all(torch.equal(o, F.embedding(idx, w)) for o, w in zip(outs, ws_in)), "forward changed"
    for a, b in zip(g, ref):
        err = (a - b).abs().max().item()
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-5), (mode, err)
    print(f"gate2 fp32 exactness [{mode}]: OK  max|err|={err:.2e}")

# Gate 3: bf16 fast path -- error is bf16 rounding of inputs + per-chunk sums.
oe.MM_DTYPE = torch.bfloat16
_, g = run("fn", idx, ws, gs)
for a, b in zip(g, ref):
    rel = ((a - b).norm() / b.norm()).item()
    assert rel < 1e-2, rel
    print(f"gate3 bf16 path: relative L2 err = {rel:.2e}")
# Same inputs pre-rounded to bf16: remaining error is only the per-chunk output rounding.
ref_bf = ref_grads(idx, ws, [x.bfloat16().float() for x in gs])
rel = max(((a - b).norm() / b.norm()).item() for a, b in zip(g, ref_bf))
print(f"gate3 bf16 path vs bf16-rounded-input ref: relative L2 err = {rel:.2e}")

# Gate 4: traceable by torch.compile, fullgraph (no graph break) for the fn path.
class M(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.wte, s.ve = torch.nn.Embedding(V, D1), torch.nn.Embedding(V, D1)
    def forward(s, idx):
        a, b = oe.embed(idx, s.wte.weight, s.ve.weight, mode="fn")
        return (a * b).sum()

m = M()
cm = torch.compile(m, backend="aot_eager", fullgraph=True)
small = idx[:, :4096]
cm(small).backward()
gc = [p.grad.clone() for p in m.parameters()]
for p in m.parameters():
    p.grad = None
m(small).backward()  # eager, same bf16 path
assert all(torch.equal(a, p.grad) for a, p in zip(gc, m.parameters())), "compiled != eager"
print("gate4 torch.compile(fullgraph=True, aot_eager) fn path: OK (no graph break)")

# Gate 5: custom_op registration sanity.
torch.library.opcheck(oe._embed_op, (idx[:, :512], torch.randn(V, 16, requires_grad=True)),
                      test_utils=("test_schema", "test_autograd_registration", "test_faketensor"))
print("gate5 custom_op opcheck: OK")
print("ALL CPU GATES PASSED")
