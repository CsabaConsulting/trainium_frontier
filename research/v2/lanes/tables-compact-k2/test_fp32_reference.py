"""Selected FP32 master/gather-BF16 contract; no large tensors/native imports."""
import torch
import torch.nn.functional as F

def exact(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8),
                       b.detach().contiguous().reshape(-1).view(torch.uint8))

def run(index_dtype=torch.int32):
    torch.set_num_threads(1)
    p = torch.zeros(8, 4, dtype=torch.float32, requires_grad=True)
    expected = torch.zeros_like(p)
    for values in ([1., 2**-8, -1.], [2**24, 1., -2**24]):
        rows = torch.tensor([0, 3, 0, 3, 0], dtype=index_dtype)
        g = torch.tensor([values[0], .5, values[1], -.5, values[2]], dtype=torch.bfloat16)[:, None].expand(5, 4)
        F.embedding(rows, p).to(torch.bfloat16).backward(g)
        part = torch.zeros_like(p)
        for i, row in enumerate(rows):
            part[row] = part[row] + g[i].float()
        expected = expected + part
        exact(p.grad, expected)
    assert p.grad[0, 0].item() == 2**-8
    assert p.grad.dtype == torch.float32
    return {'status': 'PASS_FP32_MASTER_REFERENCE', 'collision_and_microstep_gradient': p.grad[0].tolist()}

if __name__ == '__main__':
    print(run())
