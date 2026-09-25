"""Test-only compatibility shim for the existing laptop PyTorch 2.6 Gloo runtime.

Gloo here does not implement AVG. Synthetic bounded tests use SUM/world-size for
that one operation. Candidate bytes remain unchanged. Never deploy this directory.
"""
import torch.distributed as dist

_real_all_reduce = dist.all_reduce

def _all_reduce_cpu_avg(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
    if op == dist.ReduceOp.AVG and dist.get_backend(group) == "gloo":
        if async_op:
            raise RuntimeError("local test shim does not support asynchronous AVG")
        _real_all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        tensor.div_(dist.get_world_size(group))
        return None
    return _real_all_reduce(tensor, op=op, group=group, async_op=async_op)

dist.all_reduce = _all_reduce_cpu_avg
