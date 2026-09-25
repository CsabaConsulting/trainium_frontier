"""Small CPU-only tests of the disabled B2 sparse-table prototype.

The competition environment supplies ``prepare`` and ``pyarrow``.  These tests
exercise only the optimizer/gradient helpers, so the imports are replaced by
inert stubs on development machines without the data stack.  No training data,
checkpoint, evaluator, or Neuron device is accessed.

Run with a CPU PyTorch environment::

    python research/v2/probes/test_b2_sparse_cpu.py
    torchrun --standalone --nproc_per_node=4 research/v2/probes/test_b2_sparse_cpu.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import torch
import torch.distributed as dist


def _load_candidate():
    parquet = types.ModuleType("pyarrow.parquet")
    pyarrow = types.ModuleType("pyarrow")
    pyarrow.__path__ = []
    pyarrow.parquet = parquet
    sys.modules["pyarrow"] = pyarrow
    sys.modules["pyarrow.parquet"] = parquet

    prepare = types.ModuleType("prepare")
    names = (
        "_document_batches", "_row_groups_for_split", "causality_check",
        "ensure_tokenizer", "evaluate_bpb", "get_dist_info", "get_token_bytes",
        "make_dataloader", "print0", "record_step_boundary",
    )
    for name in names:
        setattr(prepare, name, lambda *args, **kwargs: None)
    prepare.Tokenizer = type("Tokenizer", (), {})
    prepare.EVAL_TIMEOUT_SECONDS = 1
    prepare.TOKENIZER_VOCAB_SIZE = 8192
    prepare.TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS = 1
    prepare.TRAIN_STARTUP_STEPS_EXCLUDED = 1
    prepare.TRAIN_TIME_BUDGET_SECONDS = 1
    sys.modules["prepare"] = prepare

    path = Path(__file__).resolve().parents[3] / "submissions/research_v2/train.py"
    spec = importlib.util.spec_from_file_location("b2_sparse_candidate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _gradient(shape, rows, values):
    return torch.sparse_coo_tensor(
        torch.tensor([rows], dtype=torch.long),
        torch.tensor(values, dtype=torch.float32).reshape(len(rows), shape[1]),
        size=shape,
    )


def _optimizer(module, param):
    return module.MuonAdamW([
        {"params": [param], "kind": "rmsprop", "lr": 0.01,
         "betas": (0.0, 0.9), "eps": 1e-8, "weight_decay": 0.0}
    ])


def _local_tests(module):
    assert module.USE_B2_SPARSE_NGRAM is False

    embedding = torch.nn.Embedding(7, 2, sparse=True)
    embedding(torch.tensor([1, 1, 3])).sum().backward()
    assert embedding.weight.grad.layout == torch.sparse_coo

    p = torch.nn.Parameter(torch.zeros(6, 2))
    p.grad = _gradient((6, 2), [1, 1, 3], [[1, 2], [3, 4], [6, 8]])
    module.sync_clip_sparse_ngram_cpu([p], max_norm=5.0, row_capacity=3)
    got = p.grad.to_dense()
    expected = torch.zeros_like(got)
    expected[1] = torch.tensor([4.0, 6.0])
    expected[3] = torch.tensor([6.0, 8.0])
    expected *= 5.0 / (torch.linalg.vector_norm(expected) + 1e-6)
    torch.testing.assert_close(got, expected, atol=2e-6, rtol=0)
    assert p.grad._nnz() == 2

    # The actual table storage is BF16, so verify that the local sparse COO
    # exchange and indexed FP32-moment update accept that dtype as well.
    bf16 = torch.nn.Parameter(torch.zeros(6, 2, dtype=torch.bfloat16))
    bf16.grad = _gradient((6, 2), [2, 2], [[1, -1], [2, 1]]).to(torch.bfloat16)
    module.sync_clip_sparse_ngram_cpu([bf16], max_norm=0, row_capacity=2)
    assert bf16.grad.layout == torch.sparse_coo
    module.NGRAM_RMS_FP32 = True
    module.USE_NKI_NGRAM_RMS = False
    module.USE_B2_SPARSE_NGRAM = True
    _optimizer(module, bf16).step()
    assert bool(torch.isfinite(bf16).all())
    module.USE_B2_SPARSE_NGRAM = False

    # A row absent for several steps gets the same logical FP32 moment as a
    # dense zero-gradient RMSProp recurrence when it is next touched.
    module.NGRAM_RMS_FP32 = True
    module.USE_NKI_NGRAM_RMS = False
    module.USE_FOREACH_OPTIM = False
    module.USE_TENSOR_LR_SCALARS = False
    dense = torch.nn.Parameter(torch.zeros(6, 2))
    sparse = torch.nn.Parameter(dense.detach().clone())
    dense_opt = _optimizer(module, dense)
    sparse_opt = _optimizer(module, sparse)
    sequence = [(1, [1.0, 2.0]), (3, [3.0, 4.0]),
                (5, [2.0, -1.0]), (1, [-2.0, 1.0])]
    for row, values in sequence:
        dense_grad = torch.zeros_like(dense)
        dense_grad[row] = torch.tensor(values)
        dense.grad = dense_grad
        sparse.grad = _gradient(tuple(sparse.shape), [row], [values])
        module.USE_B2_SPARSE_NGRAM = False
        dense_opt.step()
        module.USE_B2_SPARSE_NGRAM = True
        sparse_opt.step()
        torch.testing.assert_close(sparse, dense, atol=2e-6, rtol=0)

    # Invalid gradients must fail before mutating a parameter or optimizer
    # state, otherwise a later retry cannot be trusted.
    previous_parameter = sparse.detach().clone()
    previous_step = sparse_opt.state[sparse]["step"]
    sparse.grad = _gradient(tuple(sparse.shape), [1], [[float("nan"), 1.0]])
    try:
        sparse_opt.step()
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite gradient did not fail")
    torch.testing.assert_close(sparse, previous_parameter)
    assert sparse_opt.state[sparse]["step"] == previous_step
    module.USE_B2_SPARSE_NGRAM = False


def _distributed_tests(module):
    dist.init_process_group("gloo", init_method="env://")
    rank = dist.get_rank()
    assert dist.get_world_size() == 4
    p = torch.nn.Parameter(torch.zeros(7, 2))
    if rank == 0:
        p.grad = _gradient((7, 2), [1], [[1, 2]])
    elif rank == 1:
        p.grad = _gradient((7, 2), [1, 2], [[3, 4], [1, 0]])
    elif rank == 2:
        p.grad = None
    else:
        p.grad = _gradient((7, 2), [2], [[0, 3]])
    module.sync_clip_sparse_ngram_cpu([p], max_norm=0, row_capacity=2)
    got = p.grad.to_dense()
    expected = torch.zeros_like(got)
    expected[1] = torch.tensor([1.0, 1.5])
    expected[2] = torch.tensor([0.25, 0.75])
    torch.testing.assert_close(got, expected, atol=0, rtol=0)

    module.USE_B2_SPARSE_NGRAM = True
    module.NGRAM_RMS_FP32 = True
    module.USE_NKI_NGRAM_RMS = False
    opt = _optimizer(module, p)
    opt.step()
    gathered = [torch.zeros_like(p) for _ in range(4)]
    dist.all_gather(gathered, p.detach())
    for copy in gathered:
        torch.testing.assert_close(copy, p)

    # A rank-local invalid input fails on all ranks before the two row
    # all-gathers, avoiding an unmatched collective sequence.
    p.grad = (_gradient((7, 2), [1], [[float("nan"), 0]]) if rank == 2
              else _gradient((7, 2), [1], [[1, 0]]))
    failed = False
    try:
        module.sync_clip_sparse_ngram_cpu([p], max_norm=0, row_capacity=2)
    except ValueError:
        failed = True
    assert failed
    dist.destroy_process_group()


if __name__ == "__main__":
    candidate = _load_candidate()
    if int(os.environ.get("WORLD_SIZE", "1")) == 4:
        _distributed_tests(candidate)
        if dist.is_available():
            print("B2 CPU four-rank exchange PASS", flush=True)
    else:
        _local_tests(candidate)
        print("B2 CPU local reference PASS", flush=True)
