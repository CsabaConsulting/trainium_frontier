"""Small synthetic CPU falsifiers; never reads data or contacts a host."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from collections import Counter

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "workflow/tools/stub_prepare"))
# Import-only pyarrow stub rejects all data access; do not load the old Gloo shim.
SUPPORT = ROOT / "research/v2/lanes/recursive-schedule/test_support"
sys.path.insert(0, str(SUPPORT))
import torch

SOURCE = ROOT / "workflow/proposals/research-v2/train.py"
torch.set_num_threads(1)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def configure(mod, *, tables=True, engram=False, projection=True):
    for name, value in dict(
        COMPUTE_DTYPE=torch.bfloat16, USE_NGRAM_VE=tables, USE_VE=True,
        NGRAM_VE_LAYERS="1,-1", NGRAM_VE_TABLE_MULT=4,
        NGRAM_VE_DIM=16 if projection else 128, NGRAM_VE_TRIGRAM=True,
        NGRAM_VE_BF16=True, NGRAM_VE_OPT="rmsprop", USE_ENGRAM=engram,
        ENGRAM_SHARE_NGRAM_TABLES=False, ENGRAM_TABLE_MULT=1,
        ENGRAM_MEM_DIM=8, ENGRAM_KEY_DIM=8, ENGRAM_HEADS=1,
        USE_QK_SHIFT=False, USE_BF16_NORM_OUTPUT=True,
        COMPILE_SDPA_DIRECT=True,
        USE_NKI_ROPE_NORM=False, USE_NKI_SOFTCAP_CE=False,
        USE_NKI_MUON=False, USE_COMPILE_MUON=False,
        RECURSIVE_SCHEDULE=False,
    ).items():
        setattr(mod, name, value)
    # Keep the unmodified eval hook but bound its nonpersistent masks in this
    # small diagnostic. Exact-context G2 remains a separate gate.
    mod.EVAL_MIN_SEQ_LEN = 16
    mod.compute_dtype_for = lambda device: torch.bfloat16


def model(mod, *, nonzero):
    torch.manual_seed(42)
    m = mod.GPT(mod.GPTConfig(16, 64, 4, 1, 128))
    m.init_weights()
    if nonzero:
        with torch.no_grad():
            for table in m.ngram_embeds.values():
                table.weight.uniform_(-.05, .05)
            for block in m.transformer.h:
                block.attn.c_proj.weight.uniform_(-.05, .05)
                for gate in (block.attn.ve_gate_bi, block.attn.ve_gate_tri):
                    if gate is not None:
                        gate.weight.uniform_(-.05, .05)
    return m


def equal(a, b, label):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype
        assert torch.equal(a, b), label
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), label
        for key in a:
            equal(a[key], b[key], f"{label}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), label
        for i, (x, y) in enumerate(zip(a, b)):
            equal(x, y, f"{label}[{i}]")
    else:
        assert a == b, (label, a, b)


def inputs():
    # Repeated n-grams, distinct n-grams colliding under the actual hash, and
    # duplicates across both rows and successive microbatches.
    return torch.tensor([[3, 3, 3, 7, 8, 3, 3, 9, 7, 8, 3, 3, 3, 7, 8, 3],
                         [3, 3, 3, 7, 8, 3, 3, 9, 7, 8, 3, 3, 3, 7, 8, 3]],
                        dtype=torch.int32)


def collision_witness(m):
    witnesses = {}
    # Enumerate token triples through the original function, not a new hash.
    triples = torch.cartesian_prod(torch.arange(1, 17), torch.arange(1, 17), torch.arange(1, 17)).to(torch.int32)
    hashed = m._ngram_hash_indices(triples)
    for key, rows in hashed.items():
        seen = {}
        for tokens, row in zip(triples.tolist(), rows[:, -1].tolist()):
            gram = tuple(tokens[-(2 if key[0] == "b" else 3):])
            if row in seen and seen[row] != gram:
                witnesses[key] = [list(seen[row]), list(gram), row]
                break
            seen[row] = gram
        assert key in witnesses, key
    return witnesses


def exercise(mod, base, *, projection, nonzero):
    configure(mod, projection=projection)
    configure(base, projection=projection)
    off, parent = model(mod, nonzero=nonzero), model(base, nonzero=nonzero)
    on = copy.deepcopy(off)
    equal(off.state_dict(), parent.state_dict(), "initial parent")
    ids = [id(p) for p in on.parameters()]
    keys = list(on.state_dict())
    call = mod.ngram_gather_training_forward(on, on)
    opts = [m.setup_optimizer() for m in (parent, off, on)]
    for m, opt in zip((parent, off, on), opts):
        assert {id(p) for group in opt.param_groups for p in group["params"]} == {id(p) for p in m.parameters()}
    x = inputs().repeat(2, 1)
    witnesses = collision_witness(on)
    # Inject actual distinct colliding grams, plus repeated rows/microbatches.
    for i, (left, right, _) in enumerate(witnesses.values()):
        x[i, :len(left)] = torch.tensor(left, dtype=x.dtype)
        x[i, 8:8+len(right)] = torch.tensor(right, dtype=x.dtype)
    hashes = on._ngram_hash_indices(x)
    for i, (key, (left, right, row)) in enumerate(witnesses.items()):
        assert hashes[key][i, len(left)-1] == hashes[key][i, 8+len(right)-1] == row
    y = (x.long() + 1) % 64
    losses = []
    for step in range(3):
        for m in (parent, off, on):
            m.zero_grad(set_to_none=True)
        for micro in range(2):
            tokens = x if micro == 0 else x.flip(0)
            equal(parent(tokens), off(tokens), "parent logits")
            equal(off(tokens), call(tokens), "ON/OFF logits")
            values = [parent(tokens, y), off(tokens, y), call(tokens, y)]
            equal(values[0], values[1], "parent loss")
            equal(values[1], values[2], "ON/OFF loss")
            for value in values:
                (value / 2).backward()
            losses.append(float(values[1].detach()))
            for ps in zip(parent.named_parameters(), off.named_parameters(), on.named_parameters()):
                name = ps[0][0]
                equal(ps[0][1].grad, ps[1][1].grad, f"parent grad {name}")
                equal(ps[1][1].grad, ps[2][1].grad, f"ON/OFF grad {name}")
        for key, table in on.ngram_embeds.items():
            assert table.weight.grad is not None and not table.weight.grad.is_sparse
            if nonzero:
                assert bool(table.weight.grad.abs().max() > 0), key
            touched = on._ngram_hash_indices(x)[key].unique()
            untouched = torch.ones(on.ngram_rows, dtype=torch.bool)
            untouched[touched.long()] = False
            assert bool(untouched.any())
            assert not bool(table.weight.grad[untouched].any())
        for m, opt in zip((parent, off, on), opts):
            main, tables = mod.split_clip_params(m)
            mod.clip_and_sync_gradients(m, main, tables, max_norm=mod.GRAD_CLIP, clip_after_reduce=True)
            opt.step()
        equal(parent.state_dict(), off.state_dict(), "parent update")
        equal(off.state_dict(), on.state_dict(), "ON/OFF update")
        equal(opts[0].state_dict(), opts[1].state_dict(), "parent optimizer")
        equal(opts[1].state_dict(), opts[2].state_dict(), "ON/OFF optimizer")
        for table in on.ngram_embeds.values():
            assert opts[2].state[table.weight]["step"] == step + 1
    assert ids == [id(p) for p in on.parameters()] and keys == list(on.state_dict())
    with tempfile.TemporaryDirectory(prefix="gather-eval-") as tmp:
        frozen = []
        for name, m in (("off", off), ("on", on)):
            path = Path(tmp) / f"{name}.pt"
            mod.save_checkpoint(path, m, {"synthetic": True})
            frozen.append(mod.load_for_eval(str(path), torch.device("cpu")))
        state = {k: v.clone() for k, v in frozen[1].state_dict().items()}
        rng = torch.get_rng_state().clone()
        with torch.no_grad():
            a, b = [m(x) for m in frozen]
            equal(a, b, "eval ON/OFF")
            equal(b, frozen[1](x), "eval repeat")
            equal(b, torch.cat([frozen[1](row[None]) for row in x]), "batch decomposition")
            changed = x.clone()
            changed[:, 9:] = (changed[:, 9:] + 5) % 64
            equal(b[:, :9], frozen[1](changed)[:, :9], "prefix causality")
        equal(rng, torch.get_rng_state(), "eval RNG")
        equal(state, frozen[1].state_dict(), "eval state")
    return dict(projection=projection, nonzero=nonzero, losses=losses, collisions=witnesses,
                parameters=len(ids), tables=list(on.ngram_embeds), steps=3, microbatches_per_step=2)


def compatibility(mod, base):
    results = []
    for tables, engram in [(False, False), (True, True), (False, True)]:
        configure(mod, tables=tables, engram=engram)
        configure(base, tables=tables, engram=engram)
        off, parent = model(mod, nonzero=False), model(base, nonzero=False)
        equal(parent.state_dict(), off.state_dict(), "compatible OFF state")
        equal(parent(inputs()), off(inputs()), "compatible OFF logits")
        try:
            mod.ngram_gather_training_forward(off, off)
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported ON accepted")
        results.append(dict(tables=tables, engram=engram, off_identical=True, on_rejected=True))
    configure(mod)
    m = model(mod, nonzero=True)
    try:
        m._ngram_raw_lookup("absent", torch.zeros((1, 2), dtype=torch.int32))
    except KeyError:
        pass
    else:
        raise AssertionError("absent key accepted")
    try:
        m(inputs(), ngram_raw_rows=())
    except ValueError:
        pass
    else:
        raise AssertionError("missing rows accepted")
    return results


def graph_probe(mod):
    from torch._dynamo.backends.common import aot_autograd
    from functorch.compile import make_boxed_func
    configure(mod)
    m = model(mod, nonzero=True)
    records = []
    table_shape = [m.ngram_rows, m.ngram_dim]

    def compiler(kind):
        def capture(gm, example_inputs):
            tensor_inputs = [{"shape": list(t.shape), "dtype": str(t.dtype)}
                             for t in example_inputs if isinstance(t, torch.Tensor)]
            assert all(t["shape"] != table_shape for t in tensor_inputs), "full table entered core"
            targets = Counter(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
            forbidden = ("aten.index.Tensor", "aten.index_put", "aten._index_put", "aten.scatter")
            assert not any(any(name.startswith(p) for p in forbidden) for name in targets), targets
            # Check every intermediate as well as placeholders, including outputs.
            for node in gm.graph.nodes:
                value = node.meta.get("val")
                if isinstance(value, torch.Tensor):
                    assert list(value.shape) != table_shape, (kind, node.name)
            records.append(dict(kind=kind, inputs=tensor_inputs, nodes=dict(targets)))
            return make_boxed_func(gm.forward)
        return capture

    torch._dynamo.reset()
    core = torch.compile(m, backend=aot_autograd(fw_compiler=compiler("forward"),
                                                bw_compiler=compiler("backward")),
                         fullgraph=True, dynamic=False)
    call = mod.ngram_gather_training_forward(m, core)
    x, counts = inputs(), []
    for _ in range(3):
        m.zero_grad(set_to_none=True)
        rows_seen = []
        original = m.gather_ngram_rows

        def track(idx):
            rows = original(idx)
            for row in rows:
                assert row.grad_fn is not None and not row.is_leaf
                row.retain_grad()
            rows_seen.extend(rows)
            return rows

        m.gather_ngram_rows = track
        try:
            call(x, (x.long() + 1) % 64).backward()
        finally:
            m.gather_ngram_rows = original
        assert all(row.grad is not None and bool(row.grad.abs().max() > 0) for row in rows_seen)
        assert all(p.grad is not None and bool(p.grad.abs().max() > 0) for p in m.ngram_embeds.parameters())
        counts.append(dict(Counter(r["kind"] for r in records)))
    assert counts == [{"forward": 1, "backward": 1}] * 3, counts
    return dict(captures_after_each_call=counts, table_shape=table_shape,
                raw_rows=[2, 16, 16], graph_records=records,
                scope="CPU AOTAutograd counting backend only; no Neuron or timing claim")


def source_scope(base_path):
    """Reject any unrelated source change, including schedules/timing/SDK/eval."""
    old, new = [ast.parse(p.read_text()) for p in (base_path, SOURCE)]
    new.body = [n for n in new.body if not (isinstance(n, ast.FunctionDef)
                                          and n.name == "ngram_gather_training_forward")]
    old_gpt = next(n for n in old.body if isinstance(n, ast.ClassDef) and n.name == "GPT")
    new_gpt = next(n for n in new.body if isinstance(n, ast.ClassDef) and n.name == "GPT")
    added = {"_ngram_raw_lookup", "_ngram_project", "gather_ngram_rows"}
    new_gpt.body = [n for n in new_gpt.body if not (isinstance(n, ast.FunctionDef) and n.name in added)]
    for i, node in enumerate(new_gpt.body):
        if isinstance(node, ast.FunctionDef) and node.name in {"forward", "_ngram_lookup"}:
            new_gpt.body[i] = next(n for n in old_gpt.body if isinstance(n, ast.FunctionDef) and n.name == node.name)
    main = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    main.body = [n for n in main.body if not (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "training_forward" for t in n.targets)
        or isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and any(isinstance(a, ast.Constant) and a.value == "--ngram-gather-outside-compile" for a in n.value.args))]
    calls = []
    for node in ast.walk(main):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "training_forward":
            calls.append([ast.unparse(a) for a in node.args])
            node.func.id = "model"
    assert sorted(calls) == [["synthetic_x", "synthetic_y"], ["x", "y"]]
    assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
    return dict(unrelated_source_ast_identical=True, training_calls=calls,
                protected_paths="all source outside declared GPT helpers/forward and main flag/call wiring")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base_path = Path(args.base)
    expected = "6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3"
    assert hashlib.sha256(base_path.read_bytes()).hexdigest() == expected
    mod, base = load(SOURCE, "gather_candidate"), load(base_path, "gather_parent")
    checks = [exercise(mod, base, projection=projection, nonzero=nonzero)
              for projection, nonzero in [(True, False), (True, True), (False, True)]]
    result = dict(status="IMPLEMENTER_SELF_CHECK_PASS", torch=torch.__version__,
                  base_sha256=expected, source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  arithmetic_checks=checks, compatibility=compatibility(mod, base),
                  graph_probe=graph_probe(mod), independent_review=False,
                  native_timing=False, exact_size_g2=False, source_scope=source_scope(base_path))
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("graph_probe", "arithmetic_checks")}, indent=2))


if __name__ == "__main__":
    main()
