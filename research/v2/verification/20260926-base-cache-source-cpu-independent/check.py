#!/usr/bin/env python3
"""Independent bounded source/receipt audit and tiny Linear CPU reference.

Does not import the training module, run a full model, or contact any host.
"""
import argparse
import ast
import hashlib
import io
import json
import pathlib
import subprocess
import types

BASE = "a4aa4351789115a2d0aa947d9ca00ff689f9bd0b"
CANDIDATE = "b635c8b0cf2a10e1bcdd983e86192672ed6ad56f"
SOURCE = "workflow/proposals/research-v2/train.py"
SOURCE_SHA = {
    BASE: "6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3",
    CANDIDATE: "bcbd148dd51b5a014289cbecf3f0e684e35cf537e9588c878e15b265fcbfd522",
}
RAW_SHA = {
    "base": "842a2e36d7662e0e6a843a737fec1eb8d31d1d26ca9d95966e350db189dbe69b",
    "default": "13e5801681aea8588f38ddcd9866454f6895ddcaaf6b8ffdc1c4de47cb31cf58",
    "off": "08fbb5de1d51fabd44137c66a961f9c638bfba47506baee57c765759c7c6ac86",
    "on": "08b08c5ab2c79bafb5b197c6fa4375eee4ac59d4c3a5697cef29ff8b3653ed06",
}
checks = 0


def require(value, label):
    global checks
    checks += 1
    if not value:
        raise RuntimeError(label)


def extract_source(repo):
    sources = {}
    for revision, expected in SOURCE_SHA.items():
        data = subprocess.check_output(["git", "show", f"{revision}:{SOURCE}"], cwd=repo)
        require(hashlib.sha256(data).hexdigest() == expected, f"source hash {revision}")
        sources[revision] = ast.parse(data)
    baseline, candidate = sources[BASE], sources[CANDIDATE]
    main = next(n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    flag = next(n for n in ast.walk(main) if isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Call) and n.value.args
                and isinstance(n.value.args[0], ast.Constant)
                and n.value.args[0].value == "--initial-cache-refresh")
    block = next(n for n in ast.walk(main) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "step == 0 and args.initial_cache_refresh")

    class RemoveAdditions(ast.NodeTransformer):
        def visit_Expr(self, node):
            return None if node is flag else self.generic_visit(node)

        def visit_If(self, node):
            return None if node is block else self.generic_visit(node)

    stripped = RemoveAdditions().visit(candidate)
    require(ast.dump(stripped, include_attributes=False) == ast.dump(baseline, include_attributes=False),
            "all source AST except exact flag and first-step block must be unchanged")
    parser = argparse.ArgumentParser()
    exec(compile(ast.Module(body=[flag], type_ignores=[]), "<exact-flag>", "exec"),
         {"parser": parser, "argparse": argparse})
    for argv, expected in (([], False), (["--no-initial-cache-refresh"], False),
                           (["--initial-cache-refresh"], True)):
        require(parser.parse_args(argv).initial_cache_refresh is expected, f"parser {argv}")
    linear = next(n for n in baseline.body if isinstance(n, ast.ClassDef) and n.name == "Linear")
    return linear, block


def receipts(repo):
    root = repo / "research/v2/lanes/initial-cache-refresh"
    raw = {}
    for arm, expected in RAW_SHA.items():
        data = (root / f"tiny-{arm}.json").read_bytes()
        require(hashlib.sha256(data).hexdigest() == expected, f"raw hash {arm}")
        d = raw[arm] = json.loads(data)
        require(d["train_py_sha256"] == SOURCE_SHA[BASE if arm == "base" else CANDIDATE], "raw source")
        require(d["torchrun_rc"] == 0 and not d["timed_out"], "raw process status")
        require(d["steps"] == 3 and d["nproc"] == 4 and d["micro_batch"] == [2, 128], "raw shape")
        require(d["prepare"]["stub"] is True, "synthetic prepare scope")
        require([r["rank"] for r in d["ranks"]] == list(range(4)), "all ranks")
        require(d["plan_sha256"] == raw["base"]["plan_sha256"], "same plan")
        for rank, r in enumerate(d["ranks"]):
            ref = raw["base"]["ranks"][rank]
            obs, baseobs = r["cache_observations"], ref["cache_observations"]
            require(r["losses"] == ref["losses"] and len(r["losses"]) == 3, "three exact losses")
            require(r["n_params"] == 62, "62 parameters")
            require(obs["steps"] == baseobs["steps"], "all gradients/update/state including None match")
            require(len(obs["steps"]) == 3, "three observed steps")
            for s in obs["steps"]:
                for key in ("gradients_before_clip", "gradients_after_clip", "parameters_after_update"):
                    require(len(s[key]) == 62, f"62 records in {key}")
            require(len(obs["cache_refreshes"]) == (160 if arm == "on" else 120), "refresh count")
            initial = [v for v in obs["cache_refreshes"] if v["completed_steps"] == 0]
            require(len(initial) == (40 if arm == "on" else 0), "first-step refresh count")
            require(all(v["grad_enabled"] for v in obs["cache_refreshes"]), "grad context")
            first = obs["forwards"][0]["caches"]
            require(len(first) == 40, "forty Linear caches")
            if arm == "on":
                require(all(v["connected_first_forward"] and v["requires_grad"]
                            and v["master_dtype"] == "torch.float32"
                            and v["value"]["dtype"] == "torch.bfloat16" for v in first.values()),
                        "all first cache casts connected")
            else:
                require(all(v is None for v in first.values()), "OFF empty first cache")
            require([f["custom_mlp_nodes"] for f in obs["forwards"]] == [6, 6, 6],
                    "custom MLP graph nodes (not backward hook execution count)")
            if rank == 0:
                require(obs["checkpoint"] == baseobs["checkpoint"], "checkpoint payload digests")
                require(obs["checkpoint_file_sha256"] == baseobs["checkpoint_file_sha256"], "checkpoint file hash")
                require(r["final_digests"] == ref["final_digests"], "final parameter digests")
                require(r["checkpoint_check"]["ok"] and r["checkpoint_check"]["deterministic"], "reload record")
    return {"sha256": RAW_SHA, "ranks": 4, "steps": 3, "parameter_records": 62,
            "micro_batch": [2, 128], "plan_sha256": raw["base"]["plan_sha256"],
            "checkpoint_sha256": raw["base"]["ranks"][0]["cache_observations"]["checkpoint_file_sha256"],
            "scope": "historical intrusive tiny observer, synthetic data and fake clock; no native execution"}


def tiny_reference(linear_ast, block_ast):
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    ns = {"torch": torch, "nn": nn, "F": F, "USE_WEIGHT_CACHE": True}
    exec(compile(ast.Module(body=[linear_ast], type_ignores=[]), "<exact-Linear>", "exec"), ns)
    Linear = ns["Linear"]
    prime = compile(ast.Module(body=[block_ast], type_ignores=[]), "<exact-first-step>", "exec")
    cases = []
    # This lane uses BF16 compute with FP32 masters. An FP32 identity cast
    # returns the Parameter itself and the unchanged baseline then registers
    # _cached_weight as a Parameter; it is not this lane's cache contract.
    for dtype in (torch.bfloat16,):
        for transposed in (False, True):
            for micro_count in (1, 2):
                torch.manual_seed(412)
                initial = torch.randn(7, 5)
                xvalues = [torch.randn(2, 3, 7 if transposed else 5).to(dtype) for _ in range(6)]
                target = [torch.randn(2, 3, 5 if transposed else 7).to(dtype) for _ in range(6)]
                records = {}
                for arm in ("reference", "default", "off", "on"):
                    layer = Linear(5, 7, bias=False)
                    layer.weight.data.copy_(initial)
                    layer.weight_transposed = transposed
                    master_id = id(layer.weight)
                    momentum = torch.zeros_like(layer.weight)
                    rows = []
                    for step in range(3):
                        layer.weight.grad = None
                        exec(prime, {"step": step, "args": types.SimpleNamespace(initial_cache_refresh=arm == "on"),
                                     "cached_linears": [layer], "COMPUTE_DTYPE": dtype})
                        if step == 0:
                            require((layer._cached_weight is not None) == (arm == "on"), "exact step0 branch")
                        for micro in range(micro_count):
                            i = step * 2 + micro
                            x = xvalues[i].clone().requires_grad_(True)
                            if arm == "reference":
                                w = layer.weight.to(dtype)
                                output = x @ w if transposed else F.linear(x, w)
                            else:
                                output = layer(x)
                            loss = (output.float() - target[i].float()).square().sum() / micro_count
                            loss.backward()
                            rows.append((output.detach().clone(), x.grad.clone(), loss.detach().clone()))
                        require(layer.weight.grad is not None and layer.weight.grad.dtype == torch.float32,
                                "FP32 master receives gradient")
                        rows.append((layer.weight.grad.clone(),))
                        with torch.no_grad():
                            momentum.mul_(0.9).add_(layer.weight.grad)
                            layer.weight.add_(momentum, alpha=-0.002)
                        rows.append((layer.weight.detach().clone(), momentum.clone()))
                        layer.refresh_weight_cache(dtype)  # unchanged normal end-of-step refresh
                        require(id(layer.weight) == master_id, "master identity preserved")
                    require(list(layer.state_dict()) == ["weight"], "cache not persistent state")
                    payload = io.BytesIO()
                    torch.save(layer.state_dict(), payload)
                    payload.seek(0)
                    restored = Linear(5, 7, bias=False)
                    restored.load_state_dict(torch.load(payload, weights_only=True))
                    require(restored._cached_weight is None, "load has no stale serialized cache")
                    require(torch.equal(restored.weight, layer.weight), "weights-only state roundtrip")
                    records[arm] = rows
                for arm in ("default", "off", "on"):
                    for got, expected in zip(records[arm], records["reference"], strict=True):
                        for a, b in zip(got, expected, strict=True):
                            require(torch.equal(a, b), f"exact output/dx/dw/update/state {dtype}/{transposed}/{micro_count}/{arm}")
                cases.append({"dtype": str(dtype), "transposed": transposed, "micro_backwards": micro_count,
                              "steps": 3, "all_outputs_gradients_updates_state_exact": True})
    return {"torch": torch.__version__, "threads": torch.get_num_threads(), "cases": cases,
            "max_weight_elements": 35, "full_model_imported": False,
            "scope": "extracted Linear and exact new block vs independent functional reference; simple momentum, not production optimizer"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--tensor", action="store_true")
    args = parser.parse_args()
    linear, block = extract_source(args.repo)
    result = {"source_sha256": SOURCE_SHA, "raw_audit": receipts(args.repo)}
    result["tiny_reference"] = tiny_reference(linear, block) if args.tensor else "NOT_RUN"
    result.update(checks=checks, status="PASS", native="NOT_RUN", full117M_G2="PENDING", fresh_independent_session="NOT_RUN")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
