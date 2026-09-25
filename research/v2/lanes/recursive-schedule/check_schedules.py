"""Bounded schedule and actual optimizer-loop assignment checks; no data/hardware."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "workflow/proposals/research-v2/train.py"
BASE = ROOT / "submissions/research_v2/train.py"
HELPERS = {"lr_multiplier", "muon_momentum", "muon_beta2", "demon_beta1_scale"}

def extract(path):
    tree = ast.parse(path.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in HELPERS:
            nodes.append(node)
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns, tree

def close(a, b):
    assert abs(a - b) < 1e-14, (a, b)

def run():
    old, _ = extract(BASE)
    ns, tree = extract(SOURCE)
    assert not ns["RECURSIVE_SCHEDULE"]
    points = [-1.0, 0.0, 0.05, 0.35, 0.5, 0.955, 1.0, 1.2]
    for step in [0, 1, 2, 150, 299, 300, 3600]:
        for p in points:
            for name, args in [("lr_multiplier", (step, p * 1800, 1800)),
                               ("muon_momentum", (step, p)), ("muon_beta2", (p,))]:
                assert ns[name](*args) == old[name](*args), (name, step, p)
    ns["RECURSIVE_SCHEDULE"] = True
    rows = []
    for p in points:
        clipped = min(1.0, max(0.0, p))
        frac = max(0.0, (clipped - 0.05) / 0.95)
        muon = ns["lr_multiplier"](3600, p * 1800, 1800, "muon")
        adam = ns["lr_multiplier"](3600, p * 1800, 1800, "adamw")
        momentum = ns["muon_momentum"](3600, p)
        beta2 = ns["muon_beta2"](p)
        close(muon, 1.0 - 0.95 * frac)
        close(adam, 1.0 - 0.95 * max(0.0, (clipped - 0.35) / 0.65))
        close(momentum, 0.95 - 0.16 * frac ** 2)
        close(beta2, 0.95 + 0.02 * frac)
        rows.append(dict(progress=p, muon_lr=muon, adam_lr=adam,
                         muon_momentum=momentum, muon_beta2=beta2))
    for step in [0, 150, 300]:
        close(ns["muon_momentum"](step, 0), 0.85 + 0.10 * min(step / 300, 1))
    # Execute the actual production schedule/param-group block, including flat-table priority.
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loops = [n for n in ast.walk(main) if isinstance(n, ast.While)]
    body = next(n.body for n in loops if any(isinstance(x, ast.Assign) and
        any(isinstance(t, ast.Name) and t.id == "lrm" for t in x.targets) for x in n.body))
    start = next(i for i,n in enumerate(body) if isinstance(n, ast.Assign) and
        any(isinstance(t, ast.Name) and t.id == "lrm" for t in n.targets))
    end = next(i for i in range(start, len(body)) if isinstance(body[i], ast.For) and
        isinstance(body[i].target, ast.Name) and body[i].target.id == "group")
    block = compile(ast.Module(body=body[start:end+1], type_ignores=[]), str(SOURCE), "exec")
    from types import SimpleNamespace
    for enabled in [False, True]:
        for p in [0, 0.05, 0.35, 0.5, 1.0]:
            groups = [dict(kind=k, initial_lr=0.30, no_lr_decay=flat)
                      for k, flat in [("muon", False), ("adamw", False),
                                      ("rmsprop", True), ("adamw", True)]]
            ns.update(RECURSIVE_SCHEDULE=enabled, optimizer=SimpleNamespace(param_groups=groups),
                      step=3600, elapsed=p*1800, args=SimpleNamespace(max_train_seconds=1800),
                      batch_lr_scale=1.3)
            exec(block, ns)
            for group in groups:
                expected = 1.0 if group["no_lr_decay"] else ns["lr_multiplier"](
                    3600, p*1800, 1800, group["kind"])
                close(group["lr"], 0.30 * 1.3 * expected)
            assert groups[2]["lr"] == groups[3]["lr"] == 0.30 * 1.3
    # Compiled body must see the existing dynamic scalar for this new ramp.
    step_fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_muon_step")
    expr = next(n.value for n in ast.walk(step_fn) if isinstance(n, ast.Assign) and
        any(isinstance(t, ast.Name) and t.id == "one_beta2" for t in n.targets))
    sentinel = object()
    dynamic = eval(compile(ast.Expression(expr), str(SOURCE), "eval"),
                   dict(RECURSIVE_SCHEDULE=True, MUON_BETA2_FINAL=0.0,
                        one_b2_t=sentinel, group={"beta2": .95}))
    assert dynamic is sentinel
    return dict(verdict="PASS", base_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
                candidate_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                checks=["off path exact for all tested steps/progress", "source schedule values",
                        "actual group assignment incl flat RMSProp/Adam and batch scale",
                        "compiled Muon uses dynamic tensor scalar"], schedule=rows)

if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    (Path(__file__).parent / "schedule_checks.json").write_text(json.dumps(result, indent=2) + "\n")
