#!/usr/bin/env python3
"""G2 preflight dry-run: run the EXACT train.py bytes and launch argv on CPU (gloo, 4 ranks) for a
few real training steps, before any Trn2 host time is spent.

What it proves (per arm, in minutes, on any CPU box):
  * argparse accepts the argv, and the module's own `if __name__ == '__main__':` block runs
    (for 35c9 that is `_install_submission_training(); main()`, i.e. the same adapters as on Trn2);
  * model construction, the training-only installs, optimizer construction and the optimizer
    ownership plan succeed (this is where A1/A2 and --relu2-tau fail on exact 35c9);
  * `main()`'s real loop completes N optimizer steps (forward, backward, clip+all-reduce, step,
    weight-cache refresh, loss/control all-reduce), saves its checkpoint, and `load_for_eval`
    strict-loads that checkpoint and returns deterministic logits;
  * after the steps every parameter is byte-identical on all ranks (data-parallel consistency,
    incl. replicated tables), and every n-gram table actually changed.

What it deliberately does NOT prove: Neuron compilation, NKI kernels, BF16 device numerics,
throughput, cold-compile time, or BPB. Those stay with the Neuron smoke / full run gates.

CPU shims (all recorded in the receipt; none touches the bytes under test):
  * compute_dtype_for -> bfloat16 (the Neuron compute dtype; 35c9 asserts it);
  * `_submission_ce_install` -> no-op (it imports torch_neuronx NKI kernels);
  * `--device-type cpu --no-compile --num-steps N --out-dir <tmp>` appended to the argv;
  * GPT.forward sees only the first B x T tokens of each training micro-batch (default 2 x 256),
    so a CPU can run the real loop; every other code path is the real one;
  * a deterministic fake clock replaces `time` inside the module, so the wall-clock LR/WD
    schedules are reproducible and two files can be compared bit-for-bit (lockstep);
  * synthetic data/tokenizer by default (`--real-data` uses the organizer prepare.py loader).

Usage (launcher; spawns torchrun itself):
  python workflow/tools/preflight_dryrun.py --train-py path/to/train.py --out receipts/<label>.json \
      [--prepare-dir DIR | --stub-prepare] [--steps 1] [--nproc 4] -- <exact train.py argv>

Exit code: 0 PASS, 1 FAIL (arm is not launchable), 2 harness error/timeout.

Lockstep comparison of two PASS receipts (e.g. research file vs clean port, or 35c9 vs IMPL-1 with
the same argv): python workflow/tools/preflight_dryrun.py --compare A.json B.json
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
KNOWN_SIGNATURES = [
    ("ownership probe is restricted to AdamW and Muon", "OwnedOptimizer.make_plan rejects a non-adamw/muon group (n-gram tables: needs IMPL-1)"),
    ("ownership requires unique contiguous FP32 Parameters", "OwnedOptimizer.make_plan rejects a non-FP32 parameter (BF16 tables: needs IMPL-1)"),
    ("RELU2_TAU == 0.0", "MLP training adapter asserts RELU2_TAU == 0.0 (--relu2-tau unsupported)"),
    ("<= 24 * 1024 ** 2", "MLP training adapter SBUF budget assert (width/mlp-ratio too large)"),
    ("get_world_size() == 4", "training adapter requires world size 4 (LNC2 x 4)"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------
# launcher
# --------------------------------------------------------------------------------------------

def launcher(ns: argparse.Namespace, train_argv: list[str]) -> int:
    train_py = Path(ns.train_py).resolve()
    if not train_py.is_file():
        print(f"no such train.py: {train_py}", file=sys.stderr)
        return 2
    work = Path(tempfile.mkdtemp(prefix="g2-"))
    prepare_dir = HERE / "stub_prepare" if ns.stub_prepare else Path(ns.prepare_dir or train_py.parent).resolve()
    if not (prepare_dir / "prepare.py").is_file():
        print(f"prepare.py not found in {prepare_dir} (use --prepare-dir or --stub-prepare)", file=sys.stderr)
        return 2
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(prepare_dir), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.setdefault("OMP_NUM_THREADS", "1")
    env["G2_WORKDIR"] = str(work)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={ns.nproc}",
           str(Path(__file__).resolve()), "--worker", "--train-py", str(train_py), "--steps", str(ns.steps),
           "--micro-rows", str(ns.micro_rows), "--micro-tokens", str(ns.micro_tokens)]
    if ns.real_data:
        cmd.append("--real-data")
    if ns.trace_rmsprop:
        cmd.append("--trace-rmsprop")
    cmd += ["--"] + train_argv
    t0 = time.time()
    log_path = work / "torchrun.log"
    with open(log_path, "w") as log:
        try:
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=ns.timeout, start_new_session=True)
            rc, timed_out = proc.returncode, False
        except subprocess.TimeoutExpired:
            rc, timed_out = None, True
    ranks = []
    for r in range(ns.nproc):
        p = work / f"rank{r}.json"
        ranks.append(json.loads(p.read_text()) if p.exists() else {"rank": r, "status": "NO_REPORT"})
    receipt = aggregate(ns, train_py, prepare_dir, train_argv, ranks, rc, timed_out, time.time() - t0)
    tail = log_path.read_text(errors="replace").splitlines()[-60:]
    receipt["torchrun_log_tail"] = tail
    out = Path(ns.out) if ns.out else work / "receipt.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps({k: receipt[k] for k in ("verdict", "reason", "train_py_sha256", "plan_sha256", "elapsed_s")}, indent=2))
    print(f"receipt: {out}")
    if not ns.keep:
        shutil.rmtree(work, ignore_errors=True)
    if receipt["verdict"] == "PASS":
        return 0
    return 1 if receipt["verdict"] == "FAIL" else 2


def aggregate(ns, train_py, prepare_dir, train_argv, ranks, rc, timed_out, elapsed):
    statuses = [r.get("status") for r in ranks]
    r0 = next((r for r in ranks if r.get("rank") == 0), ranks[0])
    receipt = {
        "schema": "g2-preflight-v1",
        "train_py": str(train_py), "train_py_sha256": sha256_file(train_py),
        "argv": train_argv, "appended_argv": r0.get("appended_argv"),
        "prepare": {"dir": str(prepare_dir), "sha256": sha256_file(prepare_dir / "prepare.py"),
                    "stub": bool(ns.stub_prepare)},
        "steps": ns.steps, "micro_batch": [ns.micro_rows, ns.micro_tokens], "nproc": ns.nproc,
        "torch": r0.get("torch"), "python": platform.python_version(), "host": platform.node(),
        "shims": r0.get("shims"), "ranks": ranks, "torchrun_rc": rc, "timed_out": timed_out,
        "elapsed_s": round(elapsed, 1), "plan_sha256": r0.get("plan_sha256"),
    }
    if timed_out:
        verdict, reason = "ERROR", f"timeout after {ns.timeout}s"
    elif all(s == "PASS" for s in statuses):
        mism = r0.get("cross_rank_mismatches", [])
        stale = r0.get("unchanged_tables", [])
        ckpt = r0.get("checkpoint_check", {})
        if mism:
            verdict, reason = "FAIL", f"{len(mism)} parameters differ across ranks after the steps"
        elif stale and not ns.allow_unchanged_tables:
            verdict, reason = "FAIL", f"{len(stale)} n-gram tables did not change"
        elif not ckpt.get("ok"):
            verdict, reason = "FAIL", f"checkpoint/load_for_eval check failed: {ckpt.get('error')}"
        else:
            verdict, reason = "PASS", "all ranks completed the real training loop consistently"
    elif any(s == "FAIL" for s in statuses):
        f = next(r for r in ranks if r.get("status") == "FAIL")
        verdict, reason = "FAIL", f.get("classification") or f.get("error")
    else:
        verdict, reason = "ERROR", f"rank statuses {statuses}, torchrun rc={rc}"
    receipt["verdict"], receipt["reason"] = verdict, reason
    return receipt


# --------------------------------------------------------------------------------------------
# worker (one per rank, under torchrun)
# --------------------------------------------------------------------------------------------

class _PreflightStop(Exception):
    pass


class FakeClock(types.ModuleType):
    """Stand-in for `time` inside the module under test: deterministic monotonic clock."""

    def __init__(self, real, tick: float = 0.001):
        super().__init__("time")
        self._real, self._t, self._tick = real, 1000.0, tick

    def monotonic(self):
        self._t += self._tick
        return self._t

    perf_counter = monotonic

    def time(self):
        return self._real.time()

    def sleep(self, s):
        self._t += s

    def __getattr__(self, name):
        return getattr(self._real, name)


def param_digests(model) -> dict[str, str]:
    import torch
    out = {}
    for name, p in model.state_dict().items():
        t = p.detach().cpu().contiguous()
        if t.dtype == torch.bfloat16:
            t = t.view(torch.int16)
        out[name] = hashlib.sha256(memoryview(t.numpy()).cast("B")).hexdigest()[:16]  # zero-copy
    return out


def find_main_blocks(src: str):
    tree = ast.parse(src)
    blocks = []
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            t = node.test
            if (isinstance(t.left, ast.Name) and t.left.id == "__name__" and len(t.comparators) == 1
                    and isinstance(t.comparators[0], ast.Constant) and t.comparators[0].value == "__main__"):
                blocks.append(ast.Module(body=node.body, type_ignores=[]))
    return blocks


def classify(exc: BaseException, train_py: str) -> dict:
    tb = traceback.extract_tb(exc.__traceback__)
    frames = [f for f in tb if os.path.abspath(f.filename) == os.path.abspath(train_py)]
    inner = frames[-1] if frames else (tb[-1] if tb else None)
    text = f"{type(exc).__name__}: {exc}"
    line = inner.line if inner else ""
    label = None
    for needle, meaning in KNOWN_SIGNATURES:
        if needle in text or (line and needle in line):
            label = meaning
            break
    return {
        "error": text[:500],
        "train_py_frame": {"function": inner.name, "line": inner.lineno, "code": line} if inner else None,
        "classification": label or f"{type(exc).__name__} in {inner.name if inner else '?'}:{inner.lineno if inner else '?'}",
        "traceback_tail": traceback.format_exception(type(exc), exc, exc.__traceback__)[-8:],
    }


def worker(ns: argparse.Namespace, train_argv: list[str]) -> None:
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    work = Path(os.environ["G2_WORKDIR"])
    report = {"rank": rank, "status": "RUNNING", "torch": torch.__version__, "shims": []}
    out_dir = work / f"out_rank{rank}"
    appended = ["--device-type", "cpu", "--no-compile", "--num-steps", str(ns.steps), "--out-dir", str(work / "out")]
    report["appended_argv"] = appended

    def finish(status, **extra):
        report.update(status=status, **extra)
        (work / f"rank{rank}.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str))

    try:
        src = Path(ns.train_py).read_text()
        spec = importlib.util.spec_from_file_location("train_under_test", ns.train_py)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["train_under_test"] = mod
        spec.loader.exec_module(mod)
    except BaseException as e:  # import-time failure is an arm failure
        finish("FAIL", **classify(e, ns.train_py))
        os._exit(0)

    shims = report["shims"]
    # 1) compute dtype: mirror the Neuron BF16 compute dtype on CPU.
    if hasattr(mod, "compute_dtype_for"):
        mod.compute_dtype_for = lambda device: torch.bfloat16
        shims.append("compute_dtype_for -> torch.bfloat16")
    # 2) Neuron-only kernel installs.
    if hasattr(mod, "_submission_ce_install"):
        mod._submission_ce_install = lambda train_module: None
        shims.append("_submission_ce_install -> no-op (NKI CE kernels are Neuron-only)")
    # 3) deterministic clock inside the module.
    real_time = mod.time if hasattr(mod, "time") else time
    clock = FakeClock(real_time)
    mod.time = clock
    if hasattr(mod, "_PROCESS_STARTED"):
        mod._PROCESS_STARTED = clock.monotonic()
    shims.append("module `time` -> deterministic fake clock (1 ms per call)")
    # 4) data.
    if not ns.real_data:
        vocab = int(getattr(mod, "TOKENIZER_VOCAB_SIZE", 8192))

        class _Tok:
            def get_bos_token_id(self):
                return 0

            def get_vocab_size(self):
                return vocab

            def encode(self, docs, prepend=None, num_threads=1):
                return [[0] + [(7 * i + 3) % (vocab - 1) + 1 for i in range(64)] for _ in docs]

        def _loader(tokenizer, batch_size, seq_len, split, device, *a, **k):
            g = torch.Generator().manual_seed(1000 + rank)
            while True:
                x = torch.randint(1, vocab, (batch_size, seq_len), generator=g)
                y = torch.randint(1, vocab, (batch_size, seq_len), generator=g)
                yield x.to(torch.int32).to(device), y.to(device), {"epoch": 1}

        mod.ensure_tokenizer = lambda build_if_missing=False: _Tok()
        mod.make_dataloader = _loader
        if hasattr(mod, "make_dataloader_requeue"):
            mod.make_dataloader_requeue = _loader
        shims.append("synthetic tokenizer + synthetic dataloader (per-rank seeded)")
    # 5) shrink each training micro-batch seen by GPT.forward; capture model and losses.
    captured = {"model": None, "losses": [], "pre": None}
    orig_forward = mod.GPT.forward

    def forward(self, idx, targets=None, *a, **k):
        if self.training and targets is not None and idx.dim() == 2:
            if captured["model"] is None:
                captured["model"] = self
                captured["pre"] = param_digests(self)
            idx = idx[: ns.micro_rows, : ns.micro_tokens].contiguous()
            targets = targets[: ns.micro_rows, : ns.micro_tokens].contiguous()
            out = orig_forward(self, idx, targets, *a, **k)
            captured["losses"].append(float(out.detach().float()))
            return out
        return orig_forward(self, idx, targets, *a, **k)

    mod.GPT.forward = forward
    shims.append(f"GPT.forward training micro-batch sliced to {ns.micro_rows}x{ns.micro_tokens} tokens")
    # 6) capture the ownership plan (both 35c9 and the clean file define OwnedOptimizer).
    plan = {}
    if hasattr(mod, "OwnedOptimizer"):
        orig_init = mod.OwnedOptimizer.__init__

        def owned_init(self, *a, **k):
            orig_init(self, *a, **k)
            plan["entries"] = self.entries
            plan["summary"] = self.summary

        mod.OwnedOptimizer.__init__ = owned_init
    # 7) post-loop consistency checks run just before the module tears down the process group.
    checks = {}
    orig_cleanup = mod.cleanup_runtime

    def cleanup_runtime():
        m = captured["model"]
        if m is not None and dist.is_initialized():
            post = param_digests(m)
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, post)
            checks["cross_rank_mismatches"] = sorted(k for k in post if len({g[k] for g in gathered}) != 1)
            changed = {k for k in post if captured["pre"].get(k) != post[k]}
            tables = [k for k in post if "ngram_embeds" in k or "engram.tables" in k]
            checks["n_params"] = len(post)
            checks["n_changed"] = len(changed)
            checks["unchanged_params"] = sorted(set(post) - changed)[:50]
            checks["tables"] = tables
            checks["unchanged_tables"] = [k for k in tables if k not in changed]
            if rank == 0:
                checks["final_digests"] = post
        orig_cleanup()

    mod.cleanup_runtime = cleanup_runtime
    # 8) optional: trace the replicated table optimizer (IMPL-1 acceptance) -- count calls per
    #    optimizer step and compare every update against an independent float64 recomputation.
    trace = {"calls": 0, "checks": []}
    if ns.trace_rmsprop and hasattr(mod, "MuonAdamW") and hasattr(mod.MuonAdamW, "_rmsprop_step"):
        orig_rms = mod.MuonAdamW._rmsprop_step

        def traced(self, group):
            trace["calls"] += 1
            beta2, eps, lr = group["betas"][1], group["eps"], float(group["lr"])
            before = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state.get(p, {})
                v0 = st["exp_avg_sq"].detach().clone() if "exp_avg_sq" in st else None
                before.append((p, p.detach().clone(), p.grad.detach().clone(), v0, int(st.get("step", 0)) + 1))
            out = orig_rms(self, group)
            for p, p0, g, v0, t in before:
                rows = (g != 0).any(dim=-1) if p.dim() == 2 else (g != 0)
                idx = rows.nonzero().flatten()
                g64, p064 = g[idx].double(), p0[idx].double()
                v064 = v0[idx].double() if v0 is not None else torch.zeros_like(g64)
                v1 = beta2 * v064 + (1 - beta2) * g64 * g64
                exp = p064 - lr * g64 / ((v1 / (1 - beta2 ** t)).sqrt() + eps)
                err = (p.detach()[idx].double() - exp).abs()
                untouched_same = bool(torch.equal(p.detach()[~rows], p0[~rows]))
                nz = g64 != 0
                trace["checks"].append({"step_t": t, "numel": p.numel(), "rows_touched": int(idx.numel()),
                                        "max_abs_err_over_lr": float(err.max() / lr) if idx.numel() else 0.0,
                                        "untouched_rows_unchanged": untouched_same,
                                        "grad_abs_median_touched": float(g64.abs()[nz].median()) if nz.any() else 0.0})
            return out

        mod.MuonAdamW._rmsprop_step = traced
        shims.append("MuonAdamW._rmsprop_step traced (float64 recomputation of every table update)")
    sys.argv = [ns.train_py] + train_argv + appended
    try:
        for block in find_main_blocks(src):
            exec(compile(block, ns.train_py, "exec"), mod.__dict__)
    except SystemExit as e:
        if e.code not in (0, None):
            finish("FAIL", error=f"SystemExit({e.code}) from argparse/main", classification="argv rejected by train.py (parser.error)")
            os._exit(0)
    except BaseException as e:
        finish("FAIL", **classify(e, ns.train_py))
        os._exit(0)

    if plan:
        canon = json.dumps({"entries": plan["entries"], "summary": plan["summary"]}, sort_keys=True)
        report["plan_sha256"] = hashlib.sha256(canon.encode()).hexdigest()
        report["plan_summary"] = plan["summary"]
    report["losses"] = captured["losses"]
    if ns.trace_rmsprop:
        report["rmsprop_trace"] = {"calls": trace["calls"], "optimizer_steps": len(captured["losses"]), "checks": trace["checks"]}
    report.update(checks)
    if any(not (l == l) or abs(l) == float("inf") for l in captured["losses"]):
        finish("FAIL", classification="non-finite training loss")
        os._exit(0)
    # 8) checkpoint + load_for_eval (rank 0).
    if rank == 0:
        ck = {"ok": False}
        try:
            path = work / "out" / "final.pt"
            ck["path_exists"] = path.exists()
            ev = mod.load_for_eval(str(path), torch.device("cpu"))
            x = torch.randint(1, 8000, (1, min(ns.micro_tokens, 256)), generator=torch.Generator().manual_seed(7))
            with torch.no_grad():
                a, b = ev(x), ev(x)
            a = a[0] if isinstance(a, tuple) else a
            b = b[0] if isinstance(b, tuple) else b
            ck.update(ok=bool(torch.equal(a, b)) and bool(torch.isfinite(a.float()).all()),
                      deterministic=bool(torch.equal(a, b)), logits_shape=list(a.shape))
        except BaseException as e:
            ck.update(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
        report["checkpoint_check"] = ck
    finish("PASS")
    os._exit(0)


def compare(path_a: str, path_b: str) -> int:
    """Lockstep comparison of two receipts (same argv/steps/micro-batch): per-rank per-step losses and
    rank-0 final parameter digests must be bit-identical. Exit 0 identical, 1 different, 2 unusable."""
    ra, rb = (json.loads(Path(p).read_text()) for p in (path_a, path_b))
    if ra.get("verdict") != "PASS" or rb.get("verdict") != "PASS":
        print("both receipts must be PASS"); return 2
    if (ra["steps"], ra["micro_batch"]) != (rb["steps"], rb["micro_batch"]):
        print("steps/micro-batch differ; not comparable"); return 2
    la = {r["rank"]: r.get("losses") for r in ra["ranks"]}
    lb = {r["rank"]: r.get("losses") for r in rb["ranks"]}
    da = next(r for r in ra["ranks"] if r["rank"] == 0).get("final_digests", {})
    db = next(r for r in rb["ranks"] if r["rank"] == 0).get("final_digests", {})
    loss_diff = {k: (la[k], lb.get(k)) for k in la if la[k] != lb.get(k)}
    keys_only = sorted(set(da) ^ set(db))
    dig_diff = sorted(k for k in set(da) & set(db) if da[k] != db[k])
    result = {"a": path_a, "b": path_b, "a_sha256": ra["train_py_sha256"], "b_sha256": rb["train_py_sha256"],
              "plan_equal": ra.get("plan_sha256") == rb.get("plan_sha256"),
              "losses_identical": not loss_diff, "loss_differences": loss_diff,
              "state_keys_only_in_one": keys_only, "param_digest_differences": dig_diff,
              "n_params_compared": len(set(da) & set(db))}
    result["identical"] = result["plan_equal"] and not loss_diff and not keys_only and not dig_diff
    print(json.dumps(result, indent=2))
    return 0 if result["identical"] else 1


def main() -> int:
    argv = sys.argv[1:]
    if argv[:1] == ["--compare"] and len(argv) == 3:
        return compare(argv[1], argv[2])
    if "--" in argv:
        i = argv.index("--")
        own, train_argv = argv[:i], argv[i + 1:]
    else:
        own, train_argv = argv, []
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--train-py", required=True)
    p.add_argument("--prepare-dir", default="")
    p.add_argument("--stub-prepare", action="store_true")
    p.add_argument("--real-data", action="store_true")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--nproc", type=int, default=4)
    p.add_argument("--micro-rows", type=int, default=2)
    p.add_argument("--micro-tokens", type=int, default=256)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--out", default="")
    p.add_argument("--keep", action="store_true")
    p.add_argument("--trace-rmsprop", action="store_true")
    p.add_argument("--allow-unchanged-tables", action="store_true", help="expected when testing a table freeze")
    p.add_argument("--worker", action="store_true")
    ns = p.parse_args(own)
    if ns.worker:
        worker(ns, train_argv)
        return 0
    return launcher(ns, train_argv)


if __name__ == "__main__":
    sys.exit(main())
