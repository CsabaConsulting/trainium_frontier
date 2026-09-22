#!/usr/bin/env python3
"""CPU run-time equivalence of the lean submission train.py against the verified original.

Nothing here touches Trainium. A synthetic data cache (six tiny parquet shards + the fixed
tokenizer artifact) is created under a temporary NEURON_COMPETITION_R1_CACHE_DIR, then:

  1. main() of both files runs on CPU (`--device-type cpu --no-compile --num-steps 3`, small
     width, the launch command's feature flags) in separate processes; final.pt tensors and
     the logged per-step losses must be bit-identical.
  2. load_for_eval() of both files on both checkpoints must produce identical logits.
  3. The fused-MLP training adapter (make_forward/make_autograd, CPU reference kernel path)
     must give identical outputs and gradients in bf16.
  4. The optimizer-ownership adapter must give identical parameters after 3 steps on a
     2-rank gloo group (torchrun).
  5. Neuron-only glue that cannot run here must be AST-identical to the original.

Usage: /tmp/tfvenv/bin/python research/scratch/lean_submission_cpu_equiv.py [--lean PATH] [--steps N]
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORIG = ROOT / "submissions/dense6x1024_public0992851/train.py"
LEAN_DEFAULT = ROOT / "submissions/dense6x1024_lean/train.py"
LAUNCH_FLAGS = ["--no-ngram-ve", "--no-qk-shift", "--compile-sdpa-direct", "--nki-local-conv",
                "--bf16-norm-output", "--clip-after-reduce", "--no-eval-public"]
NKI_GLUE = ["_RopeNormNKI", "_SoftcapCrossEntropyNKI", "_Conv2NKI", "_conv2_taps", "_conv2_fwd_tile", "_conv2_dw_partial",
            "_conv2_bwd_tile", "selected_lanes", "build_kernels", "_SubmissionEvalGPT", "reference_forward"]
# _submission_ce_install is Neuron-only glue whose frozen-constant guards were folded; reviewed by diff, not identity.


def make_cache(cache: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    data = cache / "data"
    data.mkdir(parents=True)
    rng = random.Random(20260922)
    syllables = ["ka", "to", "mi", "ren", "sol", "vex", "una", "dor", "il", "pha", "ny", "que", "ost", "bri", "zel", "am"]
    punct = [".", ".", ".", ",", ";", "!", "?"]

    def doc() -> str:
        words = []
        for _ in range(rng.randint(20, 400)):
            w = "".join(rng.choice(syllables) for _ in range(rng.randint(1, 4)))
            if rng.random() < 0.08:
                w = w.capitalize()
            words.append(w + (rng.choice(punct) if rng.random() < 0.12 else ""))
        return " ".join(words)

    for shard in [0, 1, 2, 3, 4, 6542]:
        rows = [doc() for _ in range(400)]
        table = pa.table({"text": rows})
        pq.write_table(table, data / f"shard_{shard:05d}.parquet", row_group_size=100)


def run_main(train_py: Path, out_dir: Path, env: dict, steps: int, n_embd: int, seq_len: int, pack: int) -> subprocess.Popen:
    cmd = [sys.executable, str(ROOT / "research/scratch/_run_train_main.py"), str(train_py),
           "--device-type", "cpu", "--no-compile", "--num-steps", str(steps), "--max-train-seconds", "1800",
           "--n-embd", str(n_embd), "--seq-len", str(seq_len), "--pack-factor", str(pack),
           "--out-dir", str(out_dir), "--seed", "42", *LAUNCH_FLAGS]
    log = open(out_dir.parent / (out_dir.name + ".log"), "w")
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)


def load_module(path: Path, name: str, frozen: dict | None = None):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    for k, v in (frozen or {}).items():  # original module -> launch configuration (see worker)
        if hasattr(mod, k) and not isinstance(v, (list, dict)):
            setattr(mod, k, v)
    return mod


def step_losses(log: Path) -> list[str]:
    return re.findall(r"^step \d+ \| loss [^|]+\|", log.read_text(), re.M)


def compare_state(a: dict, b: dict, torch) -> list[str]:
    bad = []
    if set(a) != set(b):
        bad.append(f"key sets differ: only_a={sorted(set(a) - set(b))[:5]} only_b={sorted(set(b) - set(a))[:5]}")
    for k in sorted(set(a) & set(b)):
        if a[k].dtype != b[k].dtype or a[k].shape != b[k].shape or not torch.equal(a[k], b[k]):
            bad.append(k)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean", type=Path, default=LEAN_DEFAULT)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--pack-factor", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--reuse", type=Path, default=None, help="existing tmp dir with orig_out/final.pt and lean_out/final.pt: skip training")
    ap.add_argument("--report", type=Path, default=None, help="build-report.json of the lean file (default: next to it)")
    a = ap.parse_args()
    report_path = a.report or a.lean.with_name("build-report.json")
    frozen = json.loads(report_path.read_text())["globals_frozen"]
    report: dict = dict(original=str(ORIG), lean=str(a.lean), steps=a.steps, n_embd=a.n_embd, seq_len=a.seq_len, pack_factor=a.pack_factor,
                        clock="frozen (time.monotonic constant) so wall-clock schedules are identical in both runs")
    tmp = a.reuse or Path(tempfile.mkdtemp(prefix="lean_equiv_"))
    cache = tmp / "cache"
    if not a.reuse:
        make_cache(cache)
    env = dict(os.environ, NEURON_COMPETITION_R1_CACHE_DIR=str(cache), OMP_NUM_THREADS=str(a.threads),
               MKL_NUM_THREADS=str(a.threads), PYTHONUNBUFFERED="1", PYTHONPATH=str(ROOT))
    os.environ.update(NEURON_COMPETITION_R1_CACHE_DIR=str(cache))
    sys.path.insert(0, str(ROOT))
    import torch
    torch.set_num_threads(a.threads)

    # 1. training on CPU, both files in parallel
    out_o, out_l = tmp / "orig_out", tmp / "lean_out"
    if a.reuse:
        rcs = [0 if (out_o / "final.pt").exists() else 1, 0 if (out_l / "final.pt").exists() else 1]
    else:
        out_o.mkdir(); out_l.mkdir()
        procs = [run_main(ORIG, out_o, env, a.steps, a.n_embd, a.seq_len, a.pack_factor),
                 run_main(a.lean, out_l, env, a.steps, a.n_embd, a.seq_len, a.pack_factor)]
        rcs = [p.wait() for p in procs]
    report["train_exit_codes"] = rcs
    lo, ll = step_losses(tmp / "orig_out.log"), step_losses(tmp / "lean_out.log")
    report["train_step_lines_original"] = lo
    report["train_step_lines_lean"] = ll
    report["train_step_lines_equal"] = lo == ll and len(lo) >= a.steps
    ok = all(rc == 0 for rc in rcs) and report["train_step_lines_equal"]
    if all(rc == 0 for rc in rcs):
        po = torch.load(out_o / "final.pt", map_location="cpu", weights_only=True)
        pl = torch.load(out_l / "final.pt", map_location="cpu", weights_only=True)
        bad = compare_state(po["model_state"], pl["model_state"], torch)
        report["final_pt_model_state_mismatches"] = bad
        report["final_pt_model_config_equal"] = po["model_config"] == pl["model_config"]
        report["final_pt_meta_original"] = po["meta"]
        report["final_pt_meta_lean"] = pl["meta"]
        report["final_pt_toggles_lean_subset_of_original"] = all(po["toggles"].get(k) == v for k, v in pl["toggles"].items())
        report["final_pt_toggles_dropped"] = sorted(set(po["toggles"]) - set(pl["toggles"]))
        ok = ok and not bad and report["final_pt_model_config_equal"] and report["final_pt_toggles_lean_subset_of_original"]

        # 2. load_for_eval on both checkpoints with both modules.
        # CPU-only artefact of the ORIGINAL code (and hence the lean one): with fp32 compute,
        # `Linear.refresh_weight_cache` stores `weight.to(fp32)`, which is the Parameter itself, so
        # the alias is registered as a parameter and lands in state_dict. On Neuron the cache is a
        # bf16 copy and never appears. Strip those keys into harness-only copies before loading.
        cached_keys = sorted(k for k in po["model_state"] if k.endswith("._cached_weight"))
        report["cpu_only_cached_weight_keys_stripped"] = len(cached_keys)
        for pth, payload in ((out_o / "final.pt", po), (out_l / "final.pt", pl)):
            payload2 = dict(payload)
            payload2["model_state"] = {k: v for k, v in payload["model_state"].items() if not k.endswith("._cached_weight")}
            torch.save(payload2, pth.with_name("final.eval.pt"))
        orig = load_module(ORIG, "train_orig", frozen)
        lean = load_module(a.lean, "train_lean")
        g = torch.Generator().manual_seed(7)
        idx = torch.randint(0, orig.TOKENIZER_VOCAB_SIZE, (3, 96), generator=g)
        # The scorer imports the lean train.py and loads (a) the checkpoint that file trained and, for
        # the public replay gate, (b) the checkpoint the ORIGINAL file trained. Both must load and
        # agree with the original module on its own checkpoint. The fourth combination (original
        # module + lean checkpoint) is nobody's contract: the lean checkpoint's `toggles` are a
        # subset, and the original module's defaults for the missing names differ (USE_NGRAM_VE).
        logits = {}
        with torch.no_grad():
            for mname, mod, cname, ck in (("orig", orig, "orig", out_o / "final.eval.pt"),
                                          ("lean", lean, "lean", out_l / "final.eval.pt"),
                                          ("lean", lean, "orig", out_o / "final.eval.pt")):
                m = mod.load_for_eval(str(ck), torch.device("cpu"))
                logits[f"{mname}_module/{cname}_ckpt"] = m(idx).float()
        ref = logits["orig_module/orig_ckpt"]
        report["load_for_eval_logits_equal"] = {k: bool(torch.equal(v, ref)) for k, v in logits.items()}
        report["load_for_eval_logits_max_abs_diff"] = {k: float((v - ref).abs().max()) for k, v in logits.items()}
        ok = ok and all(report["load_for_eval_logits_equal"].values())
        try:
            orig.load_for_eval(str(out_l / "final.eval.pt"), torch.device("cpu"))
            report["orig_module_loads_lean_ckpt"] = "loads (not required)"
        except Exception as exc:
            report["orig_module_loads_lean_ckpt"] = f"not loadable, as expected (toggle subset): {str(exc)[:80]}"

        # 3. fused-MLP adapter, CPU reference kernel path, bf16
        def mlp_adapter(mod, kwargs):
            torch.manual_seed(3)
            cfg = mod.GPTConfig(sequence_len=64, vocab_size=mod.TOKENIZER_VOCAB_SIZE, n_layer=1, n_head=1, n_embd=128)
            model = mod.GPT(cfg)
            model.init_weights()
            mod._submission_mlp_install(model, "cpu", mod, **kwargs)
            mlp = model.transformer.h[0].mlp
            mlp.train()
            for lin in (mlp.c_fc, mlp.c_proj):
                lin.refresh_weight_cache(torch.bfloat16)
            x = torch.randn(2, 512, 128, generator=torch.Generator().manual_seed(5)).to(torch.bfloat16).requires_grad_(True)
            y = mlp(x)
            y.float().square().mean().backward()
            return y.detach().clone(), x.grad.clone(), mlp.c_fc.weight.grad.clone(), mlp.c_proj.weight.grad.clone()
        ra = mlp_adapter(orig, dict(forward_backend="native"))
        rb = mlp_adapter(lean, {})
        report["mlp_adapter_equal"] = [bool(torch.equal(p, q)) for p, q in zip(ra, rb)]
        ok = ok and all(report["mlp_adapter_equal"])

        # 4. optimizer ownership on a 2-rank gloo group
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
               str(ROOT / "research/scratch/_owned_optimizer_equiv.py"), str(ORIG), str(a.lean), str(report_path)]
        r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=900)
        report["owned_optimizer_exit_code"] = r.returncode
        report["owned_optimizer_output"] = [l for l in (r.stdout + r.stderr).splitlines() if "OWNED_OPTIMIZER_EQUIV" in l or "Error" in l][:10]
        ok = ok and r.returncode == 0

    # 5. Neuron-only glue is AST-identical
    def defs(path):
        out = {}
        for n in ast.walk(ast.parse(path.read_text())):
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                out.setdefault(n.name, ast.dump(n))
        return out
    do, dl = defs(ORIG), defs(a.lean)
    report["nki_glue_identical"] = {n: (do.get(n) == dl.get(n)) for n in NKI_GLUE}
    report["nki_glue_missing_in_lean"] = [n for n in NKI_GLUE if n not in dl]
    ok = ok and all(report["nki_glue_identical"].values())

    report["all_passed"] = ok
    report["tmp_dir"] = str(tmp)
    print(json.dumps(report, indent=1, default=str))
    (tmp / "cpu-equivalence.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    if not a.keep and ok:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(2)
