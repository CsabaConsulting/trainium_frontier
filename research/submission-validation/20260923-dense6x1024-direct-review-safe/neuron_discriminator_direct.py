#!/usr/bin/env python3
"""Fail-closed source-versus-direct Neuron discriminator.

The top-level process authenticates an externally finalized request and then runs
each implementation in a different process and compile cache.  Worker entry
points are private implementation details; they require a parent nonce and are
never an authorization boundary.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
HARNESS = Path(__file__).resolve()
SOURCE = ROOT / "submissions/dense6x1024_lean_review/train.py"
DIRECT = ROOT / "submissions/dense6x1024_direct_review/train.py"
PROPOSAL = EVIDENCE / "proposal.json"
REQUEST = EVIDENCE / "neuron-discriminator-authorization-request.json"
RUNBOOK = EVIDENCE / "neuron-discriminator-runbook.md"
SOURCE_SHA256 = "9173f62617789b058083c3a4722a906e2b165f4986716281f1915db3fb9fa344"
DIRECT_SHA256 = "c5956187de3302da4a424b17686768c8239606701e0d4c690042944cbf847537"
AUTH_SCHEMA = "dense6x1024-direct-neuron-discriminator-authorization-v1"
REPORT_SCHEMA = "dense6x1024-direct-neuron-discriminator-report-v1"
WORKER_SCHEMA = "dense6x1024-direct-neuron-discriminator-worker-v1"
PENDING_HEAD = "PENDING_EXTERNAL_VERIFIER_HEAD"
PENDING_SHA = "PENDING_EXTERNAL_VERIFIER_REQUEST_SHA256"
ROWS = (128, 512, 32768)
REPETITIONS = 2
WORLD_SIZE = 4
SEED = 42
HOST_THREADS = 4
OUTER_TIMEOUT = ["timeout", "--foreground", "--signal=TERM", "--kill-after=60s", "5400s"]
NATIVE_PYTHON = "/home/ubuntu/workspace/native_venv/bin/python"
TORCHRUN = "/home/ubuntu/workspace/native_venv/bin/torchrun"
NEURON_CC_FLAGS = "--optlevel=1 --auto-cast matmult --auto-cast-type bf16"
EXPECTED_HOSTNAME = "ip-172-31-32-253"
EXPECTED_HOST = "trn22"
EXPECTED_REMOTE = "/home/ubuntu/trainium_freshstart_submission_validation_20260923_direct_discriminator"
EXPECTED_PURPOSE = (
    "Source/direct Neuron component discriminator only; no training, submission, or scientific verdict."
)
STATIC_ENVIRONMENT = {
    "PATH": "/home/ubuntu/workspace/native_venv/bin:/usr/local/bin:/usr/bin:/bin:/opt/aws/neuron/bin",
    "NKI_VENV_PATH": "/home/ubuntu/workspace/native_venv",
    "NEURON_LOGICAL_NC_CONFIG": "2",
    "NEURON_CC_FLAGS": NEURON_CC_FLAGS,
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "PYTHONUNBUFFERED": "1",
    "TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION": "1",
}
RECOMPILE_PATTERNS = (
    re.compile(r"recompil(?:e|ation|ing).*(?:limit|too many|exceed)", re.I),
    re.compile(r"(?:limit|too many|exceed).*recompil", re.I),
)
PROTECTED_PATHS = (
    "research/fresh_start_guard.py",
    "research/fresh_start_host.py",
    "research/fresh_start_launch.py",
    "research/fresh_start_sync.py",
    "research/run_experiment.py",
    "research/scratch/optimizer_ownership_check.py",
    "research/run-logs/fs-w62-optimizer-ownership-padded-neuron-gate-r1.job.json",
    "submissions/dense6x1024_lean_review/train.py",
    "submissions/dense6x1024_direct_review/train.py",
    "research/submission-validation/20260923-dense6x1024-direct-review-safe/cpu_equivalence_direct.py",
    "research/submission-validation/20260923-dense6x1024-direct-review-safe/neuron_discriminator_direct.py",
    "research/submission-validation/20260923-dense6x1024-direct-review-safe/test_neuron_discriminator_static.py",
    "research/submission-validation/20260923-dense6x1024-direct-review-safe/neuron-discriminator-runbook.md",
    "research/submission-validation/20260923-dense6x1024-direct-review-safe/proposal.json",
    "research/submission-validation/20260923-dense6x1024-direct-review-safe/fs-20260923-dense6x1024-direct-neuron-component-20260923T103500000001Z-result.json",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def exclusive_json(path: Path, value: Any) -> None:
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"refusing to replace result: {path}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(json_bytes(value))
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def static_source_checks() -> dict[str, Any]:
    source_text = SOURCE.read_text()
    direct_text = DIRECT.read_text()
    ast.parse(source_text, filename=str(SOURCE))
    ast.parse(direct_text, filename=str(DIRECT))
    ast.parse(HARNESS.read_text(), filename=str(HARNESS))
    checks = {
        "source_sha256": sha256(SOURCE) == SOURCE_SHA256,
        "direct_sha256": sha256(DIRECT) == DIRECT_SHA256,
        "source_ce_adapter": "def _submission_ce_install(" in source_text,
        "source_optimizer_adapter": "def _submission_optimizer_install(" in source_text,
        "direct_native_ce": "class _SoftcapCrossEntropyNKI" in direct_text,
        "direct_explicit_owner": "owned_optimizer = OwnedOptimizer(optimizer)" in direct_text,
        "same_lane_rule": source_text[source_text.index("def selected_lanes(rows):"):].split("\n\n", 1)[0]
        == direct_text[direct_text.index("def selected_lanes(rows):"):].split("\n", 4)[0]
        + "\n" + "\n".join(direct_text[direct_text.index("def selected_lanes(rows):"):].split("\n", 4)[1:4]),
        "ce_rows_frozen": ROWS == (128, 512, 32768),
        "ce_repetitions_frozen": REPETITIONS == 2,
        "world_size_frozen": WORLD_SIZE == 4,
        "seed_frozen": SEED == 42,
    }
    # The textual comparison above is intentionally supplemented by semantic
    # pins because comments/EOF placement are not a numerical contract.
    checks["lane_rule_source"] = (
        "return 2 if rows >= 512 and rows % 256 == 0 else 1" in source_text
    )
    checks["lane_rule_direct"] = (
        "return 2 if rows >= 512 and rows % 256 == 0 else 1" in direct_text
    )
    # Do not make harmless formatting a gate.
    checks.pop("same_lane_rule")
    return {"passed": all(checks.values()), "checks": checks}


def expected_contract() -> dict[str, Any]:
    return {
        "seed": SEED,
        "host": EXPECTED_HOST,
        "expected_hostname": EXPECTED_HOSTNAME,
        "remote": EXPECTED_REMOTE,
        "host_threads": HOST_THREADS,
        "world_size": WORLD_SIZE,
        "logical_nc_config": "2",
        "source_path": rel(SOURCE),
        "source_sha256": SOURCE_SHA256,
        "direct_path": rel(DIRECT),
        "direct_sha256": DIRECT_SHA256,
        "ce_rows": list(ROWS),
        "ce_repetitions": REPETITIONS,
        "optimizer_fixtures": ["tiny_fixed", "tiny_changed", "production_width1024_fs_w62"],
        "outer_timeout_argv": OUTER_TIMEOUT,
        "native_python": NATIVE_PYTHON,
        "torchrun_argv_prefix": [TORCHRUN, "--standalone", "--nproc_per_node=4"],
        "environment_static": STATIC_ENVIRONMENT,
        "recompile_limit_warning_policy": "invalidate",
        "worker_isolation": "one variant per process; one fresh compile cache per variant",
    }


def authenticate(args: argparse.Namespace, full: bool) -> dict[str, Any]:
    if args.authorization_request.resolve() != REQUEST:
        raise ValueError(f"authorization request must be exactly {REQUEST}")
    request = json.loads(REQUEST.read_text())
    if request.get("schema") != AUTH_SCHEMA:
        raise ValueError("authorization schema mismatch")
    if request.get("request_contract") != expected_contract():
        raise ValueError("authorization contract differs from hard-coded harness contract")
    actual_request_sha = sha256(REQUEST)
    if args.static_only and args.authorized_head == PENDING_HEAD and args.authorization_sha256 == PENDING_SHA:
        if request.get("authorization_state") != "pending_external_verifier":
            raise ValueError("sentinel is only valid for pending static review")
        full = False
    else:
        if not re.fullmatch(r"[0-9a-f]{40}", args.authorized_head):
            raise ValueError("--authorized-head must be an external literal 40-hex commit")
        if not re.fullmatch(r"[0-9a-f]{64}", args.authorization_sha256):
            raise ValueError("--authorization-sha256 must be an external literal SHA-256")
        if args.authorization_sha256 != actual_request_sha:
            raise ValueError("external authorization SHA does not match request bytes")
        if request.get("authorization_state") != "finalized_by_external_verifier":
            raise ValueError("pending authorization request cannot launch")
        if request.get("authorized_head") != args.authorized_head:
            raise ValueError("request/external authorized HEAD mismatch")
    expected_hashes = request.get("expected_sha256")
    if not isinstance(expected_hashes, dict) or not expected_hashes:
        raise ValueError("authorization expected_sha256 is missing")
    if tuple(expected_hashes) != PROTECTED_PATHS:
        raise ValueError("authorization protected path set/order differs from hard-coded harness")
    actual_hashes = {}
    for name, expected in expected_hashes.items():
        path = ROOT / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"fingerprint target is not a regular file: {name}")
        actual_hashes[name] = sha256(path)
        if actual_hashes[name] != expected:
            raise ValueError(f"fingerprint mismatch: {name}")
    if sha256(SOURCE) != SOURCE_SHA256 or sha256(DIRECT) != DIRECT_SHA256:
        raise ValueError("frozen source/direct hash mismatch")
    head = git("rev-parse", "HEAD")
    if not args.static_only and head != args.authorized_head:
        raise ValueError("checkout is not the exact externally authorized commit")
    dirty = git("status", "--porcelain=v1", "--untracked-files=all", "--", *expected_hashes)
    if dirty:
        raise ValueError("fingerprinted paths are dirty/untracked")
    if full and request.get("authorized_head") != head:
        raise ValueError("full run is not externally authorized")
    return {
        "request_sha256": actual_request_sha,
        "authorized_head": args.authorized_head,
        "head": head,
        "expected_sha256": expected_hashes,
        "actual_sha256": actual_hashes,
        "state": request.get("authorization_state"),
    }


def import_train(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tensor_digest(torch, tensor) -> dict[str, Any]:
    value = tensor.detach().contiguous().cpu()
    raw = value.view(torch.uint8).numpy().tobytes()
    finite = bool(torch.isfinite(value).all()) if value.is_floating_point() else True
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": finite,
        "nonzero": bool(torch.count_nonzero(value)),
    }


def ce_fixture(torch, rows: int, repetition: int, device):
    generator = torch.Generator().manual_seed(SEED + rows * 17 + repetition)
    logits = torch.randn(rows, 8192, generator=generator, dtype=torch.float32)
    targets = torch.randint(0, 8192, (rows,), generator=generator, dtype=torch.int64)
    upstream = torch.randn(rows, generator=generator, dtype=torch.float32)
    return logits.to(device).requires_grad_(True), targets.to(device), upstream.to(device)


def run_ce_worker(args: argparse.Namespace) -> int:
    if os.environ.get("TRAINIUM_DISCRIMINATOR_PARENT_NONCE") != args.parent_nonce:
        raise RuntimeError("CE worker lacks its authenticated parent nonce")
    import torch
    import torch.nn.functional as F
    import torch_neuronx  # noqa: F401

    torch.set_num_threads(HOST_THREADS)
    module = import_train(SOURCE if args.variant == "source" else DIRECT,
                          f"discriminator_{args.variant}_{os.getpid()}")
    module._HAS_NKI = True
    device = torch.device("neuron")
    calls: list[dict[str, Any]] = []
    if args.variant == "source":
        module._submission_ce_install(module)
        original_forward, original_backward = module._softcap_ce_fwd, module._softcap_ce_bwd

        def observed_forward(z, target):
            calls.append({"op": "forward", "rows": z.shape[0], "lanes": module.selected_lanes(z.shape[0])})
            return original_forward(z, target)

        def observed_backward(z, target, lse, gradient):
            calls.append({"op": "backward", "rows": z.shape[0], "lanes": module.selected_lanes(z.shape[0])})
            return original_backward(z, target, lse, gradient)

        module._softcap_ce_fwd = observed_forward
        module._softcap_ce_bwd = observed_backward
    else:
        class Probe:
            def __init__(self, wrapped, operation):
                self.wrapped, self.operation = wrapped, operation

            def __getitem__(self, grid):
                launch = self.wrapped[grid]

                def call(*values):
                    rows = values[0].shape[0]
                    calls.append({"op": self.operation, "rows": rows, "lanes": grid})
                    return launch(*values)
                return call

        module._softcap_ce_fwd_native = Probe(module._softcap_ce_fwd_native, "forward")
        module._softcap_ce_bwd_native = Probe(module._softcap_ce_bwd_native, "backward")

    source_payload = None
    source_pt = args.result_dir / "ce-source.pt"
    if args.variant == "direct":
        source_payload = torch.load(source_pt, map_location="cpu", weights_only=True)
    saved: dict[str, Any] = {}
    cases = []
    for rows in ROWS:
        for repetition in range(REPETITIONS):
            key = f"r{rows}-n{repetition}"
            logits, targets, upstream = ce_fixture(torch, rows, repetition, device)
            loss = module.softcap_cross_entropy(logits, targets, "none")
            loss.backward(upstream)
            module.synchronize(device)
            loss_cpu = loss.detach().cpu()
            grad_cpu = logits.grad.detach().cpu()
            record: dict[str, Any] = {
                "key": key,
                "rows": rows,
                "repetition": repetition,
                "selected_lanes": module.selected_lanes(rows),
                "loss": tensor_digest(torch, loss_cpu),
                "gradient": tensor_digest(torch, grad_cpu),
                "autograd_node": type(loss.grad_fn).__name__,
            }
            if args.variant == "source":
                saved[key] = {"loss": loss_cpu, "gradient": grad_cpu}
            else:
                expected = source_payload[key]
                loss_equal = bool(torch.equal(loss_cpu, expected["loss"]))
                grad_equal = bool(torch.equal(grad_cpu, expected["gradient"]))
                loss_max = float((loss_cpu - expected["loss"]).abs().max())
                grad_max = float((grad_cpu - expected["gradient"]).abs().max())
                # Independent PyTorch is diagnostic only.  It deliberately
                # cannot veto byte-exact source/direct equivalence.
                eager_logits, eager_targets, eager_upstream = ce_fixture(torch, rows, repetition, device)
                eager_loss = F.cross_entropy(15.0 * torch.tanh(eager_logits / 15.0), eager_targets,
                                             reduction="none")
                eager_loss.backward(eager_upstream)
                module.synchronize(device)
                record["source_direct"] = {
                    "torch_equal_loss": loss_equal,
                    "torch_equal_gradient": grad_equal,
                    "max_abs_loss": loss_max,
                    "max_abs_gradient": grad_max,
                    "passed": loss_equal and grad_equal and loss_max == 0.0 and grad_max == 0.0,
                }
                record["pytorch_diagnostic_non_gating"] = {
                    "loss_max_abs": float((loss_cpu - eager_loss.detach().cpu()).abs().max()),
                    "gradient_max_abs": float((grad_cpu - eager_logits.grad.detach().cpu()).abs().max()),
                }
                del eager_logits, eager_targets, eager_upstream, eager_loss
            cases.append(record)
            del logits, targets, upstream, loss, loss_cpu, grad_cpu
    if args.variant == "source":
        if source_pt.exists() or source_pt.is_symlink():
            raise FileExistsError(source_pt)
        torch.save(saved, source_pt)
    expected_calls = [
        {"op": op, "rows": rows, "lanes": 2 if rows >= 512 else 1}
        for rows in ROWS for _ in range(REPETITIONS) for op in ("forward", "backward")
    ]
    passed = calls == expected_calls
    if args.variant == "direct":
        passed = passed and all(case["source_direct"]["passed"] for case in cases)
    result = {
        "schema": WORKER_SCHEMA,
        "worker": "ce",
        "variant": args.variant,
        "passed": passed,
        "cases": cases,
        "calls": calls,
        "expected_calls": expected_calls,
        "compile_cache": os.environ.get("NEURON_COMPILE_CACHE_DIR"),
    }
    exclusive_json(args.result_dir / f"ce-{args.variant}.json", result)
    return 0 if passed else 1


def canonical(value: Any, torch) -> Any:
    if isinstance(value, torch.Tensor):
        return tensor_digest(torch, value)
    if isinstance(value, dict):
        return {str(k): canonical(v, torch) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [canonical(v, torch) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported snapshot value: {type(value)}")


def configure_optimizer_module(module, torch) -> None:
    module.USE_FOREACH_OPTIM = True
    module.USE_TENSOR_LR_SCALARS = True
    module.USE_COMPILE_MUON = True
    module.USE_NKI_MUON = False
    module.USE_CAUTIOUS_WD = False
    module.COMPUTE_DTYPE = torch.bfloat16


def group_templates(module, torch, fixture: str) -> list[dict[str, Any]]:
    if fixture == "production_width1024_fs_w62":
        module.DEPTH, module.N_EMBD = 6, 1024
        module.USE_NGRAM_VE = module.USE_QK_SHIFT = False
        module.QK_SHIFT_BETA, module.QK_SHIFT_FREEZE = 0.0, True
        module.USE_LOCAL_CONV = module.TRAIN_LOCAL_CONV = True
        with torch.device("meta"):
            model = module.GPT(module.build_config())
            groups = model.setup_optimizer().param_groups
        if sum(p.numel() for g in groups for p in g["params"]) != 117453580:
            raise AssertionError("production width-1024 inventory drifted from fs-w62")
        return groups
    groups = [
        dict(kind="adamw", params=[torch.empty(64, 32, device="meta") for _ in range(5)],
             lr=.02, betas=(.8, .96), eps=1e-10, weight_decay=.01),
        dict(kind="adamw", params=[torch.empty(1, device="meta") for _ in range(12)],
             lr=.002, betas=(.9, .95), eps=1e-10, weight_decay=0.0),
    ]
    for shape, count in (((8, 32), 3), ((32, 32), 8), ((32, 64), 6), ((64, 32), 6)):
        groups.append(dict(kind="muon", params=[torch.empty(shape, device="meta") for _ in range(count)],
                           lr=.02, momentum=.95, beta2=.9, ns_steps=5,
                           weight_decay=.2, transposed=False))
    return groups


def instantiate_optimizer(module, torch, templates, device):
    generator = torch.Generator().manual_seed(9407)
    groups, parameters = [], []
    for template in templates:
        copied = {k: v for k, v in template.items() if k != "params"}
        copied["params"] = []
        for meta in template["params"]:
            p = torch.nn.Parameter((torch.randn(meta.shape, generator=generator) * .03).to(device))
            copied["params"].append(p)
            parameters.append(p)
        groups.append(copied)
    return parameters, module.MuonAdamW(groups), generator


def plan_snapshot(module, optimizer, owner) -> dict[str, Any]:
    entries, _, summary = module.make_plan(optimizer.param_groups, WORLD_SIZE)
    expected = {"entries": entries, "summary": summary}
    if owner is not None:
        actual = {"entries": owner.entries, "summary": owner.summary}
        if actual != expected:
            raise AssertionError("owner plan differs from canonical make_plan")
    return expected


def persistent_state(torch, optimizer, owner) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if owner is None:
        for gi, group in enumerate(optimizer.param_groups):
            if group["kind"] == "adamw":
                for pi, parameter in enumerate(group["params"]):
                    fields = optimizer.state.get(parameter, {})
                    result[f"g{gi}:p{pi}"] = canonical(fields, torch)
            else:
                fields = optimizer.state.get(group["params"][0], {})
                for pi in range(len(group["params"])):
                    mapped = {}
                    for key, value in fields.items():
                        if key in ("stacked_params", "stacked_grads"):
                            continue
                        mapped[key] = value[pi] if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == len(group["params"]) else value
                    result[f"g{gi}:p{pi}"] = canonical(mapped, torch)
        return result
    for gi, (source, owned, positions) in enumerate(zip(optimizer.param_groups, owner.groups, owner.positions)):
        if source["kind"] == "adamw":
            for local, position in enumerate(positions):
                result[f"g{gi}:p{position}"] = canonical(optimizer.state.get(owned["params"][local], {}), torch)
        elif positions:
            fields = optimizer.state.get(owned["params"][0], {})
            for local, position in enumerate(positions):
                mapped = {}
                for key, value in fields.items():
                    if key in ("stacked_params", "stacked_grads"):
                        continue
                    mapped[key] = value[local] if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == len(owned["params"]) else value
                result[f"g{gi}:p{position}"] = canonical(mapped, torch)
    return result


def snapshot_optimizer(torch, module, optimizer, owner, parameters, device, step: int) -> dict[str, Any]:
    parameter_records = {str(i): tensor_digest(torch, p) for i, p in enumerate(parameters)}
    group_scalars = []
    for group in optimizer.param_groups:
        group_scalars.append(canonical({k: v for k, v in group.items() if k != "params"}, torch))
    digest = hashlib.sha256()
    for record in parameter_records.values():
        digest.update(bytes.fromhex(record["sha256"]))
    local = torch.tensor(list(digest.digest()), device=device, dtype=torch.float32)
    gathered = torch.empty(32 * WORLD_SIZE, device=device, dtype=torch.float32)
    module.dist.all_gather_into_tensor(gathered, local)
    module.synchronize(device)
    rank_digests = [bytes(int(v) for v in row).hex() for row in gathered.cpu().view(WORLD_SIZE, 32)]
    return {
        "step": step,
        "parameters": parameter_records,
        "state": persistent_state(torch, optimizer, owner),
        "group_scalars": group_scalars,
        "owner_calls": step if owner is not None else step,
        "cross_rank_parameter_sha256": rank_digests,
        "cross_rank_exact": len(set(rank_digests)) == 1,
    }


def run_optimizer_fixture(torch, module, variant: str, fixture: str, device) -> dict[str, Any]:
    templates = group_templates(module, torch, fixture)
    parameters, optimizer, generator = instantiate_optimizer(module, torch, templates, device)
    if variant == "source":
        owner = module._submission_optimizer_install(optimizer, module)
        step_call = optimizer.step
    elif variant == "direct":
        owner = module.OwnedOptimizer(optimizer)
        step_call = owner.step
    else:
        owner = None
        step_call = optimizer.step
    plan = plan_snapshot(module, optimizer, owner)
    linear = projection = linear_input = cache_parameter = None
    if fixture == "production_width1024_fs_w62":
        cache_parameter = next(parameter for parameter in parameters if parameter.ndim == 2)
        linear = module.Linear(cache_parameter.shape[1], cache_parameter.shape[0], bias=False)
        linear.weight = cache_parameter
        projection = torch.compile(linear, backend="neuron", fullgraph=True, dynamic=False)
        linear_input = torch.randn(2, cache_parameter.shape[1], generator=generator).to(device).bfloat16()
    step_count = 2 if fixture == "tiny_fixed" else 4
    snapshots = []
    for iteration in range(1, step_count + 1):
        gradients = [torch.randn(p.shape, generator=generator) * .003 for p in parameters]
        update_adamw = True
        for group_index, group in enumerate(optimizer.param_groups):
            if fixture != "tiny_fixed":
                group["lr"] = 0.0 if iteration == 4 else templates[group_index]["lr"] / iteration
                if group["kind"] == "muon":
                    group["momentum"] = .85 + iteration * .02
                else:
                    group["betas"] = (.8 if iteration < 3 else .9, .96)
        if fixture != "tiny_fixed" and iteration == 2:
            for index in range(0, len(gradients), 5):
                gradients[index] = None
        if fixture != "tiny_fixed" and iteration == 3:
            update_adamw = False
            for index in range(1, len(gradients), 4):
                gradients[index].zero_()
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = None if gradient is None else gradient.to(device)
        step_call(update_adamw=update_adamw)
        module.synchronize(device)
        snapshot = snapshot_optimizer(torch, module, optimizer, owner, parameters, device, iteration)
        if linear is not None:
            with torch.no_grad():
                linear.refresh_weight_cache(torch.bfloat16)
                compiled = projection(linear_input)
                eager = torch.nn.functional.linear(linear_input, cache_parameter.bfloat16())
                module.synchronize(device)
                cache_exact = bool(torch.equal(linear._cached_weight.cpu(), cache_parameter.detach().cpu().bfloat16()))
                output_exact = bool(torch.equal(compiled.cpu(), eager.cpu()))
                snapshot["linear_cache"] = {
                    "cache_exact": cache_exact,
                    "output_exact": output_exact,
                    "compiled_output": tensor_digest(torch, compiled),
                    "eager_output": tensor_digest(torch, eager),
                }
                if not cache_exact or not output_exact:
                    raise AssertionError("production Linear cache refresh/output differs from fs-w62 reference")
        snapshots.append(snapshot)
        del gradients
    passed = all(item["cross_rank_exact"] for item in snapshots)
    result = {
        "fixture": fixture,
        "passed": passed,
        "parameter_count": len(parameters),
        "element_count": sum(p.numel() for p in parameters),
        "plan": plan,
        "snapshots": snapshots,
    }
    del parameters, optimizer, owner
    return result


def run_optimizer_worker(args: argparse.Namespace) -> int:
    if os.environ.get("TRAINIUM_DISCRIMINATOR_PARENT_NONCE") != args.parent_nonce:
        raise RuntimeError("optimizer worker lacks its authenticated parent nonce")
    import torch
    import torch.distributed as dist
    import torch_neuronx  # noqa: F401

    torch.set_num_threads(HOST_THREADS)
    path = SOURCE if args.variant in ("raw", "source") else DIRECT
    module = import_train(path, f"optimizer_{args.variant}_{os.getpid()}")
    configure_optimizer_module(module, torch)
    _, rank, _, world_size, device = module.init_runtime("neuron")
    if world_size != WORLD_SIZE or rank != int(os.environ["LOCAL_RANK"]):
        raise RuntimeError("optimizer worker requires exact four-rank torchrun")
    try:
        fixtures = [
            run_optimizer_fixture(torch, module, args.variant, name, device)
            for name in ("tiny_fixed", "tiny_changed", "production_width1024_fs_w62")
        ]
        result = {
            "schema": WORKER_SCHEMA,
            "worker": "optimizer",
            "variant": args.variant,
            "rank": rank,
            "world_size": world_size,
            "passed": all(item["passed"] for item in fixtures),
            "fixtures": fixtures,
            "compile_cache": os.environ.get("NEURON_COMPILE_CACHE_DIR"),
        }
        exclusive_json(args.result_dir / f"optimizer-{args.variant}-rank{rank}.json", result)
        return 0 if result["passed"] else 1
    finally:
        dist.destroy_process_group()


def scan_recompile_warning(text: str) -> list[str]:
    return [line for line in text.splitlines() if any(pattern.search(line) for pattern in RECOMPILE_PATTERNS)]


def run_child(command: list[str], env: dict[str, str], timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=timeout, check=False)
    warnings = scan_recompile_warning(completed.stdout)
    return {
        "argv": command,
        "returncode": completed.returncode,
        "seconds": time.monotonic() - started,
        "recompile_limit_warnings": warnings,
        "log_tail": completed.stdout[-12000:],
        "passed": completed.returncode == 0 and not warnings,
    }


def compare_optimizer_results(result_dir: Path) -> dict[str, Any]:
    reports = {
        variant: [json.loads((result_dir / f"optimizer-{variant}-rank{rank}.json").read_text())
                  for rank in range(WORLD_SIZE)]
        for variant in ("raw", "source", "direct")
    }
    failures = []
    for variant, ranks in reports.items():
        if not all(report["passed"] for report in ranks):
            failures.append(f"{variant}: worker failed")
    for rank in range(WORLD_SIZE):
        source = reports["source"][rank]
        direct = reports["direct"][rank]
        if source["fixtures"] != direct["fixtures"]:
            failures.append(f"source/direct rank {rank} snapshots differ")
    for fixture_index, fixture_name in enumerate(("tiny_fixed", "tiny_changed", "production_width1024_fs_w62")):
        raw = reports["raw"][0]["fixtures"][fixture_index]
        for rank in range(WORLD_SIZE):
            direct_fixture = reports["direct"][rank]["fixtures"][fixture_index]
            if direct_fixture["plan"] != raw["plan"]:
                failures.append(f"{fixture_name}: direct rank {rank} ownership plan differs from raw")
        for step_index in range(len(raw["snapshots"])):
            raw_step = raw["snapshots"][step_index]
            merged_state = {}
            for rank in range(WORLD_SIZE):
                direct_step = reports["direct"][rank]["fixtures"][fixture_index]["snapshots"][step_index]
                if direct_step["parameters"] != raw_step["parameters"]:
                    failures.append(f"{fixture_name} step {step_index + 1}: direct rank {rank} parameters differ from raw")
                if direct_step["group_scalars"] != raw_step["group_scalars"]:
                    failures.append(f"{fixture_name} step {step_index + 1}: direct rank {rank} scalars differ from raw")
                if direct_step["cross_rank_parameter_sha256"] != raw_step["cross_rank_parameter_sha256"]:
                    failures.append(f"{fixture_name} step {step_index + 1}: direct rank {rank} cross-rank digests differ from raw")
                overlap = set(merged_state) & set(direct_step["state"])
                if overlap:
                    failures.append(f"{fixture_name} step {step_index + 1}: duplicated owned state {sorted(overlap)[:3]}")
                merged_state.update(direct_step["state"])
            if merged_state != raw_step["state"]:
                failures.append(f"{fixture_name} step {step_index + 1}: merged direct state differs from raw")
    return {
        "passed": not failures,
        "failures": failures,
        "variants": {variant: {"rank_count": len(ranks), "all_workers_passed": all(r["passed"] for r in ranks)}
                     for variant, ranks in reports.items()},
        "worker_reports": reports,
    }


def validate_full_environment(args: argparse.Namespace) -> None:
    exact = {**STATIC_ENVIRONMENT,
        "NEURON_LOGICAL_NC_CONFIG": "2",
        "LOCAL_WORLD_SIZE": "4",
        "WORLD_SIZE": "4",
        "OMP_NUM_THREADS": str(HOST_THREADS),
        "TRAINIUM_DISCRIMINATOR_LABEL": args.run_label,
        "TRAINIUM_DISCRIMINATOR_EXPECTED_COMMIT": args.expected_commit,
        "TRAINIUM_DISCRIMINATOR_AUTHORIZED_HEAD": args.authorized_head,
        "TRAINIUM_DISCRIMINATOR_AUTHORIZATION_SHA256": args.authorization_sha256,
    }
    for key, expected in exact.items():
        if os.environ.get(key) != expected:
            raise ValueError(f"environment mismatch for {key}")
    if args.expected_hostname != EXPECTED_HOSTNAME or socket.gethostname() != EXPECTED_HOSTNAME:
        raise ValueError("hostname differs from external authorization")
    if args.expected_commit != args.authorized_head or git("rev-parse", "HEAD") != args.expected_commit:
        raise ValueError("expected commit, authorized HEAD, and checkout must be identical")
    if not re.fullmatch(r"fs-[a-z0-9][a-z0-9-]{15,160}", args.run_label):
        raise ValueError("run label is not unique/canonical")
    if str(Path(sys.executable).resolve()) != str(Path(NATIVE_PYTHON).resolve()):
        raise ValueError("harness is not running under the pinned native Python")
    if os.environ.get("TRAINIUM_FRESH_START_RUN_ID") != args.run_label:
        raise ValueError("fresh_start_guard run identity is missing/mismatched")
    if args.report.resolve() != (ROOT / "research/run-logs" / f"{args.run_label}-result.json").resolve():
        raise ValueError("report path is not the unique label-derived run-log target")


def exact_launch_request(args: argparse.Namespace, authorization: dict[str, Any]) -> dict[str, Any]:
    environment = dict(STATIC_ENVIRONMENT)
    environment.update({
        "TRAINIUM_DISCRIMINATOR_LABEL": args.run_label,
        "TRAINIUM_DISCRIMINATOR_EXPECTED_COMMIT": args.expected_commit,
        "TRAINIUM_DISCRIMINATOR_AUTHORIZED_HEAD": args.authorized_head,
        "TRAINIUM_DISCRIMINATOR_AUTHORIZATION_SHA256": args.authorization_sha256,
    })
    command = [
        *OUTER_TIMEOUT,
        NATIVE_PYTHON, str(HARNESS),
        "--authorization-request", str(REQUEST),
        "--authorized-head", args.authorized_head,
        "--authorization-sha256", args.authorization_sha256,
        "--expected-commit", args.expected_commit,
        "--run-label", args.run_label,
        "--report", str(args.report.resolve()),
        "--expected-hostname", args.expected_hostname,
    ]
    return {
        "host": EXPECTED_HOST,
        "remote": EXPECTED_REMOTE,
        "expected_commit": args.expected_commit,
        "authorized_head": args.authorized_head,
        "label": args.run_label,
        "purpose": EXPECTED_PURPOSE,
        "environment": environment,
        "command": command,
        "fingerprint_paths": [*PROTECTED_PATHS, rel(REQUEST)],
        "authorization_request": rel(REQUEST),
        "authorization_sha256": args.authorization_sha256,
    }


def validate_outer_receipts(args: argparse.Namespace, authorization: dict[str, Any]) -> dict[str, Any]:
    run_logs = ROOT / "research/run-logs"
    authorized_request = exact_launch_request(args, authorization)
    required_hashes = dict(authorization["expected_sha256"])
    required_hashes[rel(REQUEST)] = authorization["request_sha256"]
    receipts = {}
    augmented = {"started_at", "hostname", "git_commit", "source_dirty",
                 "preflight_free_bytes", "source_sha256"}
    for suffix in ("launch.json", "job.json"):
        path = run_logs / f"{args.run_label}.{suffix}"
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"required fresh-start receipt missing/unsafe: {path}")
        payload = json.loads(path.read_text())
        if set(payload) != set(authorized_request) | augmented:
            raise ValueError(f"{suffix} manifest key set differs from exact schema")
        if {key: payload[key] for key in authorized_request} != authorized_request:
            raise ValueError(f"{suffix} parsed request/argv/environment is not exact")
        if (payload.get("hostname") != EXPECTED_HOSTNAME
                or payload.get("git_commit") != args.expected_commit
                or payload.get("remote") != str(ROOT)
                or payload.get("source_dirty") is not False
                or payload.get("source_sha256") != required_hashes):
            raise ValueError(f"{suffix} checkout/fingerprint identity mismatch")
        receipts[suffix] = {"path": str(path), "sha256": sha256(path)}
    preflight = run_logs / f"{args.run_label}.preflight.json"
    if preflight.is_symlink() or not preflight.is_file():
        raise ValueError("required preflight receipt missing/unsafe")
    payload = json.loads(preflight.read_text())
    if (payload.get("label") != args.run_label or payload.get("remote") != str(ROOT)
            or payload.get("hostname") != EXPECTED_HOSTNAME or payload.get("passed") is not True):
        raise ValueError("preflight receipt does not prove the exact passing run")
    receipts["preflight.json"] = {"path": str(preflight), "sha256": sha256(preflight)}
    return receipts


def run_orchestrator(args: argparse.Namespace, authorization: dict[str, Any]) -> dict[str, Any]:
    validate_full_environment(args)
    outer_receipts = validate_outer_receipts(args, authorization)
    result_dir = Path(f"/tmp/{args.run_label}-discriminator-results")
    if result_dir.exists() or result_dir.is_symlink():
        raise FileExistsError(f"fresh result directory already exists: {result_dir}")
    result_dir.mkdir(mode=0o700)
    nonce = hashlib.sha256(f"{args.run_label}:{args.authorization_sha256}".encode()).hexdigest()
    common_env = os.environ.copy()
    common_env["TRAINIUM_DISCRIMINATOR_PARENT_NONCE"] = nonce
    processes = []
    python = NATIVE_PYTHON
    for variant in ("source", "direct"):
        cache = Path(f"/tmp/{args.run_label}-ce-{variant}-cache")
        if cache.exists() or cache.is_symlink():
            raise FileExistsError(f"compile cache is not fresh: {cache}")
        cache.mkdir(mode=0o700)
        env = dict(common_env, NEURON_COMPILE_CACHE_DIR=str(cache))
        command = [python, str(HARNESS), "--worker", "ce", "--variant", variant,
                   "--result-dir", str(result_dir), "--parent-nonce", nonce]
        processes.append({"name": f"ce-{variant}", **run_child(command, env, 1800)})
        if not processes[-1]["passed"]:
            raise RuntimeError(f"child failed or emitted recompile-limit warning: ce-{variant}")
    ce_source = json.loads((result_dir / "ce-source.json").read_text())
    ce_direct = json.loads((result_dir / "ce-direct.json").read_text())
    ce_passed = (
        ce_source["passed"] and ce_direct["passed"]
        and ce_source["calls"] == ce_direct["calls"]
        and len(ce_direct["cases"]) == len(ROWS) * REPETITIONS
        and all(case["source_direct"]["passed"] for case in ce_direct["cases"])
    )
    for variant in ("raw", "source", "direct"):
        cache = Path(f"/tmp/{args.run_label}-optimizer-{variant}-cache")
        if cache.exists() or cache.is_symlink():
            raise FileExistsError(f"compile cache is not fresh: {cache}")
        cache.mkdir(mode=0o700)
        env = dict(common_env, NEURON_COMPILE_CACHE_DIR=str(cache))
        command = [TORCHRUN, "--standalone", "--nproc_per_node=4", str(HARNESS),
                   "--worker", "optimizer", "--variant", variant,
                   "--result-dir", str(result_dir), "--parent-nonce", nonce]
        processes.append({"name": f"optimizer-{variant}", **run_child(command, env, 2400)})
        if not processes[-1]["passed"]:
            raise RuntimeError(f"child failed or emitted recompile-limit warning: optimizer-{variant}")
    optimizer = compare_optimizer_results(result_dir)
    return {
        "schema": REPORT_SCHEMA,
        "mode": "neuron",
        "passed": ce_passed and optimizer["passed"] and all(p["passed"] for p in processes),
        "authorization": authorization,
        "source_sha256": sha256(SOURCE),
        "direct_sha256": sha256(DIRECT),
        "ce": {
            "passed": ce_passed,
            "source_cases": ce_source["cases"],
            "direct_cases": ce_direct["cases"],
            "source_direct_exact": [case["source_direct"] for case in ce_direct["cases"]],
            "source_calls": ce_source["calls"],
            "direct_calls": ce_direct["calls"],
            "pytorch_diagnostic_non_gating": [case["pytorch_diagnostic_non_gating"] for case in ce_direct["cases"]],
        },
        "optimizer": optimizer,
        "processes": processes,
        "outer_receipts": outer_receipts,
        "result_directory": str(result_dir),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", choices=("ce", "optimizer"))
    p.add_argument("--variant", choices=("raw", "source", "direct"))
    p.add_argument("--result-dir", type=Path)
    p.add_argument("--parent-nonce")
    p.add_argument("--static-only", action="store_true")
    p.add_argument("--authorization-request", type=Path)
    p.add_argument("--authorized-head")
    p.add_argument("--authorization-sha256")
    p.add_argument("--run-label")
    p.add_argument("--expected-commit")
    p.add_argument("--report", type=Path)
    p.add_argument("--expected-hostname")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.worker:
        if not args.variant or not args.result_dir or not args.parent_nonce:
            raise ValueError("worker requires variant, result-dir, and parent-nonce")
        if args.worker == "ce" and args.variant == "raw":
            raise ValueError("CE has only source/direct variants")
        return run_ce_worker(args) if args.worker == "ce" else run_optimizer_worker(args)
    required = (args.authorization_request, args.authorized_head, args.authorization_sha256, args.report)
    if any(value is None for value in required):
        raise ValueError("top level requires authorization request/head/SHA and report")
    if args.report.parent.resolve() != (ROOT / "research/run-logs").resolve():
        raise ValueError("report must be a direct child of research/run-logs")
    if args.report.exists() or args.report.is_symlink():
        raise FileExistsError("report exists; overwrite is forbidden")
    static = static_source_checks()
    authorization = authenticate(args, full=not args.static_only)
    if args.static_only:
        report = {
            "schema": REPORT_SCHEMA,
            "mode": "static-only",
            "passed": static["passed"],
            "authorization": authorization,
            "static": static,
            "hardware_executed": False,
        }
    else:
        if not args.run_label or not args.expected_hostname or not args.expected_commit:
            raise ValueError("full mode requires run-label, expected-hostname, and expected-commit")
        try:
            report = run_orchestrator(args, authorization)
        except Exception as error:
            report = {
                "schema": REPORT_SCHEMA,
                "mode": "neuron",
                "passed": False,
                "authorization": authorization,
                "source_sha256": sha256(SOURCE),
                "direct_sha256": sha256(DIRECT),
                "failure": {"type": type(error).__name__, "message": str(error)},
            }
        report["static"] = static
        report["passed"] = report["passed"] and static["passed"]
    exclusive_json(args.report, report)
    print(json.dumps({"report": str(args.report), "passed": report["passed"], "mode": report["mode"]}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
