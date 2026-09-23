#!/usr/bin/env python3
"""Build the reviewer-safe direct-integration dense6x1024 submission.

Input:  submissions/dense6x1024_lean_review/train.py (hash pinned below)
Output: submissions/dense6x1024_direct_review/

The transformation is deliberately narrow and fail-closed:

* the validated fused-MLP autograd path becomes the ordinary ``MLP.forward`` path;
* the validated adaptive NKI CE kernels are called directly by
  ``_SoftcapCrossEntropyNKI``;
* ``main`` constructs ``OwnedOptimizer`` explicitly and calls it for updates;
* runtime method/global replacement helpers and their integration logs are removed.

The builder verifies untouched model, optimizer, data, timing, checkpoint and evaluation
definitions by AST identity, verifies the moved adaptive kernel bodies by AST identity, runs the
organiser's static checker, and rejects every reviewer-triggering construct named by the proposal.
It does not run training or claim independent approval.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import difflib
import hashlib
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import build_lean_submission as lean  # noqa: E402


SRC = ROOT / "submissions/dense6x1024_lean_review/train.py"
SRC_SHA256 = "9173f62617789b058083c3a4722a906e2b165f4986716281f1915db3fb9fa344"
DEST = ROOT / "submissions/dense6x1024_direct_review"
LAUNCH = (
    "NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py "
    "--n-embd 1024 --no-ngram-ve --no-qk-shift --compile-sdpa-direct "
    "--nki-local-conv --bf16-norm-output --pack-factor 32 --clip-after-reduce "
    "--seq-len 1024 --no-eval-public"
)


MLP_FORWARD = '''def forward(self, x: torch.Tensor, fc: torch.Tensor | None=None) -> torch.Tensor:
    tokens = x.numel() // x.shape[-1]
    tile = min(512, tokens // 2)
    if not self.training or fc is not None or x.dtype != torch.bfloat16 or tile < 128 or tile % 128 or tokens % (2 * tile):
        h = self.c_fc(x) if fc is None else fc
        return self.c_proj(F.relu(h).square())
    up = _linear_weight(self.c_fc, x.dtype)
    down = _linear_weight(self.c_proj, x.dtype)
    result = _FusedMLP.apply(x.reshape(tokens, x.shape[-1]), up, down)
    return result.reshape(x.shape)
'''


CE_FORWARD = '''@staticmethod
def forward(ctx, z, tgt):
    lanes = selected_lanes(z.shape[0])
    loss, lse = _softcap_ce_fwd_native[lanes](z, tgt, lanes, True)
    ctx.save_for_backward(z, tgt, lse)
    return loss.reshape(-1)
'''


CE_BACKWARD = '''@staticmethod
def backward(ctx, grad_out):
    z, tgt, lse = ctx.saved_tensors
    g = grad_out.reshape(-1, 1).float().contiguous()
    lanes = selected_lanes(z.shape[0])
    return (_softcap_ce_bwd_native[lanes](z, tgt, lse, g, lanes), None)
'''


CE_WRAPPERS = '''from torch_neuronx import wrap_nki as _wrap_softcap_ce_nki
_softcap_ce_fwd_native = _wrap_softcap_ce_nki(_softcap_ce_fwd)
_softcap_ce_bwd_native = _wrap_softcap_ce_nki(_softcap_ce_bwd)
'''


UNTOUCHED_DEFS = [
    "GPTConfig",
    "Linear",
    "CausalConv",
    "CausalSelfAttention",
    "Block",
    "GPT",
    "build_config",
    "_muon_math",
    "_muon_lr_g",
    "MuonAdamW",
    "sync_gradients",
    "clip_and_sync_gradients",
    "save_checkpoint",
    "load_for_eval",
    "lr_multiplier",
    "muon_momentum",
    "make_plan",
    "OwnedOptimizer",
]


REMOVED_DEFS = [
    "make_autograd",
    "make_forward",
    "_submission_mlp_install",
    "_submission_optimizer_install",
    "build_kernels",
    "_submission_ce_install",
    "_install_submission_training",
]


TRIGGERS = (
    "_submission_",
    "_install_submission_training",
    "MethodType",
    "types.MethodType",
    "SUBMISSION_",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def module_def(tree: ast.Module, name: str) -> ast.FunctionDef | ast.ClassDef:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return node
    raise AssertionError(f"module definition not found: {name}")


def direct_def(body: list[ast.stmt], name: str) -> ast.FunctionDef | ast.ClassDef:
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return node
    raise AssertionError(f"direct definition not found: {name}")


def nki_try(tree: ast.Module) -> ast.Try:
    for node in tree.body:
        if isinstance(node, ast.Try) and any(
            isinstance(child, ast.FunctionDef) and child.name == "_softcap_ce_fwd"
            for child in node.body
        ):
            return node
    raise AssertionError("NKI definition try block not found")


def replace_direct(body: list[ast.stmt], old: ast.stmt, new: ast.stmt) -> None:
    index = body.index(old)
    body[index] = new


def strip_name(node: ast.FunctionDef | ast.ClassDef, name: str = "normalized") -> str:
    copy = deepcopy(node)
    copy.name = name
    return ast.dump(copy, include_attributes=False)


def normalized_main(node: ast.FunctionDef) -> str:
    copy = deepcopy(node)
    owner_assign = None
    owner_assert = None
    for statement in copy.body:
        text = ast.unparse(statement)
        if text == "assert dist.is_initialized() and dist.get_world_size() == 4":
            owner_assert = statement
        elif text == "owned_optimizer = OwnedOptimizer(optimizer)":
            owner_assign = statement
    assert owner_assert is not None and owner_assign is not None
    copy.body.remove(owner_assert)
    copy.body.remove(owner_assign)

    class RestoreStep(ast.NodeTransformer):
        def visit_Name(self, item: ast.Name):
            if item.id == "owned_optimizer":
                return ast.copy_location(ast.Name(id="optimizer", ctx=item.ctx), item)
            return item

    copy = RestoreStep().visit(copy)
    ast.fix_missing_locations(copy)
    return ast.dump(copy, include_attributes=False)


def transform(source: str) -> tuple[str, dict]:
    tree = ast.parse(source)
    original = ast.parse(source)

    # Fused MLP: move the already-validated helper bodies beside MLP and make the optimized path
    # an ordinary class method rather than a per-instance MethodType replacement.
    reference = module_def(tree, "reference_forward")
    make_autograd = module_def(tree, "make_autograd")
    make_forward = module_def(tree, "make_forward")
    fused = deepcopy(direct_def(make_autograd.body, "FusedMLP"))
    fused.name = "_FusedMLP"
    fused_forward = direct_def(fused.body, "forward")
    kernel_call = next(
        call
        for call in ast.walk(fused_forward)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "kernel_call"
    )
    kernel_call.func.id = "_fused_mlp_kernel"
    kernel = deepcopy(direct_def(make_forward.body, "kernel_call"))
    kernel.name = "_fused_mlp_kernel"
    weight = deepcopy(direct_def(make_forward.body, "weight"))
    weight.name = "_linear_weight"

    tree.body.remove(reference)
    mlp = module_def(tree, "MLP")
    insert_at = tree.body.index(mlp)
    tree.body[insert_at:insert_at] = [reference, kernel, weight, fused]
    mlp_forward = direct_def(mlp.body, "forward")
    replace_direct(mlp.body, mlp_forward, ast.parse(MLP_FORWARD).body[0])

    # Adaptive CE: replace the unused single-lane kernel pair with the exact validated adaptive
    # kernel bodies and call their wrapped launch grids directly from the existing autograd class.
    source_build = module_def(tree, "build_kernels")
    adaptive_forward = deepcopy(direct_def(source_build.body, "forward"))
    adaptive_backward = deepcopy(direct_def(source_build.body, "backward"))
    adaptive_forward.name = "_softcap_ce_fwd"
    adaptive_backward.name = "_softcap_ce_bwd"
    try_node = nki_try(tree)
    old_ce_forward = direct_def(try_node.body, "_softcap_ce_fwd")
    old_ce_backward = direct_def(try_node.body, "_softcap_ce_bwd")
    forward_index = try_node.body.index(old_ce_forward)
    backward_index = try_node.body.index(old_ce_backward)
    assert forward_index < backward_index
    try_node.body[forward_index] = adaptive_forward
    try_node.body[backward_index] = adaptive_backward
    ce_class = direct_def(try_node.body, "_SoftcapCrossEntropyNKI")
    ce_forward = direct_def(ce_class.body, "forward")
    ce_backward = direct_def(ce_class.body, "backward")
    replace_direct(ce_class.body, ce_forward, ast.parse(CE_FORWARD).body[0])
    replace_direct(ce_class.body, ce_backward, ast.parse(CE_BACKWARD).body[0])
    ce_class_index = try_node.body.index(ce_class)
    try_node.body[ce_class_index:ce_class_index] = ast.parse(CE_WRAPPERS).body

    # Optimizer ownership: retain the base optimizer for zero_grad/schedules and use the ownership
    # object explicitly for the update, exactly matching the prior patched step target.
    main = module_def(tree, "main")
    setup = next(
        statement
        for statement in main.body
        if isinstance(statement, ast.Assign)
        and ast.unparse(statement) == "optimizer = model.setup_optimizer()"
    )
    setup_index = main.body.index(setup)
    main.body[setup_index + 1:setup_index + 1] = ast.parse(
        "assert dist.is_initialized() and dist.get_world_size() == 4\n"
        "owned_optimizer = OwnedOptimizer(optimizer)\n"
    ).body
    step_calls = [
        call
        for call in ast.walk(main)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "optimizer"
        and call.func.attr == "step"
    ]
    assert len(step_calls) == 1
    step_calls[0].func.value.id = "owned_optimizer"

    # Delete only the superseded factories/installers and their now-unused imports.
    for name in REMOVED_DEFS:
        tree.body.remove(module_def(tree, name))
    imports_removed = []
    for statement in list(tree.body):
        if isinstance(statement, ast.Import) and [alias.name for alias in statement.names] == ["types"]:
            tree.body.remove(statement)
            imports_removed.append("import types")
        elif isinstance(statement, ast.ImportFrom) and statement.module == "types" and [
            alias.name for alias in statement.names
        ] == ["MethodType"]:
            tree.body.remove(statement)
            imports_removed.append("from types import MethodType")
    assert sorted(imports_removed) == ["from types import MethodType", "import types"]

    entry = next(
        statement
        for statement in tree.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "__name__ == '__main__'"
        and "_install_submission_training" in ast.unparse(statement)
    )
    entry_calls = [
        ast.unparse(statement.value)
        for statement in entry.body
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
    ]
    assert entry_calls == ["_install_submission_training()", "main()"]
    entry.body = [ast.parse("main()").body[0]]

    ast.fix_missing_locations(tree)
    output = ast.unparse(tree) + "\n"
    final = ast.parse(output)

    # Static proof obligations.
    untouched = {
        name: ast.dump(module_def(original, name), include_attributes=False)
        == ast.dump(module_def(final, name), include_attributes=False)
        for name in UNTOUCHED_DEFS
    }
    source_main = module_def(original, "main")
    main_only_direct_ownership = normalized_main(module_def(final, "main")) == ast.dump(
        source_main, include_attributes=False
    )

    original_build = module_def(original, "build_kernels")
    original_adaptive_forward = direct_def(original_build.body, "forward")
    original_adaptive_backward = direct_def(original_build.body, "backward")
    final_try = nki_try(final)
    final_adaptive_forward = direct_def(final_try.body, "_softcap_ce_fwd")
    final_adaptive_backward = direct_def(final_try.body, "_softcap_ce_bwd")
    adaptive_kernels_identical = {
        "forward": strip_name(original_adaptive_forward) == strip_name(final_adaptive_forward),
        "backward": strip_name(original_adaptive_backward) == strip_name(final_adaptive_backward),
    }

    original_kernels = {
        node.name: ast.dump(node, include_attributes=False)
        for node in ast.walk(original)
        if lean.is_nki_kernel(node)
        and node.name not in {"_softcap_ce_fwd", "_softcap_ce_bwd", "forward", "backward"}
    }
    final_kernels = {
        node.name: ast.dump(node, include_attributes=False)
        for node in ast.walk(final)
        if lean.is_nki_kernel(node)
        and node.name not in {"_softcap_ce_fwd", "_softcap_ce_bwd"}
    }
    other_kernels_identical = {
        name: final_kernels.get(name) == body for name, body in original_kernels.items()
    }

    original_reference = module_def(original, "reference_forward")
    final_reference = module_def(final, "reference_forward")
    source_make_forward = module_def(original, "make_forward")
    source_make_autograd = module_def(original, "make_autograd")
    source_kernel = direct_def(source_make_forward.body, "kernel_call")
    source_weight = direct_def(source_make_forward.body, "weight")
    source_fused = direct_def(source_make_autograd.body, "FusedMLP")
    final_fused = module_def(final, "_FusedMLP")
    normalized_final_fused = deepcopy(final_fused)
    normalized_final_fused.name = "FusedMLP"
    for call in ast.walk(normalized_final_fused):
        if isinstance(call, ast.Name) and call.id == "_fused_mlp_kernel":
            call.id = "kernel_call"
    fused_helpers_identical = {
        "reference_forward": ast.dump(original_reference, include_attributes=False)
        == ast.dump(final_reference, include_attributes=False),
        "kernel_call": strip_name(source_kernel) == strip_name(module_def(final, "_fused_mlp_kernel")),
        "weight": strip_name(source_weight) == strip_name(module_def(final, "_linear_weight")),
        "autograd": ast.dump(source_fused, include_attributes=False)
        == ast.dump(normalized_final_fused, include_attributes=False),
    }

    removed_present = [
        name for name in REMOVED_DEFS if any(
            isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name
            for node in ast.walk(final)
        )
    ]
    trigger_hits = [trigger for trigger in TRIGGERS if trigger in output]
    replacement_assignments = [
        ast.unparse(node)
        for node in ast.walk(final)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Attribute)
            and target.attr in {"forward", "step", "state_dict", "load_state_dict"}
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    ]
    undefined = lean.undefined_names(output, "train.py")
    forbidden = lean.forbidden_calls(final)
    with tempfile.TemporaryDirectory(prefix="direct-review-build-") as temp_dir:
        check_path = Path(temp_dir) / "train.py"
        check_path.write_text(output)
        organiser_warnings = lean.organiser_check(check_path, LAUNCH)
    source_report = json.loads((SRC.parent / "build-report.json").read_text())
    report = {
        "input_sha256": SRC_SHA256,
        "input_lines": source.count("\n"),
        "output_lines": output.count("\n"),
        "output_bytes": len(output.encode()),
        "output_sha256": sha256(output.encode()),
        "launch_command": LAUNCH,
        "edits": [
            "directly integrated the validated fused MLP forward/backward path",
            "directly integrated the validated adaptive NKI cross-entropy launch path",
            "directly instantiated optimizer ownership in main",
            "removed runtime method/global replacement helpers and integration logs",
        ],
        "definitions_removed": REMOVED_DEFS,
        "imports_removed": imports_removed,
        "untouched_definitions_identical": untouched,
        "main_only_direct_ownership_changed": main_only_direct_ownership,
        "adaptive_nki_kernel_bodies_identical": adaptive_kernels_identical,
        "other_nki_kernels_identical": other_kernels_identical,
        "fused_mlp_helper_bodies_identical": fused_helpers_identical,
        "removed_definitions_still_present": removed_present,
        "review_trigger_hits": trigger_hits,
        "runtime_method_replacement_assignments": replacement_assignments,
        "undefined_names": undefined,
        "forbidden_name_hits": forbidden,
        "organiser_checker_warnings": organiser_warnings,
        "record_step_boundary_present": "record_step_boundary(step)" in output,
        "load_for_eval_identical": untouched["load_for_eval"],
        "save_checkpoint_identical": untouched["save_checkpoint"],
        "globals_frozen": source_report["globals_frozen"],
    }
    report["static_gates_passed"] = all(untouched.values()) and main_only_direct_ownership \
        and all(adaptive_kernels_identical.values()) and all(other_kernels_identical.values()) \
        and all(fused_helpers_identical.values()) and not removed_present and not trigger_hits \
        and not replacement_assignments and not undefined and not forbidden and not organiser_warnings \
        and report["record_step_boundary_present"]
    return output, report


def write_package(destination: Path, output: str, report: dict) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "train.py").write_text(output)
    (destination / "launch-command.txt").write_text(LAUNCH + "\n")
    (destination / "build-report.json").write_text(json.dumps(report, indent=1) + "\n")

    source_tree = ast.parse(SRC.read_text())
    final_tree = ast.parse(output)
    review = destination / "review"
    review.mkdir(exist_ok=True)
    for path in review.glob("*.diff"):
        path.unlink()
    source_try = nki_try(source_tree)
    final_try = nki_try(final_tree)
    pairs = {
        "MLP": (module_def(source_tree, "MLP"), module_def(final_tree, "MLP")),
        "main": (module_def(source_tree, "main"), module_def(final_tree, "main")),
        "_SoftcapCrossEntropyNKI": (
            direct_def(source_try.body, "_SoftcapCrossEntropyNKI"),
            direct_def(final_try.body, "_SoftcapCrossEntropyNKI"),
        ),
        "_softcap_ce_fwd": (
            direct_def(source_try.body, "_softcap_ce_fwd"),
            direct_def(final_try.body, "_softcap_ce_fwd"),
        ),
        "_softcap_ce_bwd": (
            direct_def(source_try.body, "_softcap_ce_bwd"),
            direct_def(final_try.body, "_softcap_ce_bwd"),
        ),
    }
    for name, (before, after) in pairs.items():
        diff = difflib.unified_diff(
            ast.unparse(before).splitlines(),
            ast.unparse(after).splitlines(),
            "lean_review",
            "direct_review",
            n=3,
            lineterm="",
        )
        (review / f"{name}.diff").write_text("\n".join(line.rstrip() for line in diff) + "\n")

    builder_hash = sha256(Path(__file__).read_bytes())
    source_provenance = json.loads((SRC.parent / "provenance.json").read_text())
    provenance = {
        "package": "dense6x1024_direct_review",
        "derived_from": {
            "package": "dense6x1024_lean_review",
            "train_py_sha256": SRC_SHA256,
            "source_package": source_provenance["derived_from"],
        },
        "builder": {
            "path": "research/build_direct_review_submission.py",
            "sha256": builder_hash,
        },
        "output": {
            "train_py_sha256": report["output_sha256"],
            "lines": report["output_lines"],
            "bytes": report["output_bytes"],
        },
        "launch_command": LAUNCH,
        "launch_command_identical_to_source": LAUNCH
        == (SRC.parent / "launch-command.txt").read_text().strip(),
        "edits": report["edits"],
        "static_gates_passed": report["static_gates_passed"],
        "cpu_equivalence": "see cpu-equivalence.json",
        "hardware_validation": (
            "NOT RUN: independent review, Neuron validation, source replay and a cold rehearsal "
            "remain required before any submission"
        ),
    }
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")

    readme = f"""# dense6x1024_direct_review

Deterministic direct-integration build of `submissions/dense6x1024_lean_review/train.py`.
It preserves the frozen dense 6x1024 model, optimizer arithmetic, NKI kernel bodies, data path,
timing, checkpoint/evaluation behavior and launch command while replacing the three runtime-installed
training adapters with ordinary explicit code paths.

Builder: `research/build_direct_review_submission.py` (input pinned to `{SRC_SHA256}`).

Changes:
{chr(10).join('- ' + edit for edit in report['edits'])}

Local implementation evidence only:

- `build-report.json` records AST identity and organiser/static gates.
- `review/*.diff` contains the five targeted definition diffs.
- `cpu-equivalence.json`, generated by the proposal's local CPU harness, records model-state,
  eval-logit, fused-MLP and distributed ownership equivalence plus static CE-kernel identity.

This package has not been independently approved or run on Trainium. Neuron contract checks,
source-checkpoint replay, a cold full-chip rehearsal and protected public evaluation remain required.

Launch command (unchanged):

    {LAUNCH}
"""
    (destination / "README.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    source_bytes = SRC.read_bytes()
    assert sha256(source_bytes) == SRC_SHA256, "input changed; re-pin deliberately"
    output, report = transform(source_bytes.decode())
    print(json.dumps({key: value for key, value in report.items() if key != "globals_frozen"}, indent=1))
    if not report["static_gates_passed"]:
        raise SystemExit("static gates failed")
    if not args.write:
        print("\n(dry run; pass --write to emit the package)")
        return
    write_package(args.dest, output, report)
    print(f"\nwrote {args.dest} at {report['output_sha256']}")


if __name__ == "__main__":
    main()
