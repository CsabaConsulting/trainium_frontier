"""Static active-path audit of the frozen base; no tensor/model import or host work."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[4]
COMMIT = "a4aa4351789115a2d0aa947d9ca00ff689f9bd0b"
SOURCE_PATH = "workflow/proposals/research-v2/train.py"
SHA = "6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3"
checks = []


def require(condition, description):
    if not condition:
        raise RuntimeError(description)
    checks.append(description)


def one(items, description):
    require(len(items) == 1, description)
    return items[0]


def function(parent, name):
    return one([n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name], name)


def names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


raw = subprocess.check_output(["git", "-C", str(REPO), "show", f"{COMMIT}:{SOURCE_PATH}"])
require(hashlib.sha256(raw).hexdigest() == SHA, "Frozen 6ef7 source SHA")
tree = ast.parse(raw)
install = function(tree, "_install_submission_training")
initialize = function(install, "initialize")
mlp_install = function(tree, "_submission_mlp_install")
make_forward = function(tree, "make_forward")
adapted_forward = function(make_forward, "forward")
reference = function(tree, "reference_forward")
autograd = function(tree, "make_autograd")
activation = function(tree, "relu_squared")
main_guards = [n for n in tree.body if isinstance(n, ast.If)
               and ast.unparse(n.test) == "__name__ == '__main__'"]
require(len(main_guards) == 2, "Separate compiler-environment and training main guards")
entrypoint = one([n for n in main_guards if "_install_submission_training" in names(n)],
                 "One training main entrypoint")
require([ast.unparse(n) for n in entrypoint.body] == ["_install_submission_training()", "main()"],
        "Training entrypoint installs adapters before main")
call = one([n for n in ast.walk(initialize) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "_submission_mlp_install"], "One MLP install")
require(ast.unparse(call) == "_submission_mlp_install(model, 'neuron', module, forward_backend='native')",
        "MLP install selects native backend, all layers by default")
require("USE_NKI_RELU2" in names(activation), "Original activation consults NKI ReLU2 flag")
for label, node in [("adapter", adapted_forward), ("reference", reference), ("backward", autograd)]:
    require(not ({"relu_squared", "USE_NKI_RELU2"} & names(node)), label + " does not call flag-controlled activation")
require([ast.unparse(n) for n in reference.body] == [
    "relu = F.relu(F.linear(x, up.to(x.dtype)))",
    "return (F.linear(relu.square(), down.to(x.dtype)), relu)",
], "Exact native forward ReLU then square")
fallback = one([n for n in adapted_forward.body if isinstance(n, ast.If)
                and len(n.body) == 1 and isinstance(n.body[0], ast.Return)
                and ast.unparse(n.body[0]) == "return native_forward(self, x, fc)"], "Single native fallback guard")
require(ast.unparse(fallback.test) == "not self.training or fc is not None or x.dtype != torch.bfloat16 or (tile < 128) or tile % 128 or tokens % (2 * tile)",
        "Exact eligibility guard")
require("module.forward = types.MethodType(forward, module)" in ast.unparse(mlp_install),
        "Installer actually binds instance forward")
require("selected = tuple(range(len(model.transformer.h))) if layer_indices is None else tuple(layer_indices)" in ast.unparse(mlp_install),
        "All MLP layers selected without layer_indices")
require("fused = make_autograd(kernel_call, transposed_relu=True" in ast.unparse(make_forward),
        "Transposed custom backward is active adapter")
geometry = []
for training, fc_present, bf16, tokens in [(True, False, True, 32768), (False, False, True, 1024),
                                          (True, False, True, 8), (True, True, True, 32768)]:
    tile = min(512, tokens // 2)
    fallback_taken = not training or fc_present or not bf16 or tile < 128 or tile % 128 != 0 or tokens % (2 * tile) != 0
    geometry.append(dict(training=training, fc_present=fc_present, bf16=bf16,
                         tokens=tokens, tile=tile, fallback=fallback_taken))
require([g["fallback"] for g in geometry] == [False, True, True, True], "Eligibility arithmetic including fallback examples")
print(json.dumps({
    "status": "STATIC_PATH_FINDING_NOT_RUNTIME_PROOF", "source_commit": COMMIT,
    "source_path": SOURCE_PATH, "source_sha256": SHA, "checks": checks,
    "geometry_examples_not_runtime_observations": geometry,
    "locations": {label: dict(start=n.lineno, end=n.end_lineno) for label, n in [
        ("activation", activation), ("native_reference", reference), ("make_autograd", autograd),
        ("adapted_forward", adapted_forward), ("mlp_installer", mlp_install), ("training_installer", install)]},
    "finding": "Do not schedule --nki-relu2 alone as a base training-speed arm: eligible training bypasses its original activation path.",
    "native_launch_authorized": False, "production_modified": False,
    "limits": ["No model imported", "No tensor or native execution", "No compiler graph or timing claim",
               "Fallback/eval may still consult the existing flag", "A wired K3 patch needs all ordinary independent gates"],
}, indent=2))
