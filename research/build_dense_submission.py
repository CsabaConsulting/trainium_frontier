#!/usr/bin/env python3
"""Freeze the verified wave98 recipe as a self-contained submission.

Run locally, from this repository. Git supplies the immutable input files; the
submission itself does not run this builder or depend on Git/research files.
AST formatting removes historical comments, not executable training code.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "1da6c0714c0755fd3b078e972fbb0aefbb43bf6d"
DEST = ROOT / "submissions/dense6x1024_public0992851"
INPUTS = [
    "train.py",
    "research/scratch/mlp_fused_bf16.py",
    "research/scratch/mlp_fused_integration.py",
    "research/scratch/optimizer_ownership.py",
    "research/scratch/softcap_ce_parallel_nki.py",
    "research/scratch/softcap_ce_parallel_integration.py",
]
LAUNCH = (
    "NEURON_CC_FLAGS='--optlevel=1 --auto-cast matmult --auto-cast-type bf16' "
    "NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py "
    "--num-steps 1000000 --max-train-seconds 1800 --depth 6 --n-embd 1024 "
    "--seed 42 --compile --compile-muon --no-ngram-ve --no-qk-shift "
    "--compile-sdpa-direct --nki-local-conv --bf16-norm-output "
    "--pack-factor 32 --clip-after-reduce --seq-len 1024 --no-eval-public"
)
INTEGRATION = '''
class _SubmissionEvalGPT(GPT):
    """Use identical per-sequence arithmetic for every scorer batch size.

    BF16 backend kernels can select different rounding paths when the batch
    dimension changes. Evaluating independent rows with batch dimension one
    preserves the original B1 logits and makes batching a pure concatenation.
    The parameters, sequence computation and training model remain unchanged.
    """
    def forward(self, idx):
        if idx.ndim != 2 or idx.shape[0] < 1:
            raise ValueError("Expected a nonempty batch of token sequences")
        single_forward = super().forward
        if idx.shape[0] == 1:
            return single_forward(idx)
        return torch.cat([single_forward(idx[i:i + 1])
                          for i in range(idx.shape[0])], dim=0)


def _install_submission_training():
    """Training-only adapters; load_for_eval uses the native frozen model."""
    import sys
    module = sys.modules[__name__]
    original_init = GPT.init_weights
    original_setup_optimizer = GPT.setup_optimizer
    ce_installed = False

    def initialize(model):
        nonlocal ce_installed
        original_init(model)
        assert USE_BF16_NORM_OUTPUT and COMPUTE_DTYPE == torch.bfloat16
        before_rng = torch.get_rng_state().clone()
        _submission_mlp_install(model, "neuron", module, forward_backend="native")
        if not ce_installed:
            _submission_ce_install(module)
            ce_installed = True
        assert torch.equal(before_rng, torch.get_rng_state())
        print("SUBMISSION_TRAIN_INTEGRATION " + json.dumps(dict(
            source_commit="1da6c0714c0755fd3b078e972fbb0aefbb43bf6d",
            native_forward_custom_mlp_backward=True,
            optimizer_ownership=True, adaptive_nki_ce=True,
            loader_and_timing_hooks_unchanged=True)), flush=True)

    def setup_optimizer(model, *args, **kwargs):
        optimizer = original_setup_optimizer(model, *args, **kwargs)
        assert USE_FOREACH_OPTIM and USE_TENSOR_LR_SCALARS
        assert not USE_CAUTIOUS_WD
        assert dist.is_initialized() and dist.get_world_size() == 4
        ownership = _submission_optimizer_install(
            optimizer, module, pad_singleton_muon=True)
        print("SUBMISSION_OPTIMIZER_INTEGRATION " + json.dumps(dict(
            owner_partitioning=True, rank=dist.get_rank(),
            **ownership.summary)), flush=True)
        return optimizer

    GPT.init_weights = initialize
    GPT.setup_optimizer = setup_optimizer


if __name__ == "__main__":
    _install_submission_training()
    main()
'''


def read_source(path):
    return subprocess.check_output(
        ["git", "show", f"{SOURCE}:{path}"], cwd=ROOT).decode()


def normalized(node):
    return ast.dump(node, include_attributes=False)


def main():
    sources = {path: read_source(path) for path in INPUTS}
    tree = ast.parse(sources["train.py"])
    original = ast.parse(sources["train.py"])
    assert ast.unparse(tree.body[-1]) == "if __name__ == '__main__':\n    main()"
    tree.body.pop()
    tree.body[0] = ast.Expr(ast.Constant(
        "Trainium Frontier submission: dense 6x1024, seed42, T1024, pack32.\n\n"
        "Frozen from source 1da6c0714c0755fd3b078e972fbb0aefbb43bf6d.\n"
        "Verified source checkpoint: public val_bpb=0.9928513116311253 over\n"
        "20,971,520 tokens; this is not a private leaderboard result.\n"
        "All required kernels and training adapters are included in this file.\n"
        "The organizer supplies the unchanged prepare.py and fixed assets.\n"
        "Use the accompanying launch-command.txt; train from fresh initialization."
    ))
    # Training used BF16 compiler matmul autocast. The standalone scorer used
    # optlevel1 WITHOUT global autocast. Set each before the heavy SDK imports.
    env = ast.parse('''
if __name__ == "__main__":
    os.environ.setdefault("NEURON_CC_FLAGS", "--optlevel=1 --auto-cast matmult --auto-cast-type bf16")
''').body[0]
    insert_at = next(i for i, node in enumerate(tree.body)
                     if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "_PROCESS_STARTED"
                             for t in node.targets)) + 1
    tree.body.insert(insert_at, env)
    # The evaluation contract asks for a frozen model. This changes no logits
    # or parameters and leaves the training model's grad flags alone.
    hook = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "load_for_eval")
    constructors = [n for n in ast.walk(hook) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == "GPT"]
    assert len(constructors) == 1
    constructors[0].func.id = "_SubmissionEvalGPT"
    hook.body.insert(-1, ast.parse("model.requires_grad_(False)").body[0])

    renames = {
        "research/scratch/mlp_fused_integration.py": "_submission_mlp_install",
        "research/scratch/optimizer_ownership.py": "_submission_optimizer_install",
        "research/scratch/softcap_ce_parallel_integration.py": "_submission_ce_install",
    }
    local_imports = {"mlp_fused_bf16", "softcap_ce_parallel_nki"}

    class InlineImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module in local_imports:
                return None
            if node.module == "mlp_prepacked_fixed_sbuf":
                # This unused, non-native MLP branch was never in the winner.
                # Do not leave an unavailable optional research dependency.
                return ast.parse('raise ValueError("Unsubmitted MLP variant; use native forward")').body[0]
            return node

    names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    vendored = []
    for path in INPUTS[1:]:
        helper = InlineImports().visit(ast.parse(sources[path]))
        for node in helper.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if node.name in ("install", "install_optimizer"):
                    node.name = renames[path]
                assert node.name not in names, f"duplicate definition: {node.name}"
                names.add(node.name)
                vendored.append(dict(source=path, definition=node.name,
                                     ast_sha256=hashlib.sha256(normalized(node).encode()).hexdigest()))
            tree.body.append(node)
    tree.body.extend(ast.parse(INTEGRATION).body)
    ast.fix_missing_locations(tree)
    text = ast.unparse(tree) + "\n"
    reparsed = ast.parse(text)
    assert normalized(reparsed) == normalized(tree)
    # Every original function/class except the explicit eval freezing addition
    # remains AST-identical, including the complete training loop and loader.
    final_defs = {n.name: n for n in reparsed.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    preserved = []
    for node in original.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name != "load_for_eval":
            assert normalized(node) == normalized(final_defs[node.name]), node.name
            preserved.append(node.name)
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "train.py").write_text(text)
    (DEST / "launch-command.txt").write_text(LAUNCH + "\n")
    manifest = dict(
        source_commit=SOURCE, source_training="fs-w98-dense1024-context1024-pack32-owned-ce-1800-r1",
        public_val_bpb=0.9928513116311253, private_leaderboard_result=False,
        train_sha256=hashlib.sha256(text.encode()).hexdigest(), train_bytes=len(text.encode()),
        launch_command=LAUNCH,
        source_sha256={p: hashlib.sha256(s.encode()).hexdigest() for p, s in sources.items()},
        preserved_base_definitions=preserved, vendored_definitions=vendored,
        packaging_changes=[
            "inline source98 kernel/adapter modules; remove local helper imports",
            "rename the three installation functions to avoid name collisions",
            "reject unused fixed-SBUF MLP branch instead of importing research code",
            "install only winning training adapters when executed as __main__",
            "set training compiler BF16 matmul flags before SDK import; eval defaults optlevel1 only",
            "freeze requires_grad in load_for_eval, without changing model weights or logits",
            "evaluate rows independently to preserve original B1 logits across all batch sizes; no training change",
            "AST-format code to remove historical comments; unchanged training loop/data/scoring hooks",
            "launch disables optional in-process public evaluation; standalone scoring remains unchanged",
        ],
    )
    (DEST / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(path=str(DEST), train_sha256=manifest["train_sha256"],
                         train_bytes=manifest["train_bytes"], preserved_definitions=len(preserved))))


if __name__ == "__main__":
    main()
