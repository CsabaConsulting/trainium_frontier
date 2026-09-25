"""Tensor-free independent preservation/hash audit of the diagnostic-only patch."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = "50a901374ae6f76b35b8556d32fd72b1ece28fc6"
CANDIDATE = "fe781c226a7c700a71324783b70018613c1dbcaf"
CHECKER = "research/v2/lanes/fp8-projections/check_native.py"
PACKET = "research/v2/proposals/20260925-l1-fp8-vjp-diagnostic"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def digest(data):
    return hashlib.sha256(data).hexdigest()


class RemoveDiagnosticAdditions(ast.NodeTransformer):
    def visit_If(self, node):
        test = node.test
        if (isinstance(test, ast.Attribute) and isinstance(test.value, ast.Name)
                and test.value.id == "args" and test.attr == "diagnose_first_case"):
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        call = node.value
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == "add_argument" and call.args
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value in ("--diagnose-first-case", "--label")):
            return None
        return self.generic_visit(node)


def main():
    source = "workflow/proposals/research-v2/train.py"
    for path in (source, "prepare.py"):
        assert (ROOT / path).read_bytes() == git("show", f"{BASE}:{path}")
    old_bytes = git("show", f"{BASE}:{CHECKER}")
    new_bytes = (ROOT / CHECKER).read_bytes()
    assert new_bytes == git("show", f"{CANDIDATE}:{CHECKER}")
    old_nodes = {n.name: n for n in ast.parse(old_bytes).body if isinstance(n, ast.FunctionDef)}
    new_nodes = {n.name: n for n in ast.parse(new_bytes).body if isinstance(n, ast.FunctionDef)}
    dump = lambda node: ast.dump(node, include_attributes=False)
    assert dump(old_nodes["metric"]) == dump(new_nodes["metric"])
    restored_main = RemoveDiagnosticAdditions().visit(copy.deepcopy(new_nodes["main"]))
    assert dump(restored_main) == dump(old_nodes["main"])
    diagnostic_returns = [n for n in ast.walk(new_nodes["first_case_diagnostic"])
                          if isinstance(n, ast.Return) and isinstance(n.value, ast.Call)
                          and isinstance(n.value.func, ast.Name) and n.value.func.id == "dict"]
    assert len(diagnostic_returns) == 1
    flags = {k.arg: k.value for k in diagnostic_returns[0].value.keywords}
    assert ast.literal_eval(flags["verdict"]) == "DIAGNOSTIC_ONLY_NOT_COMPONENT_PASS"
    assert ast.literal_eval(flags["full_component_gate_passed"]) is False
    assert ast.literal_eval(flags["whole_step_measured"]) is False
    assert ast.literal_eval(flags["original_rms_ceiling"]) == .01
    manifest = json.loads((ROOT / PACKET / "SOURCE_MANIFEST.json").read_text())
    for path, expected in manifest["files"].items():
        assert digest((ROOT / path).read_bytes()) == expected
    assert digest(old_bytes) == manifest["original_checker_sha256"]
    proposal = json.loads((ROOT / PACKET / "EXPERIMENT_PROPOSAL.json").read_text())
    assert proposal["wall_timeout_seconds"] == 300
    assert proposal["gates"]["launch_authorized"] is False
    assert proposal["submission_eligible"] is False
    assert proposal["candidate"]["argv_tail"] == ["--diagnose-first-case", "--label", "fs-20260925-l1-921271-vjpdiag-r1"]
    changed = git("diff", "--name-only", BASE, CANDIDATE).decode().splitlines()
    assert len(changed) == 6
    assert all(p in (CHECKER, "research/v2/lanes/fp8-projections/test_native_diagnostic_cpu.py")
               or p.startswith(PACKET + "/") for p in changed)
    hashes = {path: digest((ROOT / path).read_bytes()) for path in [source, "prepare.py", *changed]}
    report = dict(verdict="PASS_DIAGNOSTIC_PATCH_STATIC_ONLY", commit=CANDIDATE,
                  files_sha256=hashes, source_and_prepare_unchanged=True,
                  complete_original_main_AST_preserved_after_removing_opt_in_additions=True,
                  original_metric_AST_identical=True, diagnostic_return_cannot_claim_component_PASS=True,
                  original_rms_ceiling=.01, manifest_hashes_verified=True,
                  proposal_wall_timeout_seconds=300, native_executed=False,
                  launch_authorization=False, original_failure_unchanged="c1ac162c")
    (HERE / "static-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
