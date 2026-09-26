"""CPU-only proof for a single human-selected scalar; never imports prepare.py."""
import ast
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PARENT = REPO / 'research/v2/proposals/20260926-sprint-p0-clean-batch/train.py'
EXPECTED_PARENT = '8c6d7af73e3237690c7f9a61c24544393b21224996d60484502ab379303c34ee'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def functions(raw):
    tree = ast.parse(raw)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in ('lr_multiplier', 'muon_momentum'):
            keep.append(node)
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), '<actual-source-functions>', 'exec'), ns)
    return tree, ns


parent, child = PARENT.read_bytes(), (HERE / 'train.py').read_bytes()
require(digest(parent) == EXPECTED_PARENT, 'PARENT_DRIFT')
old, new = b'MUON_MOMENTUM_WARMUP_STEPS = 300', b'MUON_MOMENTUM_WARMUP_STEPS = 150'
require(parent.count(old) == child.count(new) == 1, 'NONUNIQUE_LITERAL')
require(child.replace(new, old) == parent, 'ANY_OTHER_BYTES_CHANGED')
ptree, p = functions(parent)
ctree, c = functions(child)
normalized = ast.parse(child.replace(new, old))
require(ast.dump(normalized) == ast.dump(ptree), 'AST_DRIFT')
checks = 0
for step in range(0, 4001):
    for progress in (0.0, 0.5, 0.954, 0.955, 0.96, 0.98, 1.0):
        fraction = min(step / 150, 1.0)
        reference = 0.85 + fraction * 0.10
        cooling = min(1.0, max(0.0, (progress - 0.955) / 0.045))
        reference = reference * (1.0 - cooling) + 0.85 * cooling
        got = c['muon_momentum'](step, progress)
        require(math.isfinite(got) and abs(got - reference) < 1e-12, 'RAMP_REFERENCE_MISMATCH')
        if step >= 300:
            require(got == p['muon_momentum'](step, progress), 'POST_WARMUP_DIFFERENCE')
        require(c['lr_multiplier'](step, progress * 1800, 1800) ==
                p['lr_multiplier'](step, progress * 1800, 1800), 'LR_SCHEDULE_CHANGED')
        checks += 1
for step in range(301):
    require(abs(c['muon_momentum'](step, 0) - p['muon_momentum'](step * 2, 0)) < 1e-12,
            'TOKEN_MATCHED_RAMP_DIFFERENCE')
    checks += 1
require(c['muon_momentum'](0) == 0.85 and c['muon_momentum'](150) == 0.95, 'ENDPOINT_FAILURE')
report = {
    'status': 'PASS', 'scope': 'CPU scalar/reference and whole-source byte/AST identity, not native training',
    'parent_sha256': digest(parent), 'source_sha256': digest(child),
    'only_change': 'MUON_MOMENTUM_WARMUP_STEPS 300 -> 150', 'cases': checks,
    'same_functions_except_constant': True, 'independent_session_review': False,
    'human_selected_scalar_existing_logic': True,
    'momenta': {str(step): c['muon_momentum'](step) for step in (0, 75, 149, 150, 299, 300)},
    'old_batch_ramp_tokens': 131072 * 300, 'new_batch_original_ramp_tokens': 262144 * 300,
    'new_batch_candidate_ramp_tokens': 262144 * 150,
}
target = HERE / 'CPU_RECEIPT.json'
encoded = (json.dumps(report, indent=2) + '\n').encode()
if target.exists():
    require(target.read_bytes() == encoded, 'RECEIPT_DRIFT')
else:
    target.write_bytes(encoded)  # Generated test evidence.
print(json.dumps(report))
