"""Read-only authentication and semantic audit of newly supplied raw receipts."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('own_review', ROOT / 'output/review_cpu.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
review.authenticate()
expected_addendum = '3a964d71e5ed1b9d800e3bfc0a07b9ba6a5cbc48155cc8fda9b6a52b6e5ab0f0'
manifest_bytes = (ROOT / 'ADDENDUM_MANIFEST.json').read_bytes()
review.require(review.sha(manifest_bytes) == expected_addendum, 'addendum manifest hash')
manifest = json.loads(manifest_bytes)
review.require(manifest['original_input_manifest_sha256'] == review.MANIFEST_SHA, 'original binding')
for item in manifest['added_files']:
    path = ROOT / item['path']
    review.require(path.is_relative_to(ROOT) and '..' not in path.parts, 'packet-only paths')
    contents = path.read_bytes()
    review.require(len(contents) == item['bytes'] and review.sha(contents) == item['sha256'], item['path'])
review.emit('addendum_authenticated', sha256=expected_addendum, files=len(manifest['added_files']))

def wrapper_ast(path):
    tree = ast.parse((ROOT / path).read_bytes())
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'make_wrapper')

initial = wrapper_ast('source-context/check_native.py')
later = wrapper_ast('source-context/check_fp32_first.py')
review.require(ast.dump(initial) == ast.dump(later), 'two original Function wrappers differ')
review.emit('wrapper_equivalence', original20_checker_sha=review.sha((ROOT / 'source-context/check_native.py').read_bytes()),
            fp32_first_checker_sha=review.sha((ROOT / 'source-context/check_fp32_first.py').read_bytes()),
            exact_AST_equal=True, wrapper_ast_sha=review.sha(ast.dump(later).encode()))

ref_spec = importlib.util.spec_from_file_location('protected_reference', ROOT / 'source-context/reference.py')
reference = importlib.util.module_from_spec(ref_spec)
ref_spec.loader.exec_module(reference)
ids, values, meta = reference.fixture('unique', 'fp32')
gradient_bytes = reference.raw(reference.dense(ids, values))
expected_gradient_sha = review.sha(gradient_bytes)
zero_sha = review.sha(bytes(len(gradient_bytes)))
launch = json.loads((ROOT / 'native-fp32-first/diagnostic-launch.json').read_bytes())
child = json.loads((ROOT / 'native-fp32-first/child-exit.json').read_bytes())
terminal = json.loads((ROOT / 'native-fp32-first/terminal.json').read_bytes())
review.require(child['returncode'] == terminal['returncode'] == 1 and terminal['complete'] is False,
               'failed historical run status not preserved')
for rank in range(4):
    r = json.loads((ROOT / f'native-fp32-first/receipts/rank{rank}.json').read_bytes())
    review.require(r['rank'] == rank and r['world_size'] == 4 and r['seed'] == 42, 'rank/world/seed')
    review.require(r['checker_sha256'] == launch['support_sha256']['check_native.py'], 'checker launch binding')
    review.require(r['checker_sha256'] == review.sha((ROOT / 'source-context/check_fp32_first.py').read_bytes()), 'checker source pin')
    review.require(r['kernel_sha256'] == review.sha((ROOT / 'source-context/consumer_nki.py').read_bytes()), 'kernel pin')
    review.require(r['reference_sha256'] == review.sha((ROOT / 'source-context/reference.py').read_bytes()), 'reference pin')
    review.require(r['case']['case'] == 'unique' and r['case']['dtype'] == 'torch.float32', 'diagnostic scope')
    review.require(r['no_training'] and r['no_eval'] and not r['native_qualification'], 'diagnostic flags')
    summaries = []
    for route in r['case']['routes']:
        gradient = route['checks']['gradient']
        review.require(route['inputs_before'] == route['inputs_after'] and len(route['inputs_before']) == 7,
                       'historical input observations differ')
        review.require(gradient['expected']['sha256'] == expected_gradient_sha, 'reference gradient identity')
        if route['name'] == 'compiled_custom':
            review.require(not gradient['exact'] and gradient['actual']['sha256'] == zero_sha
                           and gradient['actual']['nonzero'] == 0 and gradient['actual']['finite'],
                           'historical zero gradient evidence')
        else:
            review.require(gradient['exact'] and gradient['actual']['sha256'] == expected_gradient_sha,
                           'historical control evidence')
        if 'output' in route['checks']:
            review.require(route['checks']['output']['exact'], 'historical forward evidence')
        summaries.append(dict(name=route['name'], status=route['status'],
                              actual_gradient_sha=gradient['actual']['sha256'],
                              gradient_nonzero=gradient['actual']['nonzero'], inputs_unchanged=True))
    review.emit('historical_rank_audit', rank=rank, torch=r['torch'], nki=r['nki_version'],
                status=r['status'], routes=summaries, candidate_native_evidence=False)
review.emit('addendum_summary', status='AUTHENTICATED_HISTORICAL_FAILURE_PRESERVED',
            expected_gradient_sha=expected_gradient_sha, zero_gradient_sha=zero_sha,
            wrapper_identity_resolved=True, new_registered_candidate_native_tested=False,
            limits='Records only; no access to native runtime, outer archive or inner VJP values')
