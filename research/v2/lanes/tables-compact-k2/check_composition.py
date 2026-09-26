"""Lightweight exact byte/AST/component binding; not a new source review."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
TABLE = HERE.parent / 'tables-runtime-select'
COMPACT = HERE.parent / 'k2-runtime-groups'
PINS = {
    HERE / 'train.py': '4b74ed48c772a5e82807d5509529704deaf1c8cd56692705efe9047ea7ae4a7b',
    TABLE / 'train.py': '414cdd76df85450659375c0928b7f0c882b963088529ca049c19be32814fec34',
    COMPACT / 'train.py': 'f8047d07908386c8155218d89773405b1a7a6d7c2d33ce420899bb342d5fb07e',
    TABLE / 'compile-production-result-1790416485199864469.json': 'ed008129f370becbccd76c30963e07cee9aeb3499b0ae654db3bb560e77a2d31',
    TABLE / 'simulate-result-1790416504816097348.json': 'c50d6ace2f35cc737dcfce9e8735cb6a9b8643d3b19e28124ac2614f163aaafb',
    TABLE / 'simulate-result-1790416657288176259.json': '8b3a4b995f2a06458dd8b2bb4e3793b47d844b678f362f8af587e58ad48e3c3f',
    TABLE / 'artifacts-manifest.json': 'd76732c9ca0f955488035847e6596d3eda1d32bc8ef707f499417c3923f8ba99',
    COMPACT / 'compile-production-result-1790417895251858239.json': '147a300be275cb74b01e0ce2321f82a7e35dd9edf84944b25e6ad4dec80cda4a',
    COMPACT / 'simulate-result-1790417911971128655.json': '74d25e9302ba1ac64602fd33b191325f4372ac705857434026290a28b61d70cc',
    COMPACT / 'simulate-result-1790417933199676192.json': '6baa893345aa6899e90f4672f1d8cba391eb17e80df2713c594d25bcbd3c99b5',
    COMPACT / 'artifacts-manifest.json': 'ccf4914bbe35206bf2ee87866204a639eab6dc05e33a4eebea6c432212ddde40',
    REPO / 'research/v2/verification/20260926-tables-runtime-select/COMPOSITION.md': '07021ea1a6e6b35ef46e04951b1699a3993e244c7af83084b0263a718181f75e',
    REPO / 'research/v2/verification/20260926-k2-runtime-groups/COMPOSITION.md': 'aac28c5d267a38650b63f5dc12418b6d5aa774a50cbc5f46d6ec0525d6e94460',
    REPO / 'research/v2/verification/20260926-k2-runtime-groups/REVIEW.md': '58a3343e253f72533ba43304741605bcacff5a45e2e6a4ffedbc583221bfb028',
    REPO / 'research/v2/lanes/k2-last-layer-tables/review/REVIEW.md': 'c41e16d647658817b2456847d9e8221aeb1f8ff739f7d856577e406d99c5e435',
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for path, expected in PINS.items():
    assert sha(path) == expected, ('PIN_MISMATCH', str(path))

def split_factory(path, name):
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    tree = ast.parse(raw, filename=str(path))
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(matches) == 1
    node = matches[0]
    before = b''.join(lines[:node.lineno - 1])
    body = b''.join(lines[node.lineno - 1:node.end_lineno])
    after = b''.join(lines[node.end_lineno:])
    assert before + body + after == raw
    return before, body, after, tree

p_before, p_body, p_after, p_tree = split_factory(TABLE / 'train.py', '_k2_build_kernel')
c_before, c_body, c_after, c_tree = split_factory(HERE / 'train.py', '_k2_build_kernel')
_, donor_body, _, _ = split_factory(COMPACT / 'train.py', '_k2_build_kernel')
assert c_before == p_before and c_after == p_after, 'BYTES_OUTSIDE_K2_CHANGED'
assert c_body == donor_body, 'K2_NOT_BYTE_IDENTICAL_TO_COMPACT'
def other_nodes(tree):
    return [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name == '_k2_build_kernel')]
parent_nodes, candidate_nodes = other_nodes(p_tree), other_nodes(c_tree)
assert len(parent_nodes) == len(candidate_nodes)
assert all(ast.dump(a) == ast.dump(b) for a, b in zip(parent_nodes, candidate_nodes)), 'OTHER_AST_CHANGED'
unchanged_table_components = {}
for name in ('_k2_table_build_kernel', '_k2_table_build_registered_lookup',
             '_k2_prepare_metadata', '_k2_build_registered_lookup'):
    _, original, _, _ = split_factory(TABLE / 'train.py', name)
    _, actual, _, _ = split_factory(HERE / 'train.py', name)
    assert actual == original, ('UNCHANGED_COMPONENT_MISMATCH', name)
    unchanged_table_components[name] = hashlib.sha256(actual).hexdigest()

artifact_pins = {}
for lane in (TABLE, COMPACT):
    manifest = json.loads((lane / 'artifacts-manifest.json').read_text())
    assert manifest['source_sha256'] == PINS[lane / 'train.py']
    if lane == TABLE:
        entries = [(lane / name, digest) for name, digest in manifest['files'].items()]
    else:
        entries = [(lane / 'artifacts' / name, item['sha256'])
                   for name, item in manifest['artifacts'].items()]
    for path, expected in entries:
        assert sha(path) == expected, ('ARTIFACT_MISMATCH', str(path))
        artifact_pins[str(path.relative_to(REPO))] = expected

cpu = {}
for dtype in ('int32', 'int64'):
    path = HERE / ('cpu-result-' + dtype + '.json')
    receipt = json.loads(path.read_text())
    for filename, expected in receipt['sha256'].items():
        assert sha(Path(filename)) == expected, ('CPU_HASH_MISMATCH', filename)
    assert receipt['reference']['status'] == 'PASS_FP32_MASTER_REFERENCE'
    assert receipt['model']['off_3_steps'] == 'BIT_IDENTICAL'
    assert receipt['model']['on_2_steps'] == 'REGISTERED_VS_ORDINARY_BIT_IDENTICAL'
    assert receipt['native_executed'] is False
    cpu[dtype] = {'receipt_sha256': sha(path), 'reference': receipt['reference']['status'],
                  'off': receipt['model']['off_3_steps'], 'on': receipt['model']['on_2_steps']}
for filename in ('test_fp32_reference.py', 'cpu_instruction_model.py'):
    assert (HERE / filename).read_bytes() == (TABLE / filename).read_bytes()

result = {
    'status': 'PASS_EXACT_COMPONENT_COMPOSITION',
    'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source_sha256': PINS[HERE / 'train.py'],
    'source_diff_sha256': sha(HERE / 'SOURCE.diff'),
    'checker_sha256': sha(Path(__file__)),
    'compact_factory_sha256': hashlib.sha256(c_body).hexdigest(),
    'outside_compact_factory_byte_identical': True,
    'other_top_level_ast_nodes_identical': len(parent_nodes),
    'unchanged_components': unchanged_table_components,
    'pins': {str(path.relative_to(REPO)): digest for path, digest in PINS.items()},
    'reused_artifact_pins': artifact_pins,
    'cpu': cpu,
    'new_backend_or_simulator_execution': False,
    'new_independent_review': False,
    'host_calls': False,
    'launch_authorized': False,
    'scope': 'Implementer byte/AST/evidence binding; existing component reviews apply; no native qualification',
}
(HERE / 'composition-result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
