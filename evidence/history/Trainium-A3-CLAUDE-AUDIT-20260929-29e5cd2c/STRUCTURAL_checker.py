"""Independent stdlib-only extraction; never imports train.py or prepare.py."""
import argparse
import ast
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
PARENT = ROOT / 'research/v2/proposals/20260929-r69-a3-noproj-id/train.py'
CANDIDATE = ROOT / 'research/v2/proposals/20260929-r73-a3-upload/train.py'
ARCHIVE = ROOT / 'research/v2/operator/20260929-r71-a3-full-trn21-preparation/artifacts/results-bundle.json'
def sha(data):
    return hashlib.sha256(data).hexdigest()
assert sha(PARENT.read_bytes()) == 'c5ef8414e2df998a9b52eeb2e249b66c0110a3c37869e676719e93975a3432b4'
assert sha(CANDIDATE.read_bytes()) == '29e5cd2cfd35bb52e5a955057a1661786239d0812d9a797c85dc0df7fe2d7446'
assert sha(ARCHIVE.read_bytes()) == 'd38a9edd82b3c451a2f01825a404281a87c3c1fd9a3c03a1b73b4b9656c55e4b'
archive = json.loads(ARCHIVE.read_text())
for name, member in archive['files'].items():
    if 'text' in member:
        data = member['text'].encode()
        assert sha(data) == member['sha256'], name
        assert len(data) == member['bytes'], name
assert archive['files']['package/train.py']['text'].encode() == PARENT.read_bytes()
command = archive['files']['package/launch-command.txt']['text'].strip()
assert command == 'NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0'
log = archive['files']['training.log']['text']
recorded, _ = json.JSONDecoder().raw_decode(log.split('model_config=', 1)[1])
trees = [ast.parse(p.read_text()) for p in (PARENT, CANDIDATE)]
def main(tree):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
normalized = copy.deepcopy(trees)
for tree in normalized:
    assert isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant)
    tree.body[0].value.value = 'MODULE_DOCUMENTATION_ONLY'
    statement = main(tree).body[2]
    assert ast.literal_eval(statement.value.args[0]) == '--tbl_freeze_tokens'
    default = next(k for k in statement.value.keywords if k.arg == 'default')
    assert default.value.value in (0, 286000000)
    default.value.value = 0
assert ast.dump(normalized[0], include_attributes=False) == ast.dump(normalized[1], include_attributes=False)

def extracted_config(tree, argv, env):
    # Execute only the three inspected parser construction statements, not main.
    parser_nodes = copy.deepcopy(main(tree).body[:3])
    assert all(isinstance(n, (ast.Assign, ast.Expr)) for n in parser_nodes)
    ns = {'argparse': argparse, 'os': SimpleNamespace(environ=env)}
    exec(compile(ast.Module(body=parser_nodes, type_ignores=[]), '<parser-only>', 'exec'), ns)
    args = ns['parser'].parse_args(argv)
    # Constants and dataclass/build_config only. No tensor/model/import nodes.
    names = {'SEQ_LEN', 'DEPTH', 'HEAD_DIM', 'N_EMBD', 'MLP_RATIO', 'ATTN_SCALE', 'N_KV_HEADS'}
    constants = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names]
    cfg_ns = {'dataclass': dataclasses.dataclass, 'TOKENIZER_VOCAB_SIZE': 8192}
    for n in constants:
        cfg_ns[n.targets[0].id] = ast.literal_eval(n.value)
    assert names <= cfg_ns.keys()
    definitions = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in ('GPTConfig', 'build_config')]
    assert len(definitions) == 2
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<config-only>', 'exec'), cfg_ns)
    config_call = next(n.value for n in main(tree).body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'config' for t in n.targets))
    assert ast.unparse(config_call) == 'build_config(tbl_freeze_tokens=args.tbl_freeze_tokens)'
    config = cfg_ns['build_config'](tbl_freeze_tokens=args.tbl_freeze_tokens)
    return vars(args), dataclasses.asdict(config), cfg_ns['build_config'].__kwdefaults__['tbl_freeze_tokens']

for env in ({}, {'NEURON_COMPETITION_R1_OUT_DIR': '/isolated/test-output'}):
    old = extracted_config(trees[0], ['--tbl_freeze_tokens', '0'], env)
    new = extracted_config(trees[1], [], env)
    assert old == new
    assert new[1] == recorded
    assert new[2] == 286000000  # Deliberately unchanged direct helper default.
assert extracted_config(trees[0], [], {})[0]['tbl_freeze_tokens'] == 286000000
assert extracted_config(trees[1], [], {})[0]['tbl_freeze_tokens'] == 0
result = {
    'status': 'PASS_INDEPENDENT_STRUCTURAL_UPLOAD_DERIVATIVE_ONLY',
    'source_sha256': sha(CANDIDATE.read_bytes()), 'scored_parent_sha256': sha(PARENT.read_bytes()),
    'archive_sha256': sha(ARCHIVE.read_bytes()), 'checker_sha256': sha(Path(__file__).read_bytes()),
    'whole_module_ast_equal_after_only_docstring_and_parser_default_normalization': True,
    'model_optimizer_eval_and_main_remainder_unchanged': True,
    'bare_candidate_matches_scored_parent_explicit_zero_args_and_recorded_config': True,
    'recorded_config': recorded, 'recorded_command': command,
    'unchanged_direct_build_config_default': 286000000,
    'scope': 'stdlib parser/config extraction only; no model, Torch, prepare import, numerical or host execution',
    'pending': ['all three scored receipts and mean <= .97225', 'all cold/budget/checklist and packaging authority checks'],
    'adoption_authorized': False, 'upload_authorized': False, 'claude_compliance_acceptance': False,
}
print(json.dumps(result, indent=2, sort_keys=True))
