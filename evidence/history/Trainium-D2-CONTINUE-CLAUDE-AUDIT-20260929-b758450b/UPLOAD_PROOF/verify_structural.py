"""Exact authorized-edit/config proof only. No Torch, source-module or model import."""
import argparse
import ast
import copy
import dataclasses
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCE_SHA = '6d197b3cd08a4e107020e5bb03a946a06a361ed36c6d7dfc12b35ca1484be14c'
PARENT_SHA = 'b758450b599d4acc5b1fc17d37cccb87421d22bb99c213f6d8ed29fdede13fa8'
PREPARE_SHA = '5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb'
AUDIT = Path('/home/poojith-devan/Downloads/Trainium-D2-CONTINUE-CLAUDE-AUDIT-20260929-b758450b/CLAUDE_AUDIT_RESULT.md')
AUDIT_SHA = '55be15b181201ef89a420d6048081a60c417e55836a51d5e8d7a7a7bfa745df9'
BARE_COMMAND = 'NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py'
SCORED_COMMAND = BARE_COMMAND + ' --tbl_freeze_tokens 0'
ARCHIVES = [
    ('trn21', 'research/v2/operator/20260929-r62-d2-continue-full-trn21-preparation/artifacts/results-bundle.json', '5a82d3568173b16eb73d74e5b1484e9835c2b519181c1c68166bee4ecdc6da72'),
    ('trn22', 'research/v2/operator/20260929-r62-d2-continue-confirm-full-trn22-preparation/artifacts/results-bundle.json', '88acaeb5597adced2408f6e38fa840f3489ed48579695e97ea9c502e1ffe1fd8'),
]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def named(tree, name):
    return next(node for node in tree.body if getattr(node, 'name', None) == name)


def dump(node):
    return ast.dump(node, include_attributes=False)


def freeze_argument(tree):
    calls = [node for node in ast.walk(named(tree, 'main'))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name) and node.func.value.id == 'parser'
             and node.func.attr == 'add_argument' and node.args
             and isinstance(node.args[0], ast.Constant) and node.args[0].value == '--tbl_freeze_tokens']
    assert len(calls) == 1
    defaults = [keyword for keyword in calls[0].keywords if keyword.arg == 'default']
    assert len(defaults) == 1 and isinstance(defaults[0].value, ast.Constant)
    return defaults[0]


def normalized(tree, expected_default):
    tree = copy.deepcopy(tree)
    assert isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant)
    assert isinstance(tree.body[0].value.value, str)
    tree.body[0].value.value = '<authorized-header-normalization>'
    default = freeze_argument(tree)
    assert default.value.value == expected_default
    default.value.value = 286000000
    return tree


def extracted_config(tree, argv):
    """Execute only copied argparse statements, scalar constants, dataclass and build_config."""
    main = named(tree, 'main')
    statements = copy.deepcopy(main.body[:4])
    assert isinstance(statements[0], ast.Assign) and statements[0].targets[0].id == 'parser'
    assert [statement.value.args[0].value for statement in statements[1:3]] == ['--out-dir','--tbl_freeze_tokens']
    assert isinstance(statements[3], ast.Assign) and statements[3].targets[0].id == 'args'
    assert dump(statements[3].value) == dump(ast.parse('parser.parse_args()', mode='eval').body)
    # Only the proof harness supplies argv; candidate parser code stays unchanged.
    statements[3].value.args = [ast.List(elts=[ast.Constant(value=item) for item in argv], ctx=ast.Load())]
    constant_names = ('SEQ_LEN','DEPTH','N_EMBD','HEAD_DIM','MLP_RATIO','ATTN_SCALE','N_KV_HEADS')
    values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id in constant_names}
    prepare = (REPO / 'prepare.py').read_bytes()
    assert sha(prepare) == PREPARE_SHA
    ptree = ast.parse(prepare)
    values['TOKENIZER_VOCAB_SIZE'] = next(ast.literal_eval(node.value) for node in ptree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'TOKENIZER_VOCAB_SIZE')
    values.update(argparse=argparse, os=SimpleNamespace(environ={}), dataclass=dataclasses.dataclass)
    selected = [copy.deepcopy(named(tree, 'GPTConfig')), copy.deepcopy(named(tree, 'build_config'))] + statements
    ast.fix_missing_locations(module := ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, '<pure-argparse-and-config-only>', 'exec'), values)
    calls = [node for node in main.body if isinstance(node, ast.Assign)
             and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'config']
    assert len(calls) == 1
    assert dump(calls[0].value) == dump(ast.parse('build_config(tbl_freeze_tokens=args.tbl_freeze_tokens)', mode='eval').body)
    cfg = values['build_config'](tbl_freeze_tokens=values['args'].tbl_freeze_tokens)
    direct = values['build_config']()
    assert direct.tbl_freeze_tokens == 286000000  # Deliberately NOT changed.
    return dataclasses.asdict(cfg), vars(values['args'])


def authenticate_scored_archive(host, relative_path, expected_sha, parent, expected_config):
    path = REPO / relative_path
    raw = path.read_bytes()
    assert sha(raw) == expected_sha
    archive = json.loads(raw)
    files = archive['files']
    for name, record in files.items():
        content = record['text'].encode()
        assert len(content) == record['bytes'] and sha(content) == record['sha256'], name
    assert files['package/train.py']['text'].encode() == parent
    assert files['package/train.py']['sha256'] == PARENT_SHA
    command = files['package/launch-command.txt']['text'].strip()
    assert command == SCORED_COMMAND
    args = shlex.split(command)
    training_argv = args[args.index('train.py') + 1:]
    parsed_config, parsed_args = extracted_config(ast.parse(parent), training_argv)
    assert parsed_args['tbl_freeze_tokens'] == 0 and parsed_config == expected_config
    log = files['training.log']['text']
    assert log.count('model_config=') == 1
    recorded, _ = json.JSONDecoder().raw_decode(log.split('model_config=', 1)[1])
    assert recorded == expected_config
    terminal = json.loads(files['terminal.json']['text'])
    assert terminal['status'] == 'COMPLETE_PENDING_INDEPENDENT_REVIEW'
    assert terminal['eval_exit_code'] == 0
    rehearsal = json.loads(files['train/rehearsal.json']['text'])
    assert rehearsal['returncode'] == 0
    return dict(host=host, archive_path=relative_path, archive_sha256=expected_sha,
                authenticated_members=len(files), training_log_sha256=files['training.log']['sha256'],
                original_source_sha256=PARENT_SHA, actual_package_command=command,
                actual_training_argv=training_argv, model_config=recorded, config_matches=True)


def verify():
    spec = importlib.util.spec_from_file_location('upload_composer_stdlib', HERE / 'compose.py')
    composer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(composer)
    parent = composer.PARENT.read_bytes()
    source = (HERE / 'train.py').read_bytes()
    assert sha(parent) == PARENT_SHA and sha(source) == SOURCE_SHA
    assert source == composer.compose(parent)
    assert len(composer.EDITS) == 3
    assert json.loads((HERE / 'EDITS.json').read_bytes()) == [dict(before=a,after=b) for a,b in composer.EDITS]
    restored = source.decode()
    for before, after in reversed(composer.EDITS):
        assert restored.count(after) == 1
        restored = restored.replace(after, before)
    assert restored.encode() == parent
    assert sha(AUDIT.read_bytes()) == AUDIT_SHA
    assert (AUDIT.parent / 'train.py').read_bytes() == parent
    ptree, tree = ast.parse(parent), ast.parse(source)
    pnorm, norm = normalized(ptree, 286000000), normalized(tree, 0)
    assert dump(pnorm) == dump(norm)
    function_pairs = list(zip([node for node in ast.walk(pnorm) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))],
                              [node for node in ast.walk(norm) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))]))
    for first, second in function_pairs:
        assert first.name == second.name and dump(first) == dump(second), first.name
    for name in ('GPT','GPTConfig','CausalSelfAttention','Block','DetachedTableUpdate','build_config','load_for_eval','save_checkpoint'):
        assert dump(named(ptree,name)) == dump(named(tree,name)), name
    cfg, args = extracted_config(tree, [])
    assert args == {'out_dir':'out','tbl_freeze_tokens':0}
    explicit_cfg, explicit_args = extracted_config(tree, ['--tbl_freeze_tokens','0'])
    assert explicit_cfg == cfg and explicit_args == args
    old_bare_cfg, old_bare_args = extracted_config(ptree, [])
    assert old_bare_args['tbl_freeze_tokens'] == old_bare_cfg['tbl_freeze_tokens'] == 286000000
    for argv in (['--tbl_freeze_tokens','0'], ['--tbl_freeze_tokens','286000000']):
        assert extracted_config(tree,argv) == extracted_config(ptree,argv)
    archives = [authenticate_scored_archive(*row,parent,cfg) for row in ARCHIVES]
    assert not any(name == 'torch' or name.startswith(('torch.','nki','neuron')) for name in sys.modules)
    assert sha(AUDIT.read_bytes()) == AUDIT_SHA
    return dict(status='PASS_AUTHOR_STRUCTURAL_UPLOAD_DERIVATION', source_sha256=SOURCE_SHA,
        source_bytes=len(source), parent_sha256=PARENT_SHA, authorized_replacements=3,
        normalized_module_AST_identical=True, normalized_function_AST_count=len(function_pairs),
        model_and_eval_AST_unchanged=True, direct_build_config_default_unchanged=286000000,
        bare_parsed_args=args, bare_main_model_config=cfg,
        bare_main_config_canonical_json=json.dumps(cfg,sort_keys=True,separators=(',',':')),
        explicit_zero_and_286M_override_equivalent_to_parent=True,
        scored_archives=archives, audit_path=str(AUDIT), audit_sha256=AUDIT_SHA,
        package_and_audit_unchanged=True, declared_upload_command=BARE_COMMAND,
        proof_script_sha256=sha(Path(__file__).read_bytes()),
        scope='Pure argparse/config/dataclass extraction and exact source/recorded-config equivalence; not a new training, numerical, clean-box or independent compliance verdict',
        torch_imported=False, model_imported=False, numerical_execution=False, host_calls=False,
        independent_review=False, exported=False, submitted=False)


if __name__ == '__main__':
    result = verify()
    data = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    target = HERE / 'STRUCTURAL_RECEIPT.json'
    if target.exists():
        assert target.read_text() == data, 'FROZEN_RECEIPT_MISMATCH'
    else:
        patch = '*** Begin Patch\n*** Add File: ' + str(target) + '\n' + ''.join('+' + line + '\n' for line in data.splitlines()) + '*** End Patch\n'
        subprocess.run(['apply_patch'], input=patch, text=True, check=True, capture_output=True)
    print(json.dumps(dict(status=result['status'], source_sha256=SOURCE_SHA,
                          normalized_functions=result['normalized_function_AST_count'],
                          matched_scored_logs=len(result['scored_archives']), receipt_sha256=sha(data.encode()))))
