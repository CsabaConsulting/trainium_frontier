#!/usr/bin/env python3
"""Fresh reviewer proof: stdlib AST interpretation, no candidate code execution."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
CANDIDATE = ROOT / 'research/v2/proposals/20260929-r66-d2-upload/train.py'
PARENT = Path('/home/poojith-devan/Downloads/Trainium-D2-CONTINUE-CLAUDE-AUDIT-20260929-b758450b/train.py')
PARENT_SHA = 'b758450b599d4acc5b1fc17d37cccb87421d22bb99c213f6d8ed29fdede13fa8'
CANDIDATE_SHA = '6d197b3cd08a4e107020e5bb03a946a06a361ed36c6d7dfc12b35ca1484be14c'
ARCHIVES = [
    ('trn21', 'research/v2/operator/20260929-r62-d2-continue-full-trn21-preparation/artifacts/results-bundle.json', '5a82d3568173b16eb73d74e5b1484e9835c2b519181c1c68166bee4ecdc6da72', 'd3c2c030bcea19257a444342794ff26158fc77286f59b378fc84ff50e48f2fdb'),
    ('trn22', 'research/v2/operator/20260929-r62-d2-continue-confirm-full-trn22-preparation/artifacts/results-bundle.json', '88acaeb5597adced2408f6e38fa840f3489ed48579695e97ea9c502e1ffe1fd8', '1c56139d82380233cc0989c294b1f4a415d66bffec5152ae2f4de76c1568f222'),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def named(tree, name):
    hits = [n for n in tree.body if getattr(n, 'name', None) == name]
    require(len(hits) == 1, f'unique definition {name}')
    return hits[0]


def value(node, env):
    """Deliberately tiny AST interpreter. Never eval, exec, import or run train.py."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv):
        return value(node.left, env) // value(node.right, env)
    raise AssertionError(f'unapproved scalar expression: {ast.dump(node)}')


def parser_and_config(tree, argv):
    main = named(tree, 'main')
    require(len(main.body) >= 4, 'main parser setup')
    setup, outdir, freeze, parse = main.body[:4]
    require(ast.unparse(setup.value.func) == 'argparse.ArgumentParser', 'parser constructor')
    parser = argparse.ArgumentParser(**{k.arg: value(k.value, {}) for k in setup.value.keywords})
    for statement in (outdir, freeze):
        call = statement.value
        require(ast.unparse(call.func) == 'parser.add_argument', 'only parser.add_argument')
        flags = [value(arg, {}) for arg in call.args]
        kwargs = {}
        for kw in call.keywords:
            if kw.arg == 'type':
                require(isinstance(kw.value, ast.Name) and kw.value.id in ('int', 'str'), 'parser types')
                kwargs[kw.arg] = {'int': int, 'str': str}[kw.value.id]
            elif kw.arg == 'default' and isinstance(kw.value, ast.Call):
                require(ast.unparse(kw.value) == "os.environ.get('NEURON_COMPETITION_R1_OUT_DIR', 'out')", 'only optional out-dir environment lookup')
                kwargs[kw.arg] = 'out'
            else:
                kwargs[kw.arg] = value(kw.value, {})
        parser.add_argument(*flags, **kwargs)
    require(ast.unparse(parse) == 'args = parser.parse_args()', 'unmodified parse_args path')
    args = vars(parser.parse_args(argv))
    config_assignments = [n for n in main.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'config' for t in n.targets)]
    require(len(config_assignments) == 1, 'unique main config assignment')
    require(ast.unparse(config_assignments[0]) == 'config = build_config(tbl_freeze_tokens=args.tbl_freeze_tokens)', 'actual main passes parser flag directly')
    cls = named(tree, 'GPTConfig')
    require([ast.unparse(d) for d in cls.decorator_list] == ['dataclass'], 'GPTConfig dataclass')
    require(all(isinstance(n, ast.AnnAssign) for n in cls.body), 'config has only scalar fields')
    result = {n.target.id: value(n.value, {}) for n in cls.body if n.value is not None}
    required_fields = {n.target.id for n in cls.body if n.value is None}
    env = {'TOKENIZER_VOCAB_SIZE': 8192}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant):
            env[n.targets[0].id] = value(n.value, {})
    build = named(tree, 'build_config')
    require(len(build.body) == 2 and isinstance(build.body[0], ast.Expr) and isinstance(build.body[0].value, ast.Constant), 'build_config only docstring and return')
    require(isinstance(build.body[1], ast.Return) and ast.unparse(build.body[1].value.func) == 'GPTConfig', 'build_config returns GPTConfig directly')
    require(not build.args.args and not build.args.posonlyargs and not build.args.vararg and not build.args.kwarg, 'config keyword-only interface')
    defaults = {a.arg: value(v, {}) for a, v in zip(build.args.kwonlyargs, build.args.kw_defaults)}
    require(defaults['tbl_freeze_tokens'] == 286000000, 'direct build_config default unchanged')
    env.update(defaults)
    env['tbl_freeze_tokens'] = args['tbl_freeze_tokens']
    call = build.body[1].value
    require(not call.args, 'GPTConfig constructed using named fields only')
    result.update({k.arg: value(k.value, env) for k in call.keywords})
    require(required_fields <= result.keys(), 'all required config fields supplied')
    require(set(result) == {n.target.id for n in cls.body}, 'no extra model config fields')
    return args, result


def normalize(tree, original_default):
    tree = copy.deepcopy(tree)
    require(isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str), 'module docstring')
    tree.body[0].value.value = '<authorized header>'
    main = named(tree, 'main')
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'parser.add_argument' and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == '--tbl_freeze_tokens']
    require(len(calls) == 1, 'one freeze parser declaration')
    defaults = [k for k in calls[0].keywords if k.arg == 'default']
    require(len(defaults) == 1 and isinstance(defaults[0].value, ast.Constant) and defaults[0].value.value == original_default, 'expected freeze parser default')
    defaults[0].value.value = 0
    return tree


def main():
    parent = PARENT.read_bytes()
    candidate = CANDIDATE.read_bytes()
    require(sha(parent) == PARENT_SHA, 'immutable scored source SHA')
    require(sha(candidate) == CANDIDATE_SHA, 'reviewed candidate SHA')
    # The reviewer writes this allowlist from the authorization and visible diff;
    # EDITS.json, compose.py and the author verification implementation are unused.
    replacements = [
        (b'autograd uses PyTorch matrix operations. Parameters and optimizer state are\nFP32; normalized branch inputs and the residual stream are BF16 on Neuron,', b'autograd uses PyTorch matrix operations. Base parameters and optimizer state are\nFP32; the table and its RMS state are BF16; normalized branch inputs and the residual stream are BF16 on Neuron,'),
        (b'The main entrypoint trains from seed 42 and saves out/final.pt (the output\ndirectory is the only CLI option). It never reads validation data. The scorer', b'The main entrypoint trains from seed 42 and saves out/final.pt (CLI options:\n--out-dir and --tbl_freeze_tokens). It never reads validation data. The scorer'),
        (b"parser.add_argument('--tbl_freeze_tokens', type=int, default=286000000,", b"parser.add_argument('--tbl_freeze_tokens', type=int, default=0,"),
    ]
    rebuilt = parent
    for old, new in replacements:
        require(rebuilt.count(old) == 1 and parent.count(old) == 1, 'each allowed replacement occurs exactly once')
        rebuilt = rebuilt.replace(old, new, 1)
    require(rebuilt == candidate, 'complete byte identity after only three authorized replacements')
    p_tree = ast.parse(parent)
    c_tree = ast.parse(candidate)
    p_norm = normalize(p_tree, 286000000)
    c_norm = normalize(c_tree, 0)
    require(ast.dump(p_norm, include_attributes=False) == ast.dump(c_norm, include_attributes=False), 'complete normalized module AST identity')
    definitions = [n for n in p_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name != 'main']
    for node in definitions:
        require(ast.dump(node, include_attributes=False) == ast.dump(named(c_tree, node.name), include_attributes=False), f'unmodified {node.name}')
    bare_args, bare_config = parser_and_config(c_tree, [])
    require(bare_args == {'out_dir': 'out', 'tbl_freeze_tokens': 0}, 'bare parse CONTINUE')
    require(parser_and_config(p_tree, ['--tbl_freeze_tokens', '0']) == (bare_args, bare_config), 'parent explicit0 equals derivative bare')
    for choice in ('0', '286000000'):
        require(parser_and_config(p_tree, ['--tbl_freeze_tokens', choice]) == parser_and_config(c_tree, ['--tbl_freeze_tokens', choice]), 'explicit override equivalence')
    # Authenticate the tokenizer-size scalar directly without importing prepare.
    prep = ast.parse((ROOT / 'prepare.py').read_bytes())
    sizes = [n for n in prep.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TOKENIZER_VOCAB_SIZE' for t in n.targets)]
    require(len(sizes) == 1 and value(sizes[0].value, {}) == 8192, 'trusted tokenizer vocab size')
    evidence = []
    for host, relative, archive_sha, log_sha in ARCHIVES:
        data = (ROOT / relative).read_bytes()
        require(sha(data) == archive_sha, f'{host} archive SHA')
        archive = json.loads(data)
        files = archive['files']
        for name, item in files.items():
            require(isinstance(item.get('text'), str), f'{host}:{name} text member')
            content = item['text'].encode()
            require(len(content) == item['bytes'] and sha(content) == item['sha256'], f'{host}:{name} content identity')
        require(files['package/train.py']['text'].encode() == parent, f'{host} actual package source bytes')
        log = files['training.log']['text']
        require(sha(log.encode()) == log_sha, f'{host} actual training.log hash')
        require(log.count('model_config=') == 1, f'{host} unique logged config')
        recorded_config, _ = json.JSONDecoder().raw_decode(log.split('model_config=', 1)[1].lstrip())
        require(recorded_config == bare_config, f'{host} actual logged config equals derivative bare config')
        command = files['package/launch-command.txt']['text'].strip()
        require(command == 'NEURON_LOGICAL_NC_CONFIG=2 torchrun --standalone --nproc_per_node=4 train.py --tbl_freeze_tokens 0', f'{host} actual explicit0 command')
        command_lines = [s for s in log.splitlines() if s.startswith('REHEARSAL_COMMAND ')]
        require(len(command_lines) == 1, f'{host} unique actual argv')
        argv = json.loads(command_lines[0].removeprefix('REHEARSAL_COMMAND '))
        require(argv == ['torchrun', '--standalone', '--nproc_per_node=4', 'train.py', '--tbl_freeze_tokens', '0'], f'{host} exact executed argv')
        train = json.loads(files['train/rehearsal.json']['text'])
        evaluation = json.loads(files['eval/rehearsal.json']['text'])
        terminal = json.loads(files['terminal.json']['text'])
        require(train['returncode'] == evaluation['returncode'] == terminal['eval_exit_code'] == 0, f'{host} successful recorded execution')
        require(files['run-exit-code.txt']['text'].strip() == files['eval-exit-code.txt']['text'].strip() == '0', f'{host} zero exit files')
        require(train['startup_allowance_observed'] < 900 and train['charged_through_process_exit'] < 1800 and train['charged_through_process_exit_within_budget'] is True, f'{host} recorded cold/budget gate')
        require(train['checkpoint_bytes'] == 588313401, f'{host} recorded checkpoint present')
        evidence.append({'host': host, 'archive_path': relative, 'archive_sha256': sha(data), 'authenticated_members': len(files), 'training_log_sha256': log_sha, 'model_config_equal': True, 'actual_argv': argv, 'cold_seconds': train['startup_allowance_observed'], 'charged_seconds': train['charged_through_process_exit'], 'checkpoint_sha256_recorded': train['checkpoint_sha256']})
    require('torch' not in sys.modules, 'no torch imported')
    require(PARENT.read_bytes() == parent and CANDIDATE.read_bytes() == candidate, 'review inputs unchanged')
    receipt = {
        'verdict': 'PASS_INDEPENDENT_R66_DERIVATIVE_EXACTNESS',
        'reviewer': '/root/r66_upload_fresh_review',
        'check_script_sha256': sha(Path(__file__).read_bytes()),
        'parent_sha256': sha(parent), 'candidate_sha256': sha(candidate), 'candidate_bytes': len(candidate),
        'authorized_replacements': 3, 'complete_byte_allowlist_match': True,
        'normalized_entire_module_ast_equal': True, 'unchanged_top_level_definitions_excluding_main': len(definitions),
        'bare_args': bare_args, 'bare_model_config': bare_config,
        'direct_build_config_default': 286000000,
        'both_explicit_override_modes_equal_parent': True,
        'scored_evidence': evidence,
        'method': 'Reviewer-authored scalar AST interpreter plus stdlib argparse; no eval/exec, candidate imports, author code, model construction or numerical execution.',
        'scope': 'R66 derivative exactness and recorded config equivalence only; Claude conditional source audit remains bounded, no new native/clean-box/score claim.',
        'source_and_package_writes': False, 'host_calls': False, 'submission': False,
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
