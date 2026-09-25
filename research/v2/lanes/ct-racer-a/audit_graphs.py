"""Stdlib-only immutable-source and archived-graph audit; no torch or host calls."""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
SOURCES = {
    'recursive6ef7': ('a4aa4351789115a2d0aa947d9ca00ff689f9bd0b', '6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3'),
    'architecture93ae': ('1e21c3d797eed764f108c41546b270065ad5650c', '93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5'),
    'primed6bcbd': ('b635c8b0cf2a10e1bcdd983e86192672ed6ad56f', 'bcbd148dd51b5a014289cbecf3f0e684e35cf537e9588c878e15b265fcbfd522'),
    'primed8_3e79': ('b4d83b5db66971f3c348837f04f53cba65aa6d33', '3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c'),
}
GRAPH_REF = 'b635c8b0cf2a10e1bcdd983e86192672ed6ad56f'
GRAPH_PATH = 'research/v2/lanes/initial-cache-refresh/graph-inventory.json'
ARCHIVE_REF = 'c2c47f7dcc1717eb3d176b1bb01622efe891c09f'
ARCHIVE_DIR = 'research/v2/operator/20260925-trn22-l2-93ae-cpu-r1/native'


def git_bytes(ref, path):
    return subprocess.check_output(['git', 'show', ref + ':' + path], cwd=ROOT)


def need(value, error):
    if not value:
        raise RuntimeError(error)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def source_audit(ref, expected):
    raw = git_bytes(ref, 'workflow/proposals/research-v2/train.py')
    need(sha(raw) == expected, 'SOURCE_SHA_MISMATCH')
    tree = ast.parse(raw)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    def owner(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
        return '<module>'
    compiles = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == 'torch.compile':
            compiles.append(dict(line=node.lineno, owner=owner(node), call=ast.unparse(node)))
    references = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in {
                'Linear', 'load_for_eval', 'lr_multiplier', 'muon_beta2', 'muon_momentum',
                '_muon_math', '_muon_compile_scalars', '_muon_step', '_device_scalar',
                '_adamw_state', '_adamw_step_foreach', '_submission_mlp_install',
                '_install_submission_training', 'make_forward', 'OwnedOptimizer'}:
            references[node.name] = dict(line=node.lineno,
                ast_sha256=sha(ast.dump(node, include_attributes=False).encode()))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    loop = next(n for n in main.body if isinstance(n, ast.While))
    eval_if = next(n for n in main.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'args.eval_public')
    eval_loader = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_for_eval')
    need(not any(isinstance(n, ast.Call) and ast.unparse(n.func) == 'torch.compile' for n in ast.walk(eval_loader)), 'UNEXPECTED_EVAL_COMPILE')
    need(not any(isinstance(n, ast.Call) and ast.unparse(n.func) in ('evaluate_bpb', 'causality_check') for n in ast.walk(loop)), 'UNEXPECTED_TRAIN_LOOP_EVAL')
    need(len(compiles) == 3, 'COMPILE_SITE_COUNT_CHANGED')
    return dict(commit=ref, source_sha256=expected, compile_sites=sorted(compiles, key=lambda x:x['line']),
                functions=references, training_loop_line=loop.lineno, post_loop_eval_guard_line=eval_if.lineno,
                eval_contains_explicit_torch_compile=False, evaluation_inside_training_loop=False)


def audit():
    sources = {name: source_audit(*identity) for name, identity in SOURCES.items()}
    raw = git_bytes(GRAPH_REF, GRAPH_PATH)
    archive = json.loads(raw)
    groups = defaultdict(list)
    ranks = []
    for rank in archive['ranks']:
        for graph in rank['graphs']:
            groups[graph['id']].append(dict(rank=rank['rank'], **graph))
        roles = defaultdict(int)
        for graph in rank['graphs']:
            role = 'model_forward' if '_forward_' in graph['name'] else 'model_backward' if '_backward_' in graph['name'] else 'inference_no_backward'
            roles[role] += 1
        ranks.append(dict(rank=rank['rank'], recorded_graphs=rank['graph_count'], roles=dict(roles)))
    model_groups = {key: rows for key, rows in groups.items() if any('_forward_' in r['name'] or '_backward_' in r['name'] for r in rows)}
    need(len(model_groups) == 4, 'EXPECTED_TWO_MODEL_SPECIALIZATIONS')
    need(all(sorted(r['rank'] for r in rows) == [0,1,2,3] for rows in model_groups.values()), 'MODEL_IDS_NOT_IDENTICAL_ACROSS_RANKS')
    capture = json.loads(git_bytes(ARCHIVE_REF, ARCHIVE_DIR + '/live-compiler-input/capture.json'))
    mlir = git_bytes(ARCHIVE_REF, ARCHIVE_DIR + '/live-compiler-input/module.mlir')
    need(sha(mlir) == capture['input_sha256'], 'MLIR_ARCHIVE_SHA_MISMATCH')
    need(any(key.split(':')[0] == sha(mlir) for key in model_groups), 'INPUT_NOT_IN_MODEL_INVENTORY')
    compiler_args = [p['argv'] for p in capture['live_compiler_processes'] if 'neuronx-cc' in ' '.join(p['argv'])]
    return dict(scope='Read-only stdlib source/archive evidence; no graph execution or native claim',
        sources=sources, archived_inventory=dict(commit=GRAPH_REF, path=GRAPH_PATH, sha256=sha(raw)),
        native_inventory=dict(source_sha256=SOURCES['architecture93ae'][1], per_rank=ranks,
            records=sum(len(rows) for rows in groups.values()), unique_recorded_ids=len(groups),
            model_ids_shared_by_all_four_ranks=sorted(model_groups),
            unique_inference_ids=sorted(set(groups)-set(model_groups)),
            groups=[dict(id=key, observations=rows) for key, rows in sorted(groups.items())],
            interpretation='Graph IDs include input bytes and SDK-version material; not full persistent NEFF keys. Zero compile_us remains unknown. Counts exclude unreported eager/NKI/internal partition graphs.'),
        captured_first_forward=dict(commit=ARCHIVE_REF, sha256=sha(mlir), bytes=len(mlir),
            utc=capture['utc'], compiler_process_count=len(compiler_args), distinct_compiler_argv=sorted({json.dumps(a) for a in compiler_args}),
            process_mapping_limit='Relative module.mlir argv plus snapshot does not identify each process cwd/input, parentage, or duplicated versus partition compiler work. No fourfold-waste claim.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    report = audit()
    content = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.out:
        with args.out.open('x') as stream:
            stream.write(content)
    else:
        print(content, end='')
