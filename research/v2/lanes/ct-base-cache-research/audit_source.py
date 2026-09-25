"""Read-only stdlib audit of existing cache-refresh sources and receipts."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
SOURCE = 'workflow/proposals/research-v2/train.py'
SPECS = {
    'base': ('a4aa4351789115a2d0aa947d9ca00ff689f9bd0b', '6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3'),
    'base_port': ('b635c8b0cf2a10e1bcdd983e86192672ed6ad56f', 'bcbd148dd51b5a014289cbecf3f0e684e35cf537e9588c878e15b265fcbfd522'),
    'arch': ('1e21c3d797eed764f108c41546b270065ad5650c', '93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5'),
    'arch_port': ('b4d83b5db66971f3c348837f04f53cba65aa6d33', '3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c'),
}
ADDED = [
    "    parser.add_argument('--initial-cache-refresh', action=argparse.BooleanOptionalAction, default=False, help='Refresh differentiable weight casts inside the timed first step before its first forward; default OFF.')\n",
    '        if step == 0 and args.initial_cache_refresh:\n'
    '            for m in cached_linears:\n'
    '                m.refresh_weight_cache(COMPUTE_DTYPE)\n',
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(relative):
    return json.loads((ROOT / relative).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    texts, trees, sources = {}, {}, {}
    for name, (commit, expected) in SPECS.items():
        raw = subprocess.check_output(['git', 'show', commit + ':' + SOURCE], cwd=ROOT)
        actual = hashlib.sha256(raw).hexdigest()
        require(actual == expected, name + ': source hash')
        texts[name], trees[name] = raw.decode(), ast.parse(raw)
        sources[name] = dict(commit=commit, sha256=actual)
    for base, port in [('base', 'base_port'), ('arch', 'arch_port')]:
        restored = texts[port]
        for addition in ADDED:
            require(restored.count(addition) == 1, port + ': addition count')
            restored = restored.replace(addition, '', 1)
        require(restored == texts[base], port + ': extra source change')
    definitions = {}
    for name in ['Linear', 'make_forward', '_submission_mlp_install', '_install_submission_training', 'OwnedOptimizer']:
        nodes = {key: next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
                 for key, tree in trees.items()}
        equal = ast.dump(nodes['base']) == ast.dump(nodes['arch'])
        require(equal, name + ': base/architecture AST mismatch')
        definitions[name] = {'base_arch_ast_identical': equal,
                             'lines': {key: [node.lineno, node.end_lineno] for key, node in nodes.items()}}
    prior = ROOT / 'research/v2/lanes/initial-cache-refresh'
    receipts = {mode: json.loads((prior / ('tiny-' + mode + '.json')).read_text())
                for mode in ('base', 'default', 'off', 'on')}
    reference = receipts['base']
    receipt_summary = {}
    for mode, receipt in receipts.items():
        require(receipt['torchrun_rc'] == 0 and receipt['steps'] == 3 and receipt['nproc'] == 4, mode + ': receipt shape')
        expected = SPECS['base' if mode == 'base' else 'base_port'][1]
        require(receipt['train_py_sha256'] == expected, mode + ': receipt source')
        require(receipt['plan_sha256'] == reference['plan_sha256'], mode + ': ownership')
        for actual, ref in zip(receipt['ranks'], reference['ranks']):
            require(actual['losses'] == ref['losses'], mode + ': losses')
            obs, robs = actual['cache_observations'], ref['cache_observations']
            require(obs['steps'] == robs['steps'], mode + ': gradients/parameters/optimizer')
            require(obs.get('checkpoint') == robs.get('checkpoint'), mode + ': checkpoint payload')
            require(obs.get('checkpoint_file_sha256') == robs.get('checkpoint_file_sha256'), mode + ': checkpoint bytes')
        obs0 = receipt['ranks'][0]['cache_observations']
        receipt_summary[mode] = {'nproc': 4, 'steps': 3, 'micro_batch': receipt['micro_batch'],
                                'cached_module_names': sorted(obs0['forwards'][0]['caches']),
                                'refresh_count_rank0': len(obs0['cache_refreshes']),
                                'losses_gradients_updates_optimizer_checkpoint_equal_to_base': True}
    terminal = read_json('research/v2/operator/20260926-trn21-profile-recursive6ef7/completed/profile/rank0/terminal.json')
    cold_base = read_json('research/v2/operator/20260926-ct-base-startup-only/completed/train/rehearsal.json')
    cold_arch = read_json('research/v2/operator/20260926-ct-architecture-startup-only/completed/train/rehearsal.json')
    result = {'scope': 'Researcher stdlib source/receipt audit; no tensor allocation or new verification authority',
              'sources': sources, 'four_added_lines_only_both_ports': True, 'definitions': definitions,
              'historical_tiny_receipts': receipt_summary,
              'base_profile_first_two_steps': terminal['warmup_and_profile_boundary_observations'][:2],
              'base_cold_seconds': cold_base['startup_allowance_observed'],
              'arch_port_cold_seconds': cold_arch['startup_allowance_observed'],
              'native_base_port_result': None, 'base_cold_target_seconds': 600,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'four_line_ports': True, 'identical_core_definitions': list(definitions),
                      'historical_base_cache_count': len(receipt_summary['on']['cached_module_names']),
                      'native_base_port_result': None}))


if __name__ == '__main__':
    main()
