"""Reproduce exact6ef7 owner plan from shape metadata; stdlib only, no tensors."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

SOURCE_SHA = '6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3'
EXPECTED_PLAN = 'b89d62227c844eb13706fbe81bddfb9b8b7db5fefcd57b4e7d0ea3213dca5031'


class ParameterMetadata:
    def __init__(self, shape):
        self.shape, self.dtype = shape, 'float32'

    def numel(self):
        return math.prod(self.shape)

    def is_contiguous(self):
        return True


def audit(source):
    raw = Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('requires exact6ef7 source')
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == 'make_plan')
    # From exact GPT.setup_optimizer's dense6x1024/no-ngram/no-qk grouping.
    specs = [('adamw', (8192, 1024), 1), ('adamw', (8192, 1024), 1),
        ('adamw', (1,), 12), ('adamw', (8192, 1024), 3), ('adamw', (1024, 1, 2), 6),
        ('muon', (8, 32), 3), ('muon', (1024, 1024), 24),
        ('muon', (1024, 4096), 6), ('muon', (4096, 1024), 6)]
    groups = [dict(kind=kind, params=[ParameterMetadata(shape) for _ in range(count)])
              for kind, shape, count in specs]
    namespace = dict(torch=SimpleNamespace(float32='float32'))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<exact6ef7 make_plan>', 'exec'), namespace)
    entries, parameters, summary = namespace['make_plan'](groups, 4)
    canonical = json.dumps(dict(entries=entries, summary=summary), sort_keys=True)
    plan_sha = hashlib.sha256(canonical.encode()).hexdigest()
    if plan_sha != EXPECTED_PLAN:
        raise ValueError('does not match archived exact-size G2 plan')
    ranks = []
    for rank in range(4):
        rows = []
        for group, (kind, shape, _) in enumerate(specs):
            if kind != 'muon':
                continue
            positions = [e['position'] for e in entries if e['group'] == group and e['owner'] == rank]
            count = len(positions)
            rows.append(dict(shape=shape, owned_count=count, executed_batch=count + (count == 1),
                             positions=positions))
        ranks.append(dict(rank=rank, muon_groups=rows))
    return dict(source_sha256=SOURCE_SHA, plan_sha256=plan_sha, parameters=len(parameters),
        elements=sum(p.numel() for p in parameters), summary=summary, ranks=ranks,
        scope='Exact make_plan AST, metadata only; no model/tensor construction or optimizer numerics',
        archived_match='research/v2/operator/20260925-trn21-recursive116188-g2-r1/reference-on/results/g2.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-py', default='workflow/proposals/research-v2/train.py')
    args = parser.parse_args()
    print(json.dumps(audit(args.train_py), indent=2))
