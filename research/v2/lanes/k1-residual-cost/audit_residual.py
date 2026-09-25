"""Current6ef7 residual cost counts: stdlib metadata only, no candidate code."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
OLD_AUDIT = ROOT / "research/v2/lanes/k1-muon-batching/audit_plan.py"
SOURCE = ROOT / "workflow/proposals/research-v2/train.py"
spec = importlib.util.spec_from_file_location("existing_k1_metadata_audit", OLD_AUDIT)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def audit():
    checked = prior.audit(SOURCE)
    specs = [('adamw', (8192, 1024), 1), ('adamw', (8192, 1024), 1),
             ('adamw', (1,), 12), ('adamw', (8192, 1024), 3),
             ('adamw', (1024, 1, 2), 6), ('muon', (8, 32), 3),
             ('muon', (1024, 1024), 24), ('muon', (1024, 4096), 6),
             ('muon', (4096, 1024), 6)]
    groups = [dict(kind=kind, params=[prior.ParameterMetadata(shape) for _ in range(count)])
              for kind, shape, count in specs]
    node = next(n for n in ast.parse(SOURCE.read_bytes()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'make_plan')
    scope = dict(torch=SimpleNamespace(float32='float32'))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<exact6ef7 make_plan>', 'exec'), scope)
    entries, _, summary = scope['make_plan'](groups, 4)
    digest = hashlib.sha256(json.dumps(dict(entries=entries, summary=summary), sort_keys=True).encode()).hexdigest()
    if digest != checked['plan_sha256']:
        raise RuntimeError('current owner metadata does not match authenticated prior audit')
    ranks = []
    for r in checked['ranks']:
        owner = r['rank']
        owned = [e for e in entries if e['owner'] == owner]
        rows = []
        operations = 0
        for group in r['muon_groups']:
            m, n = group['shape']
            small, large = min(m, n), max(m, n)
            # Five NS iterations, three matrix multiplies per iteration.
            # Algebraic multiply+add count, NOT observed native executed FLOPs.
            per_matrix = 5 * (4 * large * small ** 2 + 2 * small ** 3)
            count = group['executed_batch']
            rows.append({**group, 'ns_matmul_algebra_ops': count * per_matrix})
            operations += count * per_matrix
        n = sum(e['elements'] for e in owned)
        pad = sum(g['executed_batch'] - g['owned_count'] for g in r['muon_groups'])
        estimated_dummy_cost = pad * (8.98251 / 6)
        ranks.append(dict(rank=owner, owned_parameter_tensors=len(owned),
                          owned_elements=n, muon_groups=rows,
                          planner_cost_ms=summary['estimated_ms_by_owner'][owner],
                          planner_plus_linear_dummy_proxy_ms=summary['estimated_ms_by_owner'][owner] + estimated_dummy_cost,
                          own_writeback_payload_bytes=4 * n,
                          own_writeback_read_plus_write_bytes=8 * n,
                          ns_matmul_algebra_ops=operations))
    total = checked['elements']
    costs = summary['estimated_ms_by_owner']
    return dict(scope='CPU-free metadata audit of existing source, not a production candidate or timing',
                source_sha256=checked['source_sha256'], plan_sha256=digest,
                parameters=checked['parameters'], elements=total,
                real_muon_matrices=39, padded_executed_muon_rows=41,
                ranks=ranks,
                planner_cost_spread_ms=max(costs)-min(costs),
                planner_cost_spread_fraction_of_mean=(max(costs)-min(costs))/(sum(costs)/4),
                live_parameter_payload_bytes_per_rank=4*total,
                received_bytes_per_rank=summary['total_padded_bytes'],
                gather_padding_bytes_per_rank=summary['total_padded_bytes']-4*total,
                gather_padding_fraction_of_live=summary['total_padded_bytes']/(4*total)-1,
                candidate_own_copy_payload_saved_all_ranks=4*total,
                candidate_copy_read_plus_write_saved_all_ranks=8*total,
                candidate_fraction_of_aggregate_final_copy_bytes=0.25,
                current_native_phase_ms=None, predicted_whole_step_speedup=None)


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
