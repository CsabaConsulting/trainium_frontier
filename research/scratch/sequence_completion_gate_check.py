#!/usr/bin/env python3
"""Exercise sequence advancement gates against real records and corrupted copies.

Local metadata only: no SSH, experiment launch, package install or device use.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fresh_start_sequence import ROOT, read_requests, records, require_completed


def main():
    root = ROOT / 'research/run-logs'
    plan = json.loads((root / 'fs-w148-output-rank-sequence-r1.plan.json').read_text())
    native_plan = json.loads((root / 'fs-w148-native-control-sequence-r1.plan.json').read_text())
    assert len(read_requests(plan)) == 3
    assert len(read_requests(native_plan)) == 1
    label = 'fs-w147-full-fp8-streamed-phase-r1'
    request = json.loads((root / (label + '.request.json')).read_text())
    post = json.loads((root / (label + '.postflight.json')).read_text())
    raw = (root / (label + '.log')).read_text()
    rules = copy.deepcopy(plan['steps'][1]['requirements'])
    del rules[0]['fields']['fp8_mlp_preserve_output_rank']  # Absent from source147.
    rules[0]['fields']['fp8_mlp_stream_packed_weights'] = True
    require_completed(post, raw, request, rules)
    phase, = records(raw, 'PHASE_JSON')
    checks = records(raw, 'NATIVE_POST_CHECK_JSON')
    def render(p=phase, c=checks):
        # Deliberately concatenate check records: distributed stdout may do so.
        return 'PHASE_JSON ' + json.dumps(p) + '\n' + ''.join(
            'NATIVE_POST_CHECK_JSON ' + json.dumps(v) for v in c)
    require_completed(post, render(), request, rules)
    rejected = []
    def reject(name, action):
        try:
            action()
        except (ValueError, KeyError, TypeError) as error:
            rejected.append(dict(case=name, error=str(error)))
        else:
            raise AssertionError('Invalid completion was accepted: ' + name)
    for field, value in [('host_clean', False), ('command_returncode', 1),
                         ('source_commit', '0' * 40), ('remote', '/wrong/checkout'),
                         ('label', 'fs-wrong-label'), ('protocol_version', 1),
                         ('failures', ['device still held']), ('guard_error', 'lost child'),
                         ('interrupted_signals', [15])]:
        broken = copy.deepcopy(post); broken[field] = value
        reject('post:' + field, lambda b=broken: require_completed(b, raw, request, rules))
    reject('missing_rank', lambda: require_completed(post, render(c=checks[:3]), request, rules))
    broken = copy.deepcopy(checks); broken[-1]['rank'] = broken[0]['rank']
    reject('duplicate_rank', lambda: require_completed(post, render(c=broken), request, rules))
    broken = copy.deepcopy(checks); broken[-1]['passed'] = False
    reject('failed_numerical_check', lambda: require_completed(post, render(c=broken), request, rules))
    for name in ['nonfinite_loss', 'nonfinite_timing', 'too_few_steps', 'stale_cache']:
        broken = copy.deepcopy(phase)
        if name == 'nonfinite_loss': broken['last_loss'] = float('nan')
        if name == 'nonfinite_timing': broken['steps'][0]['step_total_ms'] = float('inf')
        if name == 'too_few_steps': broken['steps'] = broken['steps'][:-1]
        if name == 'stale_cache': broken['fp8_mlp_cache_refresh_counts'][0] -= 1
        reject(name, lambda b=broken: require_completed(post, render(p=b), request, rules))
    for name in ['request_digest', 'cross_host', 'duplicate_request', 'no_completion_gates', 'unbounded']:
        broken = copy.deepcopy(plan)
        if name == 'request_digest': broken['steps'][0]['sha256'] = '0' * 64
        if name == 'cross_host': broken['steps'][0] = native_plan['steps'][0]
        if name == 'duplicate_request': broken['steps'][1] = broken['steps'][0]
        if name == 'no_completion_gates': broken['steps'][0]['requirements'] = []
        if name == 'unbounded': broken['max_seconds'] = 12001
        reject(name, lambda b=broken: read_requests(b))
    result = dict(passed=True, actual_completed_phase_accepted=True,
                  concatenated_distributed_output_accepted=True,
                  prepared_plans_accepted=2, rejected_cases=rejected,
                  coordinator_sha256=hashlib.sha256((ROOT / 'research/fresh_start_sequence.py').read_bytes()).hexdigest(),
                  device_use=False, launched_jobs=False)
    path = root / 'fs-sequence-completion-gate-check-20260919-r1.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, rejected_cases=len(rejected), receipt=str(path))))


if __name__ == '__main__':
    main()
