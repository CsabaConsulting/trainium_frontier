"""Fail-closed implementation self-check, not independent verification."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
NAMES = ('base', 'default', 'off', 'on779', 'on790', 'on1000')


def require(ok, text):
    if not ok:
        raise RuntimeError(text)


def observables(rank):
    cache = rank['cache_observations']
    return {'losses': rank['losses'], 'final_digests': rank.get('final_digests'),
            'steps': cache['steps'], 'forwards': cache['forwards'],
            'checkpoint': cache.get('checkpoint'),
            'checkpoint_file_sha256': cache.get('checkpoint_file_sha256'),
            'optimizer_groups': rank['schedule_observations']['optimizer_groups']}


def main():
    receipts = {name: json.loads((HERE / f'tiny-{name}.json').read_text()) for name in NAMES}
    for name, receipt in receipts.items():
        require(receipt['verdict'] == 'PASS' and receipt['torchrun_rc'] == 0, name + ':G2 failed')
        require(receipt['nproc'] == 4 and receipt['micro_batch'] == [2, 128], name + ':shape')
        require(receipt['steps'] == (3 if name in NAMES[:3] else 6), name + ':step count')
        require(receipt['plan_sha256'] == receipts['base']['plan_sha256'], name + ':ownership')
        require(receipt['schedule_observer_sha256'] == hashlib.sha256((HERE / 'g2_schedule_observe.py').read_bytes()).hexdigest(), name + ':observer')
        for rank in receipt['ranks']:
            require(rank['status'] == 'PASS', name + ':rank')
            require(rank['schedule_observations']['startup_steps_excluded'] == 5, name + ':real startup count')
            cache = rank['cache_observations']
            require(len(cache['steps']) == receipt['steps'], name + ':observed steps')
            require(cache['mlp_custom_backward_calls'] == 8 * receipt['steps'], name + ':active MLP coverage')
            require(all(len(step['parameters_after_update']) == 65 for step in cache['steps']), name + ':all65 parameters')
            if rank['rank'] == 0:
                require(rank['checkpoint_check']['ok'], name + ':checkpoint reload')
    for name in ('default', 'off'):
        require([observables(r) for r in receipts[name]['ranks']] ==
                [observables(r) for r in receipts['base']['ranks']], name + ':OFF identity')
    require([observables(r) for r in receipts['on779']['ranks']] ==
            [observables(r) for r in receipts['on790']['ranks']], 'ON two below-cap histories differ')
    for name in ('on779', 'on790'):
        for rank in receipts[name]['ranks']:
            for groups in rank['schedule_observations']['optimizer_groups'][:5]:
                for group in groups:
                    require(group['lr'] == group['initial_lr'], name + ':early LR')
                    if group['kind'] == 'muon':
                        require(group['weight_decay'] == .2, name + ':early WD')
    excess_groups = receipts['on1000']['ranks'][0]['schedule_observations']['optimizer_groups']
    require(any(g['kind'] == 'muon' and g['weight_decay'] < .2 for g in excess_groups[1]), 'capped excess must affect WD before fifth boundary')
    require(any(g['kind'] == 'muon' and g['weight_decay'] < .2 for g in excess_groups[5]), 'capped excess must survive fifth boundary')
    require(receipts['base']['ranks'][0]['losses'][2] != receipts['on790']['ranks'][0]['losses'][2], 'ON should be a scientific schedule change')
    result = {'status': 'IMPLEMENTER_TINY_CPU_PASS_ONLY', 'parameter_tensors': 65,
              'parameters': sum(receipts['base']['ranks'][0]['plan_summary']['elements_by_owner']),
              'default_off_identical': True, 'all_parameter_gradient_update_optimizer_checkpoint_observations_equal': True,
              'on_below_cap_histories_identical_six_steps': True, 'on_excess_retained': True,
              'on_not_off_equivalent': True, 'native_or_fullsize_proof': False,
              'receipt_sha256': {n: hashlib.sha256((HERE / f'tiny-{n}.json').read_bytes()).hexdigest() for n in NAMES}}
    print(json.dumps(result, indent=2, sort_keys=True))
    if len(sys.argv) == 2:
        Path(sys.argv[1]).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
