"""Strict stdlib result checks, not a launcher or independent review."""
import json
import math
from pathlib import Path
import sys


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def validate(root):
    identity = None
    for name in ('reference', 'candidate'):
        request = json.loads((root / name / 'request.json').read_text())
        report = json.loads((root / name / 'g2.json').read_text())
        terminal = json.loads((root / name / 'terminal.json').read_text())
        require(terminal['exit_code'] == report['torchrun_rc'] == 0 and report['timed_out'] is False, 'EXIT_TIMEOUT')
        require(terminal['verdict'] == report['verdict'] == 'PASS', 'VERDICT')
        require(report['train_py_sha256'] == request['source_sha256'] and report['argv'] == request['argv'], 'SOURCE_ARGV')
        require(report['prepare']['sha256'] == request['prepare_sha256'] and report['prepare']['stub'] is False, 'PREPARE')
        require(report['nproc'] == 4 and report['steps'] == 3 and report['micro_batch'] == [2,256], 'SHAPE')
        require(report['plan_sha256'] == 'df750f7f11b4b8dabc233cb476e486d0b6b997da2a1b32fe5af13142a394668a', 'FULL_PLAN')
        ranks=sorted(report['ranks'], key=lambda r:r['rank'])
        require([r['rank'] for r in ranks] == [0,1,2,3], 'RANKS')
        for rank in ranks:
            require(rank['status'] == 'PASS' and rank['n_params'] == rank['n_changed'] == 65, 'ALL65_UPDATED')
            require(not rank['cross_rank_mismatches'] and not rank.get('unchanged_params'), 'RANK_CONSISTENCY')
            require(sum(rank['plan_summary']['elements_by_owner']) == 113256724, 'FULL_PARAMETERS')
            require(len(rank['losses']) == 3 and all(type(x) in (float,int) and math.isfinite(x) for x in rank['losses']), 'FINITE_LOSSES')
        require(len(ranks[0]['final_digests']) == 65, 'ALL65_DIGESTS')
        ck=ranks[0]['checkpoint_check']
        require(ck['ok'] and ck['deterministic'] and ck['path_exists'] and ck['logits_shape'] == [1,256,8192], 'RELOAD')
        current=([r['losses'] for r in ranks], ranks[0]['final_digests'], report['plan_sha256'])
        if identity is None:
            identity=current
        require(identity == current, 'LOSSES_PARAMS_PLAN_IDENTITY')
        saved=json.loads((root/name/'checkpoint-identity.json').read_text())
        require(saved['bytes'] > 0 and len(saved['sha256']) == 64, 'CHECKPOINT_RECEIPT')
    return {'status':'FULL_PARAMETER_CPU_RECEIPTS_MATCH','native_correctness':False,'all_parameter_digests':65,'all_rank_losses':12,'checkpoint_file_bytes_need_not_match':'Research and clean payload schemas differ; actual model tensors/logits are checked separately.'}


if __name__ == '__main__':
    print(json.dumps(validate(Path(sys.argv[1])), indent=2, allow_nan=False))
