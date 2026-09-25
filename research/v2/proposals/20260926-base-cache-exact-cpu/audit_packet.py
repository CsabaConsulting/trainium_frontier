"""Packet-local stdlib checks only. Never imports train.py or launches jobs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path)
    args = parser.parse_args()
    plan = load(HERE / 'CPU_PLAN.disabled.json')
    require(plan['launch_authorized'] is False and plan['native_launch_authorized'] is False, 'disabled status')
    require(plan['local_tensor_execution_prohibited'] is True, 'laptop prohibition')
    require(plan['full_model']['parameters'] == 117453580 and plan['full_model']['parameter_tensors'] == 62, 'full shape')
    require(plan['ordering'] == ['reference-6ef7', 'candidate-default', 'candidate-off', 'candidate-on'], 'arm ordering')
    for key, source in plan['sources'].items():
        if 'commit' in source:
            raw = subprocess.check_output(['git', 'show', source['commit'] + ':' + source['git_path']], cwd=ROOT)
        else:
            raw = (ROOT / source.get('git_path', 'prepare.py')).read_bytes()
        require(sha(raw) == source['sha256'], key + ': source/tool hash')
    drivers = []
    for i, probe in enumerate(plan['probes']):
        request_path, driver_path = HERE / probe['request_path'], HERE / probe['driver_path']
        require(sha(request_path.read_bytes()) == probe['request_sha256'], probe['name'] + ': request hash')
        require(sha(driver_path.read_bytes()) == probe['driver_sha256'], probe['name'] + ': driver hash')
        request = load(request_path)
        require(request['command'] == probe['command'], 'command binding')
        canon = json.dumps(request['command'], separators=(',', ':'), ensure_ascii=True).encode()
        require(sha(canon) == probe['command_sha256'], 'command hash')
        extra = [] if i < 2 else (['--no-initial-cache-refresh'] if i == 2 else ['--initial-cache-refresh'])
        require(request['argv'] == plan['common_train_argv'] + extra, 'one flag only')
        command = request['command']
        require(command[command.index('--') + 1:] == request['argv'], 'train argv exact')
        for flag, expected in [('--steps', '3'), ('--nproc', '4'), ('--micro-rows', '2'), ('--micro-tokens', '256'), ('--timeout', '3600')]:
            require(command[command.index(flag) + 1] == expected, flag)
        require('--stub-prepare' not in command and '--real-data' not in command, 'real prepare/synthetic data')
        key = 'reference' if i == 0 else 'candidate'
        require(request['source_sha256'] == plan['sources'][key]['sha256'], 'arm source')
        require(command[command.index('--train-py') + 1] == plan['sources'][key]['stage_path'], 'source path')
        tree = ast.parse(driver_path.read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        guard = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'require_idle_under_inherited_lock']
        popen = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'Popen']
        require(len(guard) == len(popen) == 1 and guard[0].lineno < popen[0].lineno, 'guard precedes child')
        require(any(k.arg == 'pass_fds' and ast.unparse(k.value) == '(lock_fd,)' for k in popen[0].keywords), 'inherited FD')
        require('if int(p.parent.name) == os.getpid():' in driver_path.read_text(), 'authenticated self exclusion')
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'ROOT':
                node.value = ast.Constant('NORMALIZED_ARM_ROOT')
        drivers.append(ast.dump(tree))
    require(len(set(drivers)) == 1, 'only ROOT differs across drivers')
    cold = load(HERE / 'COLD_PROPOSAL.disabled.json')
    require(cold['launch_authorized'] is False and cold['full_run_automatic'] is False, 'cold separately disabled')
    require(cold['source_sha256'] == plan['sources']['candidate']['sha256'], 'cold source')
    require(cold['argv'][cold['argv'].index('--num-steps') + 1] == '5' and cold['ct_target_seconds'] == 600, 'cold scope')
    if args.results:
        reference, checkpoints = None, []
        for i, probe in enumerate(plan['probes']):
            folder = args.results / probe['name']
            report, terminal = load(folder / 'g2.json'), load(folder / 'terminal.json')
            require(terminal['exit_code'] == report['torchrun_rc'] == 0 and report['timed_out'] is False, 'explicit exits')
            require(report['verdict'] == 'PASS' and terminal['verdict'] == 'PASS', 'G2 PASS')
            request = load(HERE / probe['request_path'])
            require(report['train_py_sha256'] == request['source_sha256'] and report['argv'] == request['argv'], 'actual binding')
            require(report['prepare']['sha256'] == plan['sources']['prepare']['sha256'] and not report['prepare']['stub'], 'actual prepare')
            require(report['nproc'] == 4 and report['steps'] == 3 and report['micro_batch'] == [2, 256], 'actual test shape')
            require(report['plan_sha256'] == plan['expected_plan_sha256'], 'plan')
            ranks = sorted(report['ranks'], key=lambda r: r['rank'])
            require([r['rank'] for r in ranks] == [0, 1, 2, 3], 'all ranks')
            for rank in ranks:
                require(rank['status'] == 'PASS' and rank['n_params'] == rank['n_changed'] == 62, 'all params updated')
                require(not rank['cross_rank_mismatches'] and len(rank['losses']) == 3, 'rank consistency')
                require(sum(rank['plan_summary']['elements_by_owner']) == 117453580, 'full parameter elements')
                require(all(isinstance(x, (float, int)) and abs(x) < float('inf') for x in rank['losses']), 'finite losses')
            checkpoint = ranks[0]['checkpoint_check']
            require(checkpoint['ok'] and checkpoint['deterministic'] and checkpoint['path_exists'], 'checkpoint reload')
            require(checkpoint['logits_shape'] == [1, 256, 8192], 'checkpoint output shape')
            values = ([r['losses'] for r in ranks], ranks[0]['final_digests'], report['plan_sha256'], checkpoint['logits_shape'])
            require(len(values[1]) == 62, 'all digests')
            if reference is None:
                reference = values
            require(values == reference, 'losses/final updates/plan/checkpoint shape identity')
            ck = load(folder / 'checkpoint-identity.json')
            require(ck['bytes'] > 0 and len(ck['sha256']) == 64, 'checkpoint file identity receipt')
            checkpoints.append((ck['sha256'], ck['bytes']))
        require(len(set(checkpoints)) == 1, 'all checkpoint bytes identical')
    print(json.dumps({'packet_static_checks': 'PASS', 'native_launch_authorized': False,
                      'numeric_results_checked': bool(args.results), 'tensor_or_host_execution': False}, sort_keys=True))


if __name__ == '__main__':
    main()
