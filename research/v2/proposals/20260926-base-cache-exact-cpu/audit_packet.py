"""Packet-local stdlib checks only. Never imports train.py or launches jobs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads(path.read_text())


def check_legacy_scan(tree, outer_command):
    """Stdlib regression for the independent review's authenticated-parent fixture."""
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'checks')
    start = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'active')
    scan = compile(ast.Module(body=[function.body[0], *function.body[start:start + 3]],
                              type_ignores=[]), 'packet-process-scan', 'exec')
    class Entry:
        def __init__(self, pid, argv):
            self.parent = SimpleNamespace(name=str(pid))
            self.argv = argv
        def read_bytes(self):
            return b'\0'.join(s.encode() for s in self.argv) + b'\0'
    entries = [Entry(100, ['/usr/bin/python3', outer_command[-1]]), Entry(99, outer_command)]
    class Proc:
        def glob(self, pattern):
            return entries
    def path(value):
        return Proc() if value == '/proc' else Path(value)
    for extra, parent, expected in [
        (None, 99, None),
        (Entry(200, ['/venv/bin/torchrun', 'train.py']), 99, 'OTHER_WORKLOAD:[200]'),
        (Entry(201, ['/usr/bin/python3', '/unrelated/host_driver.py']), 99, 'OTHER_WORKLOAD:[201]'),
        (None, 98, 'AUTHENTICATED_PARENT_CHANGED'),
    ]:
        entries[:] = entries[:2] + ([] if extra is None else [extra])
        context = {'Path': path, 'os': SimpleNamespace(getpid=lambda: 100, getppid=lambda: parent),
                   'authenticated_outer_pid': 99, 'require': require}
        error = None
        try:
            exec(scan, context)
        except RuntimeError as exc:
            error = str(exc)
        require(error == expected, 'legacy scan regression: ' + repr((error, expected)))


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
        check_legacy_scan(tree, probe['expected_outer_command'])
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        guard = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'require_idle_under_inherited_lock']
        popen = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'Popen']
        require(len(guard) == len(popen) == 1 and guard[0].lineno < popen[0].lineno, 'guard precedes child')
        require(any(k.arg == 'pass_fds' and ast.unparse(k.value) == '(lock_fd,)' for k in popen[0].keywords), 'inherited FD')
        require('if int(p.parent.name) in (os.getpid(), authenticated_outer_pid):' in driver_path.read_text(), 'authenticated self/parent exclusion')
        require("require(os.getppid() == authenticated_outer_pid, 'AUTHENTICATED_PARENT_CHANGED')" in driver_path.read_text(), 'parent identity retained')
        parent_capture = [n for n in tree.body[-1].body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'authenticated_outer_pid' for t in n.targets)]
        require(len(parent_capture) == 1 and parent_capture[0].lineno < guard[0].lineno, 'parent captured before authentication')
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
