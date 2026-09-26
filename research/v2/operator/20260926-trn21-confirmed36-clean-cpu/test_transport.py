"""Local stdlib/Git mechanics only. Does not import the transport or call hosts."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SCRIPT = HERE / 'stage-dispatch.py'
raw = SCRIPT.read_bytes()
tree = ast.parse(raw)
checks = []

def check(ok, label):
    if not ok:
        raise AssertionError(label)
    checks.append(label)

def literal(module, name):
    return ast.literal_eval(next(n.value for n in module.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)))

body = literal(tree, 'body')
ast.parse(body)
check(True, 'local and remote programs parse without execution')
d = json.loads((HERE / 'REMOTE_BODY_DERIVATION.json').read_bytes())
parent = subprocess.check_output(['git', 'show', d['parent_commit'] + ':' + d['parent_path']], cwd=REPO)
check(hashlib.sha256(parent).hexdigest() == d['parent_sha256'], 'proven parent exact hash')
derived = literal(ast.parse(parent), 'body')
for old, new in d['replacements']:
    check(derived.count(old) == 1, 'unique mechanical replacement ' + str(len(checks)))
    derived = derived.replace(old, new)
check(derived == body, 'entire remote body exact mechanical transform')
check('fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)' in body, 'original exclusive staging lock')
check('(66305,596823)' in body, 'original trn21 inode pinned')
check(body.index('fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)') < body.index("ps=subprocess.run("), 'fresh ps under lock')
check(body.index("ps=subprocess.run(") < body.index("ls=subprocess.run(") < body.index('root.mkdir('), 'ps and neuron-ls before stage creation')
check('holders.returncode==1' in body, 'device-holder supplementary check')
check('free>=10.5*1024**3+projected' in body and 'projected=2*1024**3' in body, 'disk floor plus exact two GiB footprint')
check('mem>=64*1024**3' in body, '64 GiB memory floor')
check("for role in ('reference','candidate')" in body, 'only two CPU arm directories')
check('NORMAL_PYTHON_REQUIRED' in raw.decode(), 'optimized Python refused')
check(raw.index(b"if sys.argv[1:] == ['--local-check']:") < raw.index(b"spec = importlib.util.spec_from_file_location('transport'"), 'local check returns before transport helper')
check(raw.index(b"assert len(sys.argv) == 4") < raw.index(b"spec = importlib.util.spec_from_file_location('transport'"), 'missing assignment rejected before helper')

cases = [
    ('local-only authentication', [sys.executable, '-B', str(SCRIPT), '--local-check'], 0, 'EXACT_LOCAL_PACKET_AUTHENTICATED_ONLY'),
    ('missing release refused', [sys.executable, '-B', str(SCRIPT)], 1, 'EXACT_RELEASE_COMMIT_PATH_AND_PREDECESSOR_FINAL_COMMIT_REQUIRED'),
    ('malformed release refused', [sys.executable, '-B', str(SCRIPT), 'bad', 'bad', 'bad'], 1, 'EXACT_COMMIT_FORMAT'),
    ('optimized invocation refused', [sys.executable, '-O', '-B', str(SCRIPT), '--local-check'], 1, 'NORMAL_PYTHON_REQUIRED'),
]
outcomes = []
for label, argv, code, text in cases:
    r = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    check(r.returncode == code and text in r.stdout + r.stderr, label)
    outcomes.append({'case': label, 'returncode': r.returncode, 'expected_marker': text})
    if code == 0:
        report = json.loads(r.stdout)
        check(report['host_calls'] is False and report['stage_authorized'] is False and report['launch_authorized'] is False, 'local result cannot authorize launch')
print(json.dumps({'status': 'PASS_LOCAL_MECHANICS_ONLY', 'checks': len(checks), 'script_sha256': hashlib.sha256(raw).hexdigest(), 'remote_body_sha256': hashlib.sha256(body.encode()).hexdigest(), 'host_calls': False, 'tensors': False, 'cases': outcomes}, indent=2))
