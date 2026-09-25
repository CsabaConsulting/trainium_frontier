"""Fixed, sequential local CPU self-check recipe; no remote/native operation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SUPPORT = Path('/home/poojith-devan/Desktop/Trainium')
BASE = Path('/home/poojith-devan/Desktop/Trainium-architecture-initial-cache-refresh-20260925/workflow/proposals/research-v2/train.py')
CANDIDATE = ROOT / 'workflow/proposals/research-v2/train.py'
PINS = {BASE: '3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c',
        CANDIDATE: '36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb'}
ARGV = ['--n-embd', '128', '--depth', '8', '--unet-skip', '--partial-attn-8',
        '--attention-layers', '0,2,4,6,7', '--initial-cache-refresh', '--no-ngram-ve',
        '--no-qk-shift', '--compile-sdpa-direct', '--nki-local-conv', '--bf16-norm-output',
        '--pack-factor', '32', '--clip-after-reduce', '--seq-len', '1024',
        '--no-eval-public', '--matrix-lr', '0.015', '--seed', '42', '--max-train-seconds', '1800']
ARMS = [('base', BASE, 3, 790, []), ('default', CANDIDATE, 3, 790, []),
        ('off', CANDIDATE, 3, 790, ['--no-startup-adjusted-schedule']),
        ('on779', CANDIDATE, 6, 779, ['--startup-adjusted-schedule']),
        ('on790', CANDIDATE, 6, 790, ['--startup-adjusted-schedule']),
        ('on1000', CANDIDATE, 6, 1000, ['--startup-adjusted-schedule'])]


if __name__ == '__main__':
    for path, pin in PINS.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != pin:
            raise RuntimeError('source mismatch: ' + str(path))
    available_kib = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
    if available_kib < 6 * 1024**2:
        raise RuntimeError('tiny CPU recipe requires6GiB available; no full-size fallback')
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', SCHEDULE_TEST_SUPPORT_ROOT=str(SUPPORT),
               PYTHONPATH=str(SUPPORT / 'research/v2/lanes/architecture-recursive-stack/test_support'))
    commands = []
    for name, source, steps, delay, delta in ARMS:
        command = [sys.executable, '-B', str(HERE / 'g2_schedule_observe.py'), '--train-py', str(source),
                   '--stub-prepare', '--steps', str(steps), '--nproc', '4', '--micro-rows', '2',
                   '--micro-tokens', '128', '--timeout', '150', '--out', str(HERE / f'tiny-{name}.json'),
                   '--'] + ARGV + delta
        task_env = dict(env, SCHEDULE_TEST_FIRST_STEP_DELAY=str(delay))
        commands.append({'name': name, 'argv': command, 'first_step_fake_delay_s': delay,
                         'env': {k: task_env[k] for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                  'OPENBLAS_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE', 'PYTHONPATH',
                                  'SCHEDULE_TEST_SUPPORT_ROOT', 'SCHEDULE_TEST_FIRST_STEP_DELAY')}})
        (HERE / 'tiny-commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        prior = HERE / f'tiny-{name}.json'
        if name == 'base' and prior.exists():
            receipt = json.loads(prior.read_text())
            if receipt['verdict'] != 'PASS' or receipt['train_py_sha256'] != PINS[source]:
                raise RuntimeError('pre-existing first-arm receipt invalid; preserve for inspection')
            print('Reusing just-completed pinned baseline receipt', flush=True)
            continue
        if prior.exists():
            raise RuntimeError('will not overwrite prior receipt: ' + str(prior))
        print('Starting bounded arm ' + name, flush=True)
        subprocess.run(command, env=task_env, cwd=ROOT, check=True, timeout=180)
