#!/usr/bin/env python3
"""Bounded fresh training and official public evaluation; no evaluator edits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/ubuntu/workspace/native_venv/bin/python'
TORCHRUN = '/home/ubuntu/workspace/native_venv/bin/torchrun'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=['mlpnki', 'mlp6nki'])
    ap.add_argument('--label', required=True)
    args = ap.parse_args()
    source = ROOT / 'candidates' / args.variant / 'train.py'
    manifest = json.loads((ROOT / 'deadline/candidate-manifest.json').read_text())
    assert digest(source) == manifest['candidates'][args.variant]['sha256']
    out = ROOT / 'out' / args.label
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['NEURON_COMPETITION_R1_CACHE_DIR'] = '/home/ubuntu/.cache/neuron_competition_r1'
    env['NEURON_LOGICAL_NC_CONFIG'] = '2'
    # Empty per-job compiler caches; preserve all prior sources/cache evidence.
    cache = out / 'caches'
    for key, sub in [('NEURON_COMPILE_CACHE_DIR', 'compiler'),
                     ('NEURON_COMPILE_CACHE_URL', 'compiler'),
                     ('TORCH_NEURONX_HLO_CACHE_DIR', 'hlo'),
                     ('TORCH_NEURONX_NEFF_CACHE_DIR', 'neff'),
                     ('TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR', 'local'),
                     ('TORCH_NEURONX_DEBUG_DIR', 'debug'),
                     ('TORCHINDUCTOR_CACHE_DIR', 'inductor')]:
        path = cache / sub
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env['NEURON_CC_FLAGS'] = '--optlevel=3 --auto-cast matmult --auto-cast-type bf16'
    started = time.monotonic()
    print('DEADLINE_START ' + json.dumps({'variant': args.variant, 'source_sha256': digest(source),
          'prepare_sha256': digest(ROOT / 'prepare.py'), 'fresh_cache': str(cache),
          'time_budget_seconds': 1800}), flush=True)
    gate_env = env.copy()
    for key in ('NEURON_COMPILE_CACHE_DIR', 'NEURON_COMPILE_CACHE_URL',
                'TORCH_NEURONX_HLO_CACHE_DIR', 'TORCH_NEURONX_NEFF_CACHE_DIR',
                'TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR', 'TORCH_NEURONX_DEBUG_DIR',
                'TORCHINDUCTOR_CACHE_DIR'):
        gate = out / 'gate-caches' / key
        gate.mkdir(parents=True, exist_ok=True)
        gate_env[key] = str(gate)
    gate_env['NKI_VENV_PATH'] = '/home/ubuntu/workspace/native_venv'
    subprocess.run([PYTHON, str(ROOT / 'deadline/check_pairs.py'), '--native'], cwd=ROOT,
                   env=gate_env, check=True, timeout=300)
    # Native gate child has exited. Verify release before starting a separate
    # fresh training process; its caches above are still empty.
    import sys
    sys.path.insert(0, str(ROOT / 'research'))
    from fresh_start_host import snapshot, health_failures
    released = snapshot(ROOT)
    failures = health_failures(released, (os.getpid(), os.getppid()))
    (out / 'native-gate-release.json').write_text(json.dumps(dict(health=released, failures=failures), indent=2)+'\n')
    assert not failures, failures
    train_command = [TORCHRUN, '--standalone', '--nproc_per_node=4', str(source),
                     '--tbl_freeze_tokens', '0', '--out-dir', str(out / 'train')]
    with (out / 'train.log').open('x') as log:
        subprocess.run(train_command, cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=3100)
    checkpoint = out / 'train/final.pt'
    assert checkpoint.is_file()
    training_text = (out / 'train.log').read_text()
    assert 'over_budget:       False' in training_text
    print('DEADLINE_TRAIN_COMPLETE ' + json.dumps({'elapsed_seconds': time.monotonic() - started,
          'checkpoint': str(checkpoint), 'checkpoint_bytes': checkpoint.stat().st_size,
          'checkpoint_sha256': digest(checkpoint)}), flush=True)
    eval_env = env.copy()
    eval_env['NEURON_LOGICAL_NC_CONFIG'] = '2'
    eval_env.pop('NEURON_CC_FLAGS', None)
    with (out / 'eval.log').open('x') as log:
        subprocess.run([PYTHON, str(ROOT / 'prepare.py'), 'eval-public', '--train-py', str(source),
                        '--checkpoint', str(checkpoint), '--eval-tokens', '20971520'], cwd=ROOT, env=eval_env, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=1300)
    print('DEADLINE_EVAL_LOG_BEGIN', flush=True)
    print((out / 'eval.log').read_text(), flush=True)
    print('DEADLINE_FINISH ' + json.dumps({'variant': args.variant, 'elapsed_seconds': time.monotonic()-started,
          'source_sha256': digest(source), 'checkpoint_sha256': digest(checkpoint)}), flush=True)


if __name__ == '__main__':
    main()
