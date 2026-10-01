#!/usr/bin/env python3
"""Fresh bounded training, then matched native evaluation on an unchanged scorer."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['middle6', 'middle6rms'], required=True)
    ap.add_argument('--label', required=True)
    args = ap.parse_args()
    source = ROOT/'candidates'/args.variant/'train.py'
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = json.loads((ROOT/'deadline/candidate-manifest.json').read_text())
    assert sha == manifest['candidates'][args.variant]['sha256']
    out = ROOT/'out'/args.label
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for key, sub in [('NEURON_COMPILE_CACHE_DIR', 'compiler'),
                     ('NEURON_COMPILE_CACHE_URL', 'compiler'),
                     ('TORCH_NEURONX_HLO_CACHE_DIR', 'hlo'),
                     ('TORCH_NEURONX_NEFF_CACHE_DIR', 'neff'),
                     ('TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR', 'local'),
                     ('TORCH_NEURONX_DEBUG_DIR', 'debug'),
                     ('TORCHINDUCTOR_CACHE_DIR', 'inductor')]:
        path = out/'caches'/sub
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env['NEURON_LOGICAL_NC_CONFIG'] = '2'
    env['NEURON_CC_FLAGS'] = '--optlevel=3 --auto-cast matmult --auto-cast-type bf16'
    started = time.monotonic()
    print('DEADLINE_START '+json.dumps(dict(variant=args.variant, source_sha256=sha,
        prepare_sha256=hashlib.sha256((ROOT/'prepare.py').read_bytes()).hexdigest(),
        time_budget_seconds=1800, fresh_cache=str(out/'caches'))), flush=True)
    subprocess.run([PYTHON, str(ROOT/'deadline/check_more.py')], cwd=ROOT,
                   env=env, check=True, timeout=90)
    command = [TORCHRUN, '--standalone', '--nproc_per_node=4', str(source),
               '--tbl_freeze_tokens', '0', '--out-dir', str(out/'train')]
    with (out/'train.log').open('x') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=3100)
    checkpoint = out/'train/final.pt'
    assert checkpoint.is_file()
    assert 'over_budget:       False' in (out/'train.log').read_text()
    record = dict(elapsed_seconds=time.monotonic()-started, checkpoint=str(checkpoint),
        checkpoint_bytes=checkpoint.stat().st_size,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    print('DEADLINE_TRAIN_COMPLETE '+json.dumps(record), flush=True)
    subprocess.run([PYTHON, str(ROOT/'deadline/evaluate_match.py'),
        '--variant', args.variant, '--checkpoint', str(checkpoint), '--label', args.label],
        cwd=ROOT, env=env, check=True, timeout=1800)


if __name__ == '__main__':
    main()
