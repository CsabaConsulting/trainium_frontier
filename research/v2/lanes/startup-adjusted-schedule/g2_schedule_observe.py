"""Tiny CPU-only layer on the unchanged G2/cache observer; no native launcher."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

SUPPORT = Path(os.environ['SCHEDULE_TEST_SUPPORT_ROOT']).resolve()
PINS = {
    'research/v2/lanes/initial-cache-refresh/g2_observe.py': 'a1b9108c3aa3af22cd819e6c0553e6c5c2bb7f3887bc0d3c1b8b4e6d856db03d',
    'workflow/tools/preflight_dryrun.py': '6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8',
    'workflow/tools/stub_prepare/prepare.py': 'b2ae4d6f7de54aa1b0f5a9cb2c2629c38c86c30178e6516a7b48b8ac004ca1a2',
    'prepare.py': '5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb',
}
for relative, digest in PINS.items():
    if hashlib.sha256((SUPPORT / relative).read_bytes()).hexdigest() != digest:
        raise RuntimeError('test-support pin mismatch: ' + relative)
spec = importlib.util.spec_from_file_location('cache_observer', SUPPORT / next(iter(PINS)))
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)
g2 = observer.g2
g2.__file__ = __file__
original_observe = observer.observe


def observe(mod):
    # Standard G2's data-free stub uses2. Restore the real pinned prepare's5
    # only in this CPU test module; neither source nor prepare file is edited.
    mod.TRAIN_STARTUP_STEPS_EXCLUDED = 5
    original_observe(mod)
    records = []
    delay = float(os.environ.get('SCHEDULE_TEST_FIRST_STEP_DELAY', '790'))
    original_step = mod.OwnedOptimizer.step

    def step(self, *args, **kwargs):
        keys = ('kind', 'lr', 'initial_lr', 'momentum', 'weight_decay', 'beta2', 'betas')
        records.append([{k: group[k] for k in keys if k in group} for group in self.optimizer.param_groups])
        result = original_step(self, *args, **kwargs)
        if len(records) == 1:
            # Simulated compilation duration only. No sleep/work outside accounting.
            # elapsed for update0 was already sampled; subsequent steps observe the jump.
            mod.time._t += delay
        return result
    mod.OwnedOptimizer.step = step
    original_cleanup = mod.cleanup_runtime

    def cleanup():
        path = Path(os.environ['G2_WORKDIR']) / ('schedule-observe-rank' + os.environ['RANK'] + '.json')
        path.write_text(json.dumps({'first_step_fake_delay_s': delay, 'startup_steps_excluded': mod.TRAIN_STARTUP_STEPS_EXCLUDED,
                                    'optimizer_groups': records}, sort_keys=True))
        return original_cleanup()
    mod.cleanup_runtime = cleanup


observer.observe = observe
original_aggregate = g2.aggregate


def aggregate(ns, train_py, prepare_dir, train_argv, ranks, *args):
    result = original_aggregate(ns, train_py, prepare_dir, train_argv, ranks, *args)
    for rank in ranks:
        if rank.get('appended_argv'):
            p = Path(rank['appended_argv'][-1]).parent / f"schedule-observe-rank{rank['rank']}.json"
            rank['schedule_observations'] = json.loads(p.read_text()) if p.exists() else None
    result['schedule_observer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['support_pins'] = PINS
    result['shims'].append('test-only compile delay after first optimizer step; actual model/loop remains CPU synthetic')
    result['shims'].append('test-only imported TRAIN_STARTUP_STEPS_EXCLUDED restored from stub2 to real pinned prepare5')
    return result


g2.aggregate = aggregate
if __name__ == '__main__':
    sys.exit(g2.main())
