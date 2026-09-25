"""TEST ONLY: existing synthetic G2, with optional P trace + CPU profiler at step1.

Never used by the native wrapper. Existing G2 applies its documented CPU shims;
this bridge wraps only its imported main entry to exercise the unchanged P hook
against the actual source function. Native capture stays fixed at step20.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location('p_wrapper', HERE / 'profile_actual_loop.py')
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)
spec = importlib.util.spec_from_file_location('g2', ROOT / 'workflow/tools/preflight_dryrun.py')
g2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g2)
g2.__file__ = __file__


class CPUEngine:
    def prepare(self, frame):
        import torch
        torch.set_num_threads(1)
        self.torch = torch
        self.prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True, with_stack=True, with_flops=True)

    def start(self, frame):
        self.start_step = frame.f_locals['step']
        self.prof.__enter__()

    def region(self, label):
        return self.torch.profiler.record_function('P/' + label)

    def stop(self, frame):
        self.prof.__exit__(None, None, None)
        events = [str(event.name) for event in self.prof.events() if str(event.name).startswith('P/')]
        report = dict(scope='TEST ONLY CPU profiler and reduced synthetic G2, not native',
            start_step=self.start_step, stop_step=frame.f_locals['step'], events=events)
        (Path(os.environ['G2_WORKDIR']) / ('p-cpu-rank' + os.environ['RANK'] + '.json')).write_text(json.dumps(report))

    def abort(self):
        pass


original_spec = g2.importlib.util.spec_from_file_location


def instrumented_spec(name, *args, **kwargs):
    result = original_spec(name, *args, **kwargs)
    if name == 'train_under_test' and os.environ.get('P_CPU_CAPTURE') == '1':
        original_exec = result.loader.exec_module
        def load(mod):
            original_exec(mod)
            source = Path(mod.__file__).resolve()
            original_main = mod.main
            def main():
                trace = wrapper.LoopTrace(source, wrapper.landmarks(source.read_text()), CPUEngine(), target=1)
                wrapper.require(sys.gettrace() is None, 'EXISTING_TEST_TRACE')
                try:
                    sys.settrace(trace)
                    result = original_main()
                    wrapper.require(trace.finished, 'CPU_PROFILE_INCOMPLETE')
                    return result
                finally:
                    sys.settrace(None)
            mod.main = main
        result.loader.exec_module = load
    return result


g2.importlib.util.spec_from_file_location = instrumented_spec
original_aggregate = g2.aggregate


def aggregate(ns, train_py, prepare_dir, train_argv, ranks, *args):
    result = original_aggregate(ns, train_py, prepare_dir, train_argv, ranks, *args)
    for rank in ranks:
        if rank.get('appended_argv'):
            path = Path(rank['appended_argv'][-1]).parent / ('p-cpu-rank' + str(rank['rank']) + '.json')
            rank['p_cpu_profile'] = json.loads(path.read_text()) if path.exists() else None
    result['p_cpu_capture'] = os.environ.get('P_CPU_CAPTURE') == '1'
    return result


g2.aggregate = aggregate
if __name__ == '__main__':
    sys.exit(g2.main())
