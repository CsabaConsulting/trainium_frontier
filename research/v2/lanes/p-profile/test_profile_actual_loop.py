"""Bounded stdlib tests. No torch import, real data access, host or native calls."""
import ast
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('profile_actual_loop', HERE / 'profile_actual_loop.py')
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)

SOURCE_REFS = {
    'a4aa4351789115a2d0aa947d9ca00ff689f9bd0b': '6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3',
    '1e21c3d797eed764f108c41546b270065ad5650c': '93ae1f92e6222fcb31215ad1bc7983c6d013fff53b7b3337a911ee2e2b11cfd5',
    'b4d83b5db66971f3c348837f04f53cba65aa6d33': '3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c',
}


class Engine:
    def __init__(self):
        self.events = []
        self.stack = []

    def prepare(self, frame):
        self.events.append(('prepare', frame.f_locals['step']))

    def start(self, frame):
        self.events.append(('start', frame.f_locals['step']))

    def stop(self, frame):
        if self.stack:
            raise RuntimeError('unbalanced annotation')
        self.events.append(('stop', frame.f_locals['step']))

    def abort(self):
        self.events.append(('abort', None))

    @contextlib.contextmanager
    def region(self, label):
        self.stack.append(label)
        self.events.append(('enter', label))
        try:
            yield
        finally:
            if self.stack.pop() != label:
                raise RuntimeError('annotation ordering')
            self.events.append(('exit', label))


class Tests(unittest.TestCase):
    def test_all_three_exact_sources_have_the_expected_statements(self):
        for ref, expected_sha in SOURCE_REFS.items():
            raw = subprocess.check_output(['git', 'show', ref + ':workflow/proposals/research-v2/train.py'], cwd=HERE)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected_sha)
            self.assertIn(expected_sha, wrapper.SOURCES)
            plan = wrapper.landmarks(raw)
            labels = list(plan['statements'].values())
            self.assertEqual(labels.count('weight_cache_refresh'), 2 if expected_sha.startswith('3e79') else 1)
            self.assertEqual(labels.count('existing_synchronize'), 2)
            self.assertEqual(labels.count('forward'), 1)
            self.assertEqual(labels.count('backward'), 1)

    def test_real_runpy_loop_single_window_identical_arithmetic_and_restoration(self):
        source = HERE / 'test_fixture.py'
        args_before, path_before = sys.argv, sys.path[:]
        off = wrapper.run_exact(source, ['22'])['RESULT']
        engine = Engine()
        trace = wrapper.LoopTrace(source, wrapper.landmarks(source.read_text()), engine)
        on = wrapper.run_exact(source, ['22'], trace)['RESULT']
        self.assertEqual(off, on)
        self.assertTrue(trace.finished)
        self.assertEqual([x for x in engine.events if x[0] in ('prepare', 'start', 'stop')],
                         [('prepare', 0), ('start', 20), ('stop', 21)])
        labels = [label for event, label in engine.events if event == 'enter']
        for label in ('forward', 'backward', 'optimizer', 'optimizer_muon_group_inclusive',
                      'optimizer_muon_eager_math_if_executed', 'data_fetch', 'weight_cache_refresh'):
            self.assertEqual(labels.count(label), 1)
        self.assertEqual(labels.count('existing_synchronize'), 2)
        self.assertEqual(engine.stack, [])
        self.assertEqual(len(trace.warm_steps), 21)
        self.assertIs(sys.argv, args_before)
        self.assertEqual(sys.path, path_before)
        self.assertIsNone(sys.gettrace())

    def test_missing_window_remains_incomplete(self):
        source = HERE / 'test_fixture.py'
        engine = Engine()
        trace = wrapper.LoopTrace(source, wrapper.landmarks(source.read_text()), engine)
        wrapper.run_exact(source, ['3'], trace)
        self.assertFalse(trace.finished)
        self.assertFalse(trace.active)
        self.assertNotIn(('start', 20), engine.events)
        trace.abort()
        self.assertEqual(engine.events[-1], ('abort', None))

    def test_fail_closed_if_source_landmark_changes(self):
        raw = (HERE / 'test_fixture.py').read_text().replace('loss = model(x, y)', 'loss = model(x, y + 1)')
        with self.assertRaisesRegex(RuntimeError, 'EXPECTED_LANDMARK_COUNT:forward'):
            wrapper.landmarks(raw)

    def test_capture_argv_has_no_overriding_or_extra_flags(self):
        for kind in wrapper.SOURCES.values():
            arguments = wrapper.launch_arguments(kind, 'out')
            wrapper.check_capture_arguments(kind, arguments)
            for extra in (['--max-train-seconds', '9999'], ['--num-steps', '1000'], ['--no-compile']):
                with self.assertRaisesRegex(RuntimeError, 'EXACT_PROFILE_ARGV_REQUIRED'):
                    wrapper.check_capture_arguments(kind, arguments + extra)

    def test_archived_tiny_actual_source_cpu_profile_identity(self):
        off = json.loads((HERE / 'tiny-off-r3.json').read_text())
        on = json.loads((HERE / 'tiny-on.json').read_text())
        self.assertEqual(off['verdict'], 'PASS')
        self.assertEqual(on['verdict'], 'PASS')
        self.assertEqual(off['train_py_sha256'], SOURCE_REFS['a4aa4351789115a2d0aa947d9ca00ff689f9bd0b'])
        self.assertEqual(off['train_py_sha256'], on['train_py_sha256'])
        self.assertEqual(off['plan_sha256'], on['plan_sha256'])
        for a, b in zip(off['ranks'], on['ranks'], strict=True):
            self.assertEqual(a['rank'], b['rank'])
            self.assertEqual(a['losses'], b['losses'])
            self.assertEqual(len(a['losses']), 3)
            self.assertEqual(a.get('final_digests'), b.get('final_digests'))
            self.assertEqual(b['cross_rank_mismatches'], [])
            capture = b['p_cpu_profile']
            self.assertEqual((capture['start_step'], capture['stop_step']), (1, 2))
            for name in ('forward', 'backward', 'optimizer', 'data_fetch', 'record_boundary'):
                self.assertEqual(capture['events'].count('P/' + name), 1)
            self.assertEqual(capture['events'].count('P/existing_synchronize'), 2)
        self.assertEqual(len(on['ranks'][0]['final_digests']), 22)
        self.assertTrue(on['ranks'][0]['checkpoint_check']['ok'])

    def test_existing_trace_hook_is_not_overridden(self):
        hook = lambda *args: None
        sys.settrace(hook)
        try:
            with self.assertRaisesRegex(RuntimeError, 'EXISTING_TRACE_HOOK_REFUSED'):
                wrapper.run_exact(HERE / 'test_fixture.py', ['1'])
            self.assertIs(sys.gettrace(), hook)
        finally:
            sys.settrace(None)

    def test_no_training_source_writes_or_dispatch_modes(self):
        source = (HERE / 'profile_actual_loop.py').read_text()
        tree = ast.parse(source)
        self.assertNotIn('TorchDispatchMode', source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    self.assertNotIn('f_locals[', ast.unparse(target))
                    self.assertNotIn('f_globals[', ast.unparse(target))


if __name__ == '__main__':
    unittest.main()
