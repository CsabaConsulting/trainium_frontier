"""Own bounded stdlib references; no Torch/NKI/tensors/hosts/data."""
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

from reference import C, T, V, CASE_NAMES, dense, fixture, grouped_wrong, metadata, raw
from reference_symmetric import symmetric

HERE = Path(__file__).resolve().parent
ROWS = []
resource.setrlimit(resource.RLIMIT_AS, (128 * 1024**2,) * 2)


class Symmetric(unittest.TestCase):
    def compare_case(self, name, ids, gradients, meta):
        actual, traces, owners = symmetric(ids, gradients, meta)
        expected = dense(ids, gradients)
        self.assertEqual(raw(actual), raw(expected), name)
        self.assertEqual(len(owners), V*C)
        self.assertEqual(set(owners), {1})
        self.assertEqual([t['stores'] for t in traces], [V*64, V*64])
        self.assertEqual([t['group_rounds'] for t in traces], [meta[3], meta[3]])
        self.assertEqual(traces[0]['schedule'], traces[1]['schedule'])
        self.assertEqual(traces[0]['visits'], traces[1]['visits'])
        self.assertEqual(sum(t['active_gradient_elements'] for t in traces), len(ids)*C)
        for row in range(V):
            self.assertEqual([t for r,t in traces[0]['visits'] if r == row],
                             [i for i,r in enumerate(ids) if r == row])
        ROWS.append(dict(case=name, exact=True, group_rounds=meta[3],
                         per_program_iterations=[len(t['schedule']) for t in traces],
                         output_sha256=hashlib.sha256(raw(actual)).hexdigest(),
                         unique_output_elements=len(owners), stores_per_program=V*64))

    def test_all_preserved_cases_both_dtypes(self):
        for dtype in ('bf16','fp32'):
            for name in CASE_NAMES:
                with self.subTest(dtype=dtype,case=name):
                    self.compare_case(dtype+':'+name,*fixture(name,dtype))

    def test_uneven_counts_and_feature_signatures(self):
        for case,ids in (
                ('left_only',[0]*256),('right_only',[255]*256),
                ('uneven_left',[127]*255+[128]),('uneven_right',[0]+[128]*255),
                ('interleaved_uneven',[255,0,127]*85+[0]),
                ('one_token',[128]),('zero_trip',[])):
            gradients=[[float((token%7-3)*128+feature) for feature in range(C)] for token in range(len(ids))]
            self.compare_case(case,ids,gradients,metadata(ids))

    def test_rejected_grouped_sum_counterexamples_still_fail(self):
        for dtype in ('bf16','fp32'):
            for name in ('cross_tile_large','cross_tile_unit'):
                ids,gradients,_=fixture(name,dtype)
                self.assertNotEqual(raw(grouped_wrong(ids,gradients)),raw(dense(ids,gradients)))

    def test_unstable_metadata_and_nonfinite_refused(self):
        ids,gradients,meta=fixture('all_same','fp32')
        broken=tuple(list(x) for x in meta)
        broken[2][0],broken[2][1]=broken[2][1],broken[2][0]
        with self.assertRaises(ValueError):symmetric(ids,gradients,broken)
        for value in (math.nan,math.inf,-math.inf):
            bad=[row.copy() for row in gradients]
            bad[0][0]=value
            with self.assertRaises(ValueError):symmetric(ids,bad,meta)

    def test_factory_disabled_then_explicit_grid_and_global_aliases(self):
        name='symmetric_consumer_under_test'
        spec=importlib.util.spec_from_file_location(name,HERE/'consumer_nki.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.dict(sys.modules,{'nki':None,'nki.isa':None,'nki.language':None}):
            with self.assertRaisesRegex(RuntimeError,'EXPERIMENT_DISABLED'):module.build_kernel()
        fake=types.ModuleType('nki')
        fake.isa=types.ModuleType('nki.isa')
        fake.language=types.ModuleType('nki.language')
        captured={}
        class Jit:
            def __init__(self,function):captured['function']=function
            def __getitem__(self,grid):
                captured['grid']=grid
                return self
        fake.jit=Jit
        with patch.dict(sys.modules,{'nki':fake,'nki.isa':fake.isa,'nki.language':fake.language}):
            module.build_kernel(enabled=True)
        self.assertEqual(captured['grid'],2)
        self.assertIs(captured['function'].__globals__['nl'],fake.language)
        self.assertIs(captured['function'].__globals__['nisa'],fake.isa)
        self.assertNotIn('nl',captured['function'].__code__.co_freevars)

    def test_source_ownership_and_program_independent_control(self):
        source=(HERE/'consumer_nki.py').read_text()
        tree=ast.parse(source)
        kernel=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='resident_consumer')
        self.assertFalse(any(isinstance(n,ast.If) for n in ast.walk(kernel)))
        program_calls=[n for n in ast.walk(kernel) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='program_id']
        self.assertEqual(len(program_calls),1)
        self.assertIn('feature_first = nl.program_id(0) * 64',source)
        self.assertIn('assert nl.num_programs() == 2',source)
        loops=[ast.unparse(n.iter) for n in ast.walk(kernel) if isinstance(n,ast.For)]
        self.assertEqual(loops,['range(2)','nl.dynamic_range(limit)'])
        self.assertIn('src=group_maximums[group:group + 1, :]',source)
        self.assertIn('pattern=[[128, 128], [1, 64]], offset=feature_first',source)
        self.assertIn('output[first:first + 128, feature_first:feature_first + 64]',source)
        self.assertIn('return resident_consumer[2]',source)
        # Program-dependent value can occur only at definition, gather offset, store interval.
        uses=[n for n in ast.walk(kernel) if isinstance(n,ast.Name) and n.id=='feature_first']
        self.assertEqual(len(uses),4)

    def test_oracle_checker_unchanged_and_disabled_checker(self):
        for name,expected in (
            ('reference.py','14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f'),
            ('check_native.py','89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850')):
            self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).hexdigest(),expected)
        result=subprocess.run([sys.executable,'-B',str(HERE/'check_native.py')],
                              text=True,capture_output=True,timeout=5,check=True)
        self.assertEqual(json.loads(result.stdout),dict(status='DISABLED',native_imported=False,native_executed=False))


if __name__ == '__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Symmetric))
    memory={line.split(':',1)[0]:line.split(':',1)[1].strip()
            for line in Path('/proc/self/status').read_text().splitlines() if line.startswith(('VmRSS:','VmHWM:'))}
    print(json.dumps(dict(scope='IMPLEMENTER_STDLIB_ONLY_NOT_NATIVE',passed=result.wasSuccessful(),
                          tests=result.testsRun,records=ROWS,maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          current_exec_memory=memory,address_space_hard_bytes=128*1024**2),
                     sort_keys=True,allow_nan=False))
    sys.exit(0 if result.wasSuccessful() else 1)
