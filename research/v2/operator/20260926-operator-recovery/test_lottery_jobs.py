"""Local package invariants; no SSH, compiler or launch."""
import ast
import base64
import importlib.util
import json
from pathlib import Path
import types
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('lottery_guard',HERE/'manual_job.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Packets(unittest.TestCase):
    def test_both_keep_original_guards_and_unique_empty_cache_plan(self):
        for key,parent in [('trn22-compilerdefault5','trn22-muon150'),('trn21-a1framework','trn21-finalcold')]:
            _,p=m.checked_packet(key);_,q=m.checked_packet(parent)
            for name in ('idle_guard.py','lifetime_guard.py','dispatch_driver.py'):
                self.assertEqual(p['files'][name],q['files'][name])
            self.assertEqual(p['binding']['lock_identity'],q['binding']['lock_identity'])
            paths=p['binding']['fresh_cache_directories']
            self.assertEqual(len(paths),6)
            self.assertTrue(all(x.startswith(p['binding']['root']+'/cold-training-caches/') for x in paths))
            self.assertEqual(p['binding']['budget']['source_seconds'],1800)
            self.assertFalse(p['binding']['submission_authorized'])
            self.assertEqual(p['binding']['eval_environment']['NEURON_CC_FLAGS'],'--optlevel=1')
    def test_default_compiler_does_not_fall_back_to_optlevel1(self):
        _,p=m.checked_packet('trn22-compilerdefault5')
        code=base64.b64decode(p['files']['package/train.py']['b64']).decode()
        env=dict(p['binding']['train_environment'])
        self.assertEqual(env['NEURON_CC_FLAGS'],'--auto-cast matmult --auto-cast-type bf16')
        # Execute actual import-time setdefaults only, with an isolated os facade.
        nodes=[n for n in ast.parse(code).body if isinstance(n,ast.If) and ast.unparse(n.test)=="__name__ == '__main__'"]
        self.assertEqual(len(nodes),2)
        prefix=nodes[0]
        exec(compile(ast.Module(body=[prefix],type_ignores=[]),'<actual-defaults>','exec'),{'os':types.SimpleNamespace(environ=env),'__name__':'__main__'})
        self.assertNotIn('optlevel',env['NEURON_CC_FLAGS'])
        self.assertTrue(p['binding']['no_eval']);self.assertEqual(p['binding']['steps'],5)
        self.assertTrue(p['binding']['launch_command'].endswith('--num-steps 5'))
    def test_table_program_rejects_custom_kernel_and_reuses_functions(self):
        _,p=m.checked_packet('trn21-a1framework')
        code=base64.b64decode(p['files']['package/train.py']['b64']).decode()
        parent=(m.REPO/'research/v2/lanes/tables-compact-k2/train.py').read_text()
        def bodies(text): return {n.name:ast.dump(n) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        a,b=bodies(parent),bodies(code)
        self.assertEqual([k for k in a if a[k]!=b[k]],['main'])
        self.assertIn("if K2_EMBEDDING:\n        parser.error('K2 is closed",code)
        self.assertFalse(p['binding']['no_eval'])
        self.assertNotIn('--num-steps',p['binding']['launch_command'])
        self.assertIsNone(p['binding']['independent_review_sha256'])

if __name__=='__main__':unittest.main()
