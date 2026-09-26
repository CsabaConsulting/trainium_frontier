"""Independent local-only review. Never imports driver main or runs host commands."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest
import io
import math
import re
import sys
from unittest.mock import patch
from types import SimpleNamespace

HERE = Path(__file__).parent
INPUT = Path('/tmp/trainium-anchored-control-review-gbfk9v')
SNAP = INPUT/'snapshot'
REL = Path('research/v2/proposals/20260926-clock36-anchored-warm-control')
PACK = SNAP/REL
PLAN = json.loads((PACK/'ANCHORED_BINDING.disabled.json').read_text())
RAW = json.loads((SNAP/PLAN['anchor_raw_path']).read_text())
ACTUAL = RAW['remote']['completed_parent_metadata_only']

def must(ok, message):
    if not ok:
        raise ValueError(message)

def exact_anchor(anchor):
    must(anchor.keys() == ACTUAL.keys(), 'path set')
    for path, expected in ACTUAL.items():
        must(anchor[path].keys() == expected.keys(), 'field set')
        for key, value in expected.items():
            must(type(anchor[path][key]) is type(value), 'type '+key)
            must(anchor[path][key] == value, 'value '+key)

def load_identity():
    spec = importlib.util.spec_from_file_location('review_identity', PACK/'identity_checks.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class Review(unittest.TestCase):
    def test_manifest(self):
        raw = (INPUT/'FILE_SHA256.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), '0eaf757f0843e6d7c0ecdfb2dbefd1d666566f31f484bef143850cbc2df2a913')
        manifest = json.loads(raw)
        self.assertEqual(len(manifest['files']), 30)
        for entry in manifest['files']:
            data = (INPUT/entry['relative_path']).read_bytes()
            self.assertEqual(len(data), entry['bytes'])
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry['sha256'])

    def test_integer_anchor_and_negative_mutations(self):
        exact_anchor(PLAN['anchor_observation'])
        original = json.loads((INPUT/'original-refused'/REL/'ANCHORED_BINDING.disabled.json').read_text())
        with self.assertRaises(ValueError): exact_anchor(original['anchor_observation'])
        for path in ACTUAL:
            for field in ('mtime_ns', 'ctime_ns'):
                value = ACTUAL[path][field]
                self.assertGreater(value, 2**53)
                self.assertNotEqual(int(float(value)), value)
                for bad in (value+1, int(float(value)), float(value)):
                    changed = copy.deepcopy(PLAN['anchor_observation'])
                    changed[path][field] = bad
                    with self.assertRaises(ValueError): exact_anchor(changed)

    def test_argv_environment_and_output_reference(self):
        parent = SNAP/'research/v2/proposals/20260926-startup-adjusted-full-trn21-replication'
        old = shlex.split((parent/'package/launch-command.txt').read_text())
        new = shlex.split((PACK/'package/launch-command.txt').read_text())
        self.assertEqual(old, PLAN['parent_training_command'])
        self.assertEqual(new, PLAN['training_command'])
        self.assertEqual(len(old), len(new))
        self.assertEqual([i for i,(a,b) in enumerate(zip(old,new)) if a != b], [3])
        self.assertEqual(new[3], PLAN['executed_source'])
        self.assertEqual(new[new.index('--out-dir')+1], 'out')
        parent_plan = json.loads((parent/'NATIVE_BINDING.disabled.json').read_text())
        for phase, allowed in [('training', {'TRN_EXP_LABEL','TORCH_NEURONX_METRICS_DIR'}),('evaluation', {'TRN_EXP_LABEL'})]:
            current = PLAN[phase+'_environment']
            previous = PLAN['parent_'+phase+'_environment']
            self.assertEqual(previous, parent_plan[phase+'_environment'])
            self.assertEqual(set(current),set(previous))
            self.assertEqual({k for k in current if current[k]!=previous[k]},allowed)
            self.assertNotIn('PYTHONPATH', current)
            self.assertNotIn('PYTHONHOME', current)
            self.assertEqual(current['PYTHONDONTWRITEBYTECODE'],'1')
        root = Path(PLAN['root'])
        self.assertFalse(Path(PLAN['executed_source']).is_relative_to(root))
        self.assertEqual(Path(PLAN['train_sandbox']),root/'train')
        self.assertEqual(Path(PLAN['eval_sandbox']),root/'eval')
        command=PLAN['eval_command']
        self.assertEqual(command[command.index('--checkpoint')+1], str(root/'train/out/final.pt'))
        self.assertEqual(command[command.index('--eval-tokens')+1],'20971520')

    def test_real_filesystem_negatives_and_prepare0600(self):
        checks=load_identity()
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root=Path(directory)
            prepare=root/'prepare.py'
            prepare.write_text('VALUE = 1\n')
            prepare.chmod(0o600)
            pin=hashlib.sha256(prepare.read_bytes()).hexdigest()
            baseline=checks.identity(prepare,pin,readonly=False)
            self.assertEqual(baseline['st_mode'] & 0o777,0o600)
            with self.assertRaises(RuntimeError): checks.identity(prepare,pin,readonly=True)
            with self.assertRaises(RuntimeError): checks.identity(prepare,'0'*64)
            os.link(prepare,root/'hard')
            with self.assertRaises(RuntimeError): checks.identity(prepare,pin)
            (root/'hard').unlink()
            (root/'link').symlink_to(prepare)
            with self.assertRaises(RuntimeError): checks.identity(root/'link',pin)
            (root/'ancestor').symlink_to(root,target_is_directory=True)
            with self.assertRaises(RuntimeError): checks.identity(root/'ancestor/prepare.py',pin)
            before=checks.identity(prepare,pin)
            replacement=root/'replacement'
            replacement.write_bytes(prepare.read_bytes())
            replacement.chmod(0o600)
            os.replace(replacement,prepare)
            after=checks.identity(prepare,pin)
            self.assertEqual(before['sha256'],after['sha256'])
            self.assertNotEqual(before['st_ino'],after['st_ino'])
            self.assertNotEqual(before,after)

    def test_actual_import_precheck_rejects_shadow(self):
        tree=ast.parse((PACK/'host_driver.py').read_text())
        names={'require','anchored_inputs'}
        module=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names],type_ignores=[])
        from importlib.machinery import PathFinder
        scope={'Path':Path,'PathFinder':PathFinder}
        exec(compile(module,'isolated-anchored-inputs','exec'),scope)
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root=Path(directory)
            (root/'prepare.py').write_text('VALUE=1\n')
            (root/'json.py').write_text('raise RuntimeError("must never execute")\n')
            plan={'executed_source':str(root/'train.py'),'expected_prepare_import':str(root/'prepare.py'),'source_import_roots':['prepare','json']}
            with self.assertRaisesRegex(RuntimeError,'SCRIPT_DIRECTORY_IMPORT_SHADOW:json'):
                scope['anchored_inputs'](None,plan)
            (root/'json.py').unlink()
            (root/'prepare').mkdir()
            (root/'prepare/__init__.py').write_text('VALUE=2\n')
            with self.assertRaisesRegex(RuntimeError,'EXPECTED_PREPARE_RESOLUTION'):
                scope['anchored_inputs'](None,plan)

    def test_parent_operations_lock_and_warm_miss(self):
        current=ast.parse((PACK/'host_driver.py').read_text())
        parent=ast.parse((SNAP/'research/v2/proposals/20260926-startup-adjusted-full-trn21-replication/host_driver.py').read_text())
        for name in ('phase','preflight'):
            self.assertEqual(ast.dump(next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name==name)),ast.dump(next(n for n in parent.body if isinstance(n,ast.FunctionDef) and n.name==name)))
        scope={'os':os,'math':math,'re':re}
        names={'require','check_lock','account_training'}
        module=ast.Module(body=[n for n in current.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
        exec(compile(module,'isolated-lock-accounting','exec'),scope)
        with patch.object(os,'fstat',return_value=SimpleNamespace(st_dev=66305,st_ino=596823)):
            scope['check_lock'](7,PLAN)
        with patch.object(os,'fstat',return_value=SimpleNamespace(st_dev=66305,st_ino=596824)):
            with self.assertRaisesRegex(RuntimeError,'ORIGINAL_LOCK_IDENTITY'):
                scope['check_lock'](7,PLAN)
        receipt={'returncode':0,'mode':'train','step_boundaries':5,'observations':[{'step':i,'observed_seconds':t} for i,t in enumerate([50,100,150,180,181])],'wall_seconds':1900,'startup_allowance_observed':181,'charged_through_process_exit':1719,'charged_through_process_exit_within_budget':True,'fresh_empty_sdk_caches':{}}
        log='\n'.join(f'step {i} | loss 3.0 | lrm 1.0 | dt 1.0s |' for i in range(5))
        result=scope['account_training'](receipt,log)
        self.assertEqual(result['warm_target_disposition'],'WARM_TARGET_MISSED')
        self.assertEqual(result['cache_hit'],'UNPROVEN')
        receipt.update(wall_seconds=2000,charged_through_process_exit=1819,charged_through_process_exit_within_budget=False)
        with self.assertRaisesRegex(RuntimeError,'CHARGED_THROUGH_EXIT'):
            scope['account_training'](receipt,log)

if __name__ == '__main__':
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Review))
    output=stream.getvalue()
    (HERE/('raw-optimized.log' if sys.flags.optimize else 'raw-normal.log')).write_text(output)
    print(output,end='')
    raise SystemExit(0 if result.wasSuccessful() else 1)
