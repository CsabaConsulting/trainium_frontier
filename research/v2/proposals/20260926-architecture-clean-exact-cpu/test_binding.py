"""Stdlib binding/negative-fixture tests only; never runs the host driver."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PLAN=json.loads((HERE/'CPU_PLAN.disabled.json').read_text())
BASE=ROOT/'research/v2/proposals/20260926-base-cache-exact-cpu'


def dump(node): return ast.dump(node,include_attributes=False)
def function(tree,name): return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj


class BindingTests(unittest.TestCase):
    def test_source_pins(self):
        for name,source in PLAN['sources'].items():
            if 'commit' in source:
                raw=subprocess.check_output(['git','show',source['commit']+':'+source['git_path']],cwd=ROOT)
            else:
                raw=(ROOT/source.get('git_path','prepare.py')).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),source['sha256'],name)

    def test_exact_scored_and_long_argv(self):
        archived=json.loads((ROOT/'research/v2/operator/20260926-trn22-arch3e79-full/EXPERIMENT_PROPOSAL.json').read_text())['argv'][4:]
        long=(ROOT/'submissions/dense6x1024_direct_review_matrixlr015/launch-command.txt').read_text().strip().split('train.py ')[1].split(' ')
        for probe,argv in zip(PLAN['probes'],(archived,long)):
            request=json.loads((HERE/probe['request_path']).read_text())
            self.assertEqual(request['argv'],argv)
            self.assertEqual(request['command'],probe['command'])
            self.assertEqual(request['command'][request['command'].index('--')+1:],argv)
            for flag,value in [('--steps','3'),('--nproc','4'),('--micro-rows','2'),('--micro-tokens','256'),('--timeout','3600')]:
                self.assertEqual(request['command'][request['command'].index(flag)+1],value)
            self.assertNotIn('--trace-rmsprop',request['command'])
            self.assertNotIn('--stub-prepare',request['command'])
            self.assertNotIn('--real-data',request['command'])

    def test_disabled_full_shape(self):
        self.assertIs(PLAN['launch_authorized'],False)
        self.assertIs(PLAN['local_tensor_execution_prohibited'],True)
        self.assertEqual(PLAN['minimum_available_ram_gib'],64)
        self.assertEqual(PLAN['full_model']['parameters'],113256724)
        self.assertEqual(PLAN['full_model']['parameter_tensors'],65)
        self.assertEqual(PLAN['device_lock'],'/home/ubuntu/.cache/trainium_freshstart/device0.lock')

    def test_guard_parent_and_resource_checks_unchanged(self):
        tree=ast.parse((HERE/'host_driver.py').read_text())
        old=ast.parse((BASE/'candidate-on/host_driver.py').read_text())
        for name in ('checks','require','sha','record','utc'):
            self.assertEqual(dump(function(tree,name)),dump(function(old,name)),name)
        module(BASE/'audit_packet.py','old_audit').check_legacy_scan(tree,PLAN['expected_outer_command'])
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        children=[n for n in calls if n.func.attr in ('Popen','call')]
        self.assertEqual(len(children),2)
        for child in children:
            self.assertIn('(lock_fd,)',[ast.unparse(k.value) for k in child.keywords if k.arg=='pass_fds'])
        self.assertEqual(sum(n.func.attr=='require_idle_under_inherited_lock' for n in calls),3)
        for node in tree.body:
            if isinstance(node,ast.Assign) and getattr(node.targets[0],'id',None)=='PINS':
                pins=ast.literal_eval(node.value)
        for name,sha in pins.items():
            self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).hexdigest(),sha,name)

    def test_strict_receipts_with_negative_cases(self):
        validate=module(HERE/'validate_receipts.py','strict_receipts').validate
        raw=json.loads((ROOT/'research/v2/operator/20260926-trn22-arch-initial-cache-cpu-r1/results/candidate-on/g2.json').read_text())
        original={}
        for probe in PLAN['probes']:
            name=probe['name'];request=json.loads((HERE/probe['request_path']).read_text())
            report=copy.deepcopy(raw);report['argv']=request['argv'];report['train_py_sha256']=request['source_sha256']
            original[name+'/request.json']=request;original[name+'/g2.json']=report
            original[name+'/terminal.json']={'exit_code':0,'verdict':'PASS'}
            original[name+'/checkpoint-identity.json']={'bytes':453000000,'sha256':'a'*64}
        class FakePath:
            def __init__(self,key=''):self.key=key
            def __truediv__(self,name):return FakePath((self.key+'/'+name).lstrip('/'))
            def read_text(self):return json.dumps(data[self.key])
        data=copy.deepcopy(original)
        self.assertEqual(validate(FakePath())['status'],'FULL_PARAMETER_CPU_RECEIPTS_MATCH')
        mutations=[lambda d:d['candidate/g2.json'].update(timed_out=True),lambda d:d['candidate/g2.json'].update(torchrun_rc=1),lambda d:d['candidate/g2.json'].update(train_py_sha256='wrong'),lambda d:d['candidate/g2.json'].update(plan_sha256='wrong'),lambda d:d['candidate/g2.json'].update(micro_batch=[1,16]),lambda d:d['candidate/g2.json']['ranks'].pop(),lambda d:d['candidate/g2.json']['ranks'][0]['losses'].__setitem__(0,float('nan')),lambda d:d['candidate/g2.json']['ranks'][0]['losses'].__setitem__(0,99.),lambda d:d['candidate/g2.json']['ranks'][0].update(n_changed=64),lambda d:d['candidate/g2.json']['ranks'][0]['final_digests'].pop(next(iter(d['candidate/g2.json']['ranks'][0]['final_digests']))),lambda d:d['candidate/g2.json']['ranks'][0]['checkpoint_check'].update(ok=False)]
        for mutate in mutations:
            data=copy.deepcopy(original);mutate(data)
            with self.assertRaises((RuntimeError,KeyError)):
                validate(FakePath())


if __name__=='__main__':unittest.main(verbosity=2)
