import ast
import copy
import hashlib
import json
import shlex
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
B = json.loads((ROOT/'CT_BINDING.disabled.json').read_text())
P = json.loads((ROOT/'parent_binding.json').read_text())
Q = json.loads((ROOT/'EXPERIMENT_PROPOSAL.json').read_text())
def sha(name):
    return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
def literals(name):
    result = {}
    for n in ast.parse((ROOT/name).read_text()).body:
        if isinstance(n, ast.Assign):
            value = n.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'Path':
                value = value.args[0]
            try:
                result[n.targets[0].id] = ast.literal_eval(value)
            except (ValueError, AttributeError):
                pass
    return result

def decision(observations, wall, checkpoint, workers, finite):
    # Independently specified qualification reference; NOT a production gate.
    if [x[0] for x in observations] != list(range(5)):
        return False
    fifth = observations[4][1]
    return (0 <= fifth <= 900 and wall >= fifth and
            wall-min(fifth, 900) <= 1800 and checkpoint and workers == 4 and finite)

class Review(unittest.TestCase):
    def test_entire_driver(self):
        parent = (ROOT/'parent_host_driver.py').read_text()
        replaced = parent.replace(P['label'], B['label']).replace(P['source_sha256'], B['source_sha256'])
        self.assertEqual(replaced, (ROOT/'host_driver.py').read_text())
        d = literals('host_driver.py')
        for key, field in [('ROOT','root'), ('LABEL','label'), ('SOURCE_SHA','source_sha256'), ('COMMAND','command'), ('ENV','training_environment')]:
            self.assertEqual(d[key], B[field])
        tree = ast.parse((ROOT/'host_driver.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        launch = [n for n in calls if ast.unparse(n.func) == 'subprocess.call']
        self.assertEqual(len(launch),1)
        self.assertEqual(ast.unparse(next(k.value for k in launch[0].keywords if k.arg=='pass_fds')), '(lock_fd,)')

    def test_hashes(self):
        for file, key in [('host_driver.py','driver_sha256'),('idle_guard.py','idle_guard_sha256'),('dispatch_driver.py','dispatch_driver_sha256'),('submission_rehearsal.py','harness_sha256'),('EXPERIMENT_PROPOSAL.json','proposal_sha256')]:
            self.assertEqual(sha(file), B[key])
        self.assertEqual(sha('parent_host_driver.py'),B['driver_derivation']['sha256'])
        for file in ['launch-command.txt','provenance.json']:
            self.assertEqual(sha(file), B['package_files']['package/'+file])
        for key in ['command','training_command','outer_command','detached_command']:
            self.assertEqual(hashlib.sha256(json.dumps(B[key],separators=(',',':'),ensure_ascii=True).encode()).hexdigest(),B[key+'_sha256'])

    def test_argv(self):
        command = B['training_command']
        self.assertEqual(command, P['training_command']+['--recursive-schedule'])
        self.assertEqual(command,Q['argv'])
        self.assertEqual(command,shlex.split((ROOT/'launch-command.txt').read_text()))
        provenance=json.loads((ROOT/'provenance.json').read_text())
        self.assertEqual(shlex.split(provenance['launch_command']),command)
        self.assertEqual(provenance['train_sha256'],B['source_sha256'])
        for flag,value in [('--num-steps','5'),('--max-train-seconds','1800'),('--seed','42')]:
            self.assertEqual(command[command.index(flag)+1],value)
        self.assertIn('--nproc_per_node=4',command)
        self.assertIn('--no-eval-public',command)
        self.assertEqual(B['training_environment']['NEURON_LOGICAL_NC_CONFIG'],'2')
        self.assertEqual(B['command'][:5],['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=60s','1800s'])
        self.assertEqual(B['command'][-3:],['--mode','train','--fresh-caches'])
        self.assertNotIn('-O',B['outer_command'])

    def test_cache_mechanics(self):
        tree=ast.parse((ROOT/'submission_rehearsal.py').read_text())
        cache_loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='(key, directory)')
        pairs=ast.literal_eval(cache_loop.iter)
        self.assertEqual(len(pairs),6)
        self.assertEqual(len(set(d for k,d in pairs)),6)
        for key,directory in pairs:
            self.assertEqual(B['harness_injected_environment'][key],B['sandbox']+'/caches/'+directory)
        text=ast.unparse(cache_loop)
        self.assertIn('exist_ok=False',text)
        self.assertIn('assert not list(cache.iterdir())',text)
        self.assertEqual(B['training_environment']['NEURON_COMPILE_CACHE_URL'],B['harness_injected_environment']['NEURON_COMPILE_CACHE_DIR'])
        self.assertNotIn('NEURON_COMPETITION_R1_CACHE_DIR',dict(pairs))

    def test_disabled_and_operator_requirements(self):
        for obj in [B,Q]:
            for key in ['stage_authorized','launch_authorized','native_launch_authorized']:
                self.assertIs(obj[key],False)
            for word in ['Operator','hostname','refuse mismatch']:
                self.assertIn(word,obj['host_identity'])
            for word in ['Every outcome','starttime','PPID/PGID/SID','compiler','device','HOLD']:
                self.assertIn(word,obj['postflight'])
        self.assertEqual((B['original_lock']['device'],B['original_lock']['inode']),(66305,832589))
        self.assertIn('Operator',B['original_lock']['enforcement'])
        self.assertTrue(B['no_eval'] and B['no_automatic_full'])
        self.assertIn('HOLD',B['source_gate']['fullsize_cpu'])

    def test_uncapped_reference(self):
        observations=list(enumerate([10,20,30,40,900]))
        self.assertTrue(decision(observations,950,True,4,True))
        for obs,wall,ck,workers,finite in [(observations[:-1],950,True,4,True),(list(enumerate([10,20,30,40,900.01])),950,True,4,True),(observations,2701,True,4,True),(observations,950,False,4,True),(observations,950,True,3,True),(observations,950,True,4,False)]:
            self.assertFalse(decision(obs,wall,ck,workers,finite))
        self.assertEqual(B['cold_decision']['maximum_seconds'],900)
        self.assertIn('Uncapped',B['cold_decision']['measurement'])

    def test_idle_mock(self):
        tree=ast.parse((ROOT/'idle_guard.py').read_text())
        # Only pure helpers extracted; never execute imports, top-level drivers or /proc access.
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['digest','check_idle']]
        ns={'hashlib':hashlib,'Path':Path,'subprocess':subprocess,'json':json}
        ns.update(literals('idle_guard.py'))
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'extracted_pure_guard','exec'),ns)
        device={'neuron_device':0,'instance_type':'trn2.3xlarge','nc_count':4,'logical_neuroncore_config':2,'neuroncore_ids':[0,1,2,3],'neuron_processes':[]}
        def check(ps,dev):
            def run(cmd,**kwargs):
                return SimpleNamespace(returncode=0,stdout=ps if cmd==ns['PS'] else json.dumps(dev),stderr='')
            ns['check_idle'](run,10,11,{})
        rows='10 11 python host_driver.py\n11 1 flock flock\n'
        check(rows,[device])
        for ps,dev in [(rows+'12 1 neuronx-cc neuronx-cc\n',[device]),(rows,[]),(rows,[dict(device,neuron_processes=[{'pid':50}])]),('bad',[device])]:
            with self.assertRaises((RuntimeError,ValueError)):
                check(ps,dev)

if __name__ == '__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Review)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt={'status':'PASS' if result.wasSuccessful() else 'REFUSE','tests':result.testsRun,'optimized':sys.flags.optimize,'input_sha256':{p.name:sha(p.name) for p in sorted(ROOT.iterdir()) if p.is_file()},'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'stdlib packet mechanics only; CPU/native/host qualification absent; no launch release'}
    (Path(__file__).parent/('optimized.json' if sys.flags.optimize else 'normal.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
