"""Local stdlib packet checks; never imports or executes a native driver."""
import ast, hashlib, json, shlex, subprocess, unittest
from pathlib import Path
from types import SimpleNamespace
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
B = json.loads((HERE/'NATIVE_BINDING.disabled.json').read_text())
P = json.loads((HERE/'EXPERIMENT_PROPOSAL.json').read_text())
DRIVER = (HERE/'host_driver.py').read_text()
def sha(x): return hashlib.sha256(x).hexdigest()
def blob(ref,path): return subprocess.check_output(['git','show',ref+':'+path],cwd=REPO)
def consts():
    out = {}
    for node in ast.parse(DRIVER).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1:
            value=node.value
            if isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=='Path': value=value.args[0]
            out[node.targets[0].id]=ast.literal_eval(value)
    return out
class Binding(unittest.TestCase):
    def test_exact_driver_identity_and_pins(self):
        d=B['derivation'];raw=blob(d['parent_commit'],d['parent_driver_path'])
        self.assertEqual(sha(raw),d['parent_driver_sha256'])
        old=raw.decode()
        expected=old.replace('fs-20260926-arch36a8-startup-clock-trn22-r1-full',B['label']).replace('36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb',B['source_sha256']).replace('Exact cold-qualified architecture full research then fresh public20M','Exact formal Recursive6ef7 control then fresh public20M')
        self.assertEqual(expected.rstrip()+'\n',DRIVER)
        self.assertEqual(ast.dump(ast.parse(expected)),ast.dump(ast.parse(DRIVER)))
        self.assertEqual(sha(DRIVER.encode()),B['driver_sha256'])
        c=consts()
        for constant,field in [('ROOT','root'),('PREP','prepare_path'),('HARNESS','harness_path'),('SOURCE_SHA','source_sha256'),('PREP_SHA','prepare_sha256'),('HARNESS_SHA','harness_sha256'),('LABEL','label'),('TRAIN_COMMAND','command'),('EVAL_COMMAND','evaluation_command'),('TRAIN_ENV','training_environment'),('EVAL_ENV','evaluation_environment')]:self.assertEqual(c[constant],B[field])
    def test_source_support_and_unchanged_budget(self):
        source=blob(B['source_commit'],B['source_git_path'])
        self.assertEqual(sha(source),B['source_sha256']);self.assertEqual(B['source_sha256'],B['base_sha256'])
        self.assertEqual(B['source_commit'],'a4aa4351789115a2d0aa947d9ca00ff689f9bd0b')
        self.assertIn(b'--recursive-schedule',source)
        for flag in [b'--initial-cache-refresh',b'--unet-8x1024',b'--startup-adjusted-schedule',b'--fp8-proj']:self.assertNotIn(flag,source)
        for name,key in [('idle_guard.py','idle_guard_sha256'),('dispatch_driver.py','dispatch_driver_sha256')]:
            self.assertEqual(sha((REPO/'research/v2/operator/20260926-common-host-idle-guard'/name).read_bytes()),B[key])
        harness=blob('00f2ca7f79696e65ce52458f074dd13b871da90c','research/scratch/submission_rehearsal.py')
        self.assertEqual(sha(harness),B['harness_sha256'])
        self.assertIn(b'"--batch-size", "1", "--seq-len", "1024"',harness)
        self.assertIn(b'<= 1800',harness);self.assertIn(b'min(observed[4]["observed_seconds"], 900.0)',harness)
        self.assertEqual(sha((REPO/'prepare.py').read_bytes()),B['prepare_sha256'])
        self.assertEqual(B['budgets']['source_seconds'],1800);self.assertEqual(B['evaluation']['tokens'],20971520)
    def test_authentic_genuine_control_argv_env(self):
        p=P['provenance'];raw=blob(p['raw_launch_commit'],p['raw_launch'])
        self.assertEqual(sha(raw),p['raw_launch_sha256']);old=json.loads(raw)
        self.assertEqual(B['training_command'],old['command']);self.assertEqual(P['argv'],old['command'])
        self.assertEqual(shlex.split((HERE/'package/launch-command.txt').read_text()),old['command'])
        prov=json.loads((HERE/'package/provenance.json').read_text())
        self.assertEqual(shlex.split(prov['launch_command']),old['command'])
        self.assertEqual(prov['kind'],'control');self.assertEqual(prov['base_sha256'],B['base_sha256'])
        for name,digest in B['package_support_sha256'].items():self.assertEqual(sha((HERE/'package'/name).read_bytes()),digest)
        for key in ['NEURON_CC_FLAGS','NEURON_LOGICAL_NC_CONFIG','NEURON_COMPETITION_R1_SEED','NKI_VENV_PATH','PATH','PYTHONDONTWRITEBYTECODE','PYTHONUNBUFFERED','NEURON_COMPETITION_R1_CACHE_DIR']:
            self.assertEqual(B['training_environment'][key],old['environment'][key])
        self.assertEqual(B['training_environment']['NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS'],'1800')
        self.assertNotIn('PYTHONPATH',B['training_environment'])
        self.assertEqual(B['training_environment']['NEURON_COMPILE_CACHE_URL'],B['training_environment']['NEURON_COMPILE_CACHE_DIR'])
        self.assertNotIn('--fresh-caches',B['command']);self.assertNotIn('--fresh-caches',B['evaluation_command'])
    def test_disabled_roles_and_exact_host_guard(self):
        for obj in [P,B]:
            for key in ['launch_authorized','stage_authorized','submission_authorized','source_adoption','host_reserved']:self.assertFalse(obj[key])
            self.assertEqual(obj['kind'],'control');self.assertEqual(obj['host'],'trn22')
        self.assertIsNone(B['planned_record']['public_20m'])
        self.assertEqual(B['existing_trn22_schedule_arm']['kind'],'schedule_arm')
        self.assertEqual(B['existing_trn22_schedule_arm']['public20m'],.9902787975734985)
        self.assertEqual(B['source_provenance']['genuine_trn21_public20m'],.9899528600562257)
        self.assertEqual(B['hostname'],'ip-172-31-32-253');self.assertEqual(B['lock_identity']['expected_inode'],832589)
        self.assertEqual(B['outer_command'],['/usr/bin/flock','-n',B['device_lock'],'/usr/bin/python3','-I','-B',B['driver_path']])
        self.assertEqual(DRIVER.count('require_idle_under_inherited_lock('),2)
        self.assertIn('pass_fds=(lock_fd,)',DRIVER);self.assertIn("eval_fd==lock_fd",DRIVER)
        self.assertIn("10.5*1024**3",DRIVER)
    def test_real_preflight_negative_fixtures(self):
        node = next(n for n in ast.parse(DRIVER).body if isinstance(n, ast.FunctionDef) and n.name == 'preflight')
        state = {'stop':False, 'disk':11 * 1024**3, 'bad_sha':None, 'cache_missing':False}
        class FakePath:
            def __init__(self, value): self.value = str(value)
            def __truediv__(self, other): return FakePath(self.value + '/' + str(other))
            @property
            def parent(self): return FakePath(self.value.rsplit('/',1)[0])
            def __str__(self): return self.value
            def glob(self, pattern): return []
            def is_dir(self): return not state['cache_missing']
        def digest(path):
            value = str(path)
            if state['bad_sha'] and state['bad_sha'] in value: return 'wrong'
            if value.endswith('package/train.py'): return B['source_sha256']
            if value.endswith('prepare.py'): return B['prepare_sha256']
            return B['harness_sha256']
        ns = {'Path':FakePath,'ROOT':FakePath(B['root']),'PREP':FakePath(B['prepare_path']),
              'HARNESS':FakePath(B['harness_path']),'SOURCE_SHA':B['source_sha256'],
              'PREP_SHA':B['prepare_sha256'],'HARNESS_SHA':B['harness_sha256'],'sha':digest,
              'os':SimpleNamespace(path=SimpleNamespace(lexists=lambda p:state['stop']),
                                   statvfs=lambda p:SimpleNamespace(f_bavail=state['disk'],f_frsize=1))}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'exact-preflight','exec',optimize=0),ns)
        ns['preflight'](B['training_environment'])
        for field, bad in [('stop',True),('disk',0),('bad_sha','train.py'),('bad_sha','prepare.py'),
                           ('bad_sha','submission_rehearsal.py'),('cache_missing',True)]:
            old = state[field];state[field] = bad
            with self.assertRaises(AssertionError): ns['preflight'](B['training_environment'])
            state[field] = old


    def test_manifest(self):
        manifest=json.loads((HERE/'MANIFEST.sha256.json').read_text())
        self.assertFalse(manifest['launch_authorized'])
        for name,digest in manifest['files'].items():self.assertEqual(sha((HERE/name).read_bytes()),digest,name)
        present={str(x.relative_to(HERE)) for x in HERE.rglob('*') if x.is_file() and x.name!='MANIFEST.sha256.json' and '__pycache__' not in x.parts}
        self.assertEqual(present,set(manifest['files']))
if __name__=='__main__':unittest.main(verbosity=2)
