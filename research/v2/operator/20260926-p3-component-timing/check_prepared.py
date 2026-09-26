"""Local mechanical binding checks only, never imports diagnostic source."""
import ast, base64, hashlib, json, shlex
from pathlib import Path
here=Path(__file__).resolve().parent
checks=0
def require(value):
    global checks
    if not value:raise RuntimeError('P3_BINDING_CHECK_FAILED')
    checks+=1
def sha(raw):return hashlib.sha256(raw).hexdigest()
for row in json.loads((here/'PREPARED.json').read_text())['alternatives_for_one_selected_host']:
    raw=(here/row['packet']).read_bytes();require(sha(raw)==row['packet_sha256'])
    packet=json.loads(raw);b=packet['binding']
    files={name:base64.b64decode(value['b64'],validate=True) for name,value in packet['files'].items()}
    for name,value in files.items():require(sha(value)==packet['files'][name]['sha256'])
    require(json.loads(files['NATIVE_BINDING.disabled.json'])==b)
    require(sha(files['package/train.py'])==b['source_sha256']=='3637166b43a0b996e7eb7c71bb3b5d56e7f8119241b0278107a72787044dcc99')
    require(sha(files['package/prepare.py'])==b['prepare_sha256']=='5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb')
    require(sha(files['host_driver.py'])==b['driver_sha256']==row['driver_sha256'])
    parent=json.loads((here.parent/'20260926-final-sprint-p0'/(row['host']+'-bound-packet.json')).read_text())
    for name in ('dispatch_driver.py','idle_guard.py','lifetime_guard.py'):require(packet['files'][name]==parent['files'][name])
    tree=ast.parse(files['host_driver.py']);compile(tree,row['label'],'exec')
    assigns={target.id:ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) for target in node.targets if isinstance(target,ast.Name) and target.id in ('TRAIN_ENV','TRAIN_COMMAND','EVAL_ENV','EVAL_COMMAND','LABEL','SOURCE_SHA')}
    require(assigns['TRAIN_ENV']==b['environment'] and assigns['TRAIN_COMMAND']==b['diagnostic_command'])
    require(assigns['LABEL']==b['label'] and assigns['SOURCE_SHA']==b['source_sha256'])
    require(assigns['EVAL_ENV'] is None and assigns['EVAL_COMMAND'] is None)
    require(not any(isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load) and node.id in ('HARNESS','HARNESS_SHA','EVAL_ENV','EVAL_COMMAND') for node in ast.walk(tree)))
    phases=[node.args[0].value for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='phase' and node.args and isinstance(node.args[0],ast.Constant)]
    require(phases==['component-diagnostic'])
    require('rehearsal.json' not in files['host_driver.py'].decode() and 'checkpoint.json' not in files['host_driver.py'].decode())
    argv=b['diagnostic_command'];require(argv[:6]==['/usr/bin/timeout','--foreground','--signal=TERM','--kill-after=15s','930s','/home/ubuntu/workspace/native_venv/bin/torchrun'])
    require(argv[6:]==shlex.split(b['prospective_launch_command'])[2:])
    require(argv[argv.index('--p3-max-seconds')+1]=='900' and argv[argv.index('--seed')+1]=='42')
    require(argv[argv.index('--p3-blocks')+1]=='4' and argv[argv.index('--p3-repeats')+1]=='3')
    require(argv[argv.index('--out-dir')+1]==b['output_directory'])
    require(len(b['fresh_cache_directories'])==6 and all(path.startswith(b['root']+'/cold-diagnostic-caches/') for path in b['fresh_cache_directories']))
    require(b['lock_identity']==parent['binding']['lock_identity'])
    require(not b['launch_authorized'] and not b['stage_authorized'] and 'RELEASE.json' not in files and 'EXPERIMENT_PROPOSAL.json' not in files)
    require(b['no_harness'] and b['no_checkpoint'] and b['no_eval'] and b['no_ntff'])
    require(b['source_max_seconds']==900 and b['outer_timeout_seconds']==930 and b['outer_kill_after_seconds']==15)
    require(b['guardian_term_grace_seconds']==15 and b['guardian_kill_wait_seconds']==10)
    require('grace=15.0,kill_wait=10.0' in files['host_driver.py'].decode())
    require('guardian.run(command,cwd=ROOT/\'package\',env=env,out=out)' in files['host_driver.py'].decode())
    require('UNCHANGED_IMPORT_PREPARE_SHA' in files['host_driver.py'].decode())
print(json.dumps({'status':'MECHANICAL_BINDING_PASS','checks':checks,'native_result':None,'host_calls':0,'model_imports':0,'new_containment_framework':False}))
