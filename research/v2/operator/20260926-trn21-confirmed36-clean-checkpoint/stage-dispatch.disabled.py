"""Nonlaunching local transport preparation. Actual checkpoint INPUTS do not exist."""
import json,subprocess,sys
from pathlib import Path
REPO=Path('/home/poojith-devan/Desktop/Trainium-trn22-queue-operator')
if sys.argv[1:] != ['--local-check']:
    raise SystemExit('REAL_INPUTS_PENDING_NO_HOST_TRANSPORT_OR_LAUNCH_AUTHORITY')
result=subprocess.run([sys.executable,'-B',str(REPO/'research/v2/proposals/20260926-confirmed36-clean-checkpoint-cpu/test_operator_binding.py')],capture_output=True,text=True,timeout=30)
if result.returncode:
    sys.stderr.write(result.stderr);raise SystemExit(result.returncode)
receipt=json.loads(result.stdout)
if receipt['stage_authorized'] is not False or receipt['launch_authorized'] is not False:
    raise RuntimeError('DISABLED_CONTRACT_CHANGED')
print(json.dumps({'status':'DISABLED_TRANSPORT_INPUTS_NOT_MATERIALIZED','binding_checks':receipt['checks'],'host_calls':False,'stage_authorized':False,'launch_authorized':False}))
