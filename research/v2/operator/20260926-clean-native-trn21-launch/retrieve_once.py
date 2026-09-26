"""Use the exact accepted transport once; retain partials and literal refusal."""
import base64,importlib.util,json,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('local_one_use',HERE/'execute_once.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
con=m.load('connector',m.C/'research/v2/observations/20260925-dual-host-readonly-operator/collect_snapshot.py','7f9650a4ec071852f0db5a215a005d54c9a4813e5ae6bd37db393cb4984de899')
folder=m.C/'research/v2/proposals/20260926-confirmed36-clean-native-trn21-transport-r1'
tr=m.load('transport',folder/'transport.py','51e37ab4ca72a1cb48682a50dcf1eec64be101182f0c41dede5680a923997a8e')
packet=m.C/'research/v2/proposals/20260926-confirmed36-clean-native-trn21-packet-r1/packet'
wire=tr.make_wire(packet,HERE/'RELEASE.json',None,'retrieve')
argv,*_=con.load_transport('trn21');argv[-1]='exec /usr/bin/python3 -I -B -'
m.require(m.sha(json.dumps(argv,separators=(',',':')).encode())=='29bddb10fbd5fa7b0b351ba4e6a18fd8247c1754250f16911da887576fa9101e','EXACT_CONNECTION_ARGV')
started=con.utc();r=subprocess.run(argv,input=wire,capture_output=True,timeout=60)
m.require(con.SENSITIVE.search(r.stdout+r.stderr) is None,'SENSITIVE_OUTPUT')
for line in r.stdout.splitlines():
 row=json.loads(line)
 if row.get('kind')=='chunk':m.require(con.SENSITIVE.search(base64.b64decode(row['b64'],validate=True)) is None,'SENSITIVE_CHUNK')
with (HERE/'retrieval-wire.jsonl').open('xb') as f:f.write(r.stdout)
result={'started_utc':started,'ended_utc':con.utc(),'returncode':r.returncode,'stderr':r.stderr.decode(),'wire_sha256':m.sha(r.stdout),'wire_bytes':len(r.stdout),'source_sha256':m.sha(wire)}
try:result['receiver']=tr.receive(iter(r.stdout.splitlines(keepends=True)),HERE/'retrieved')
except Exception as exc:result['receiver_error']=repr(exc)
m.save('retrieval-outcome.json',result)
print(json.dumps(result,indent=2))
