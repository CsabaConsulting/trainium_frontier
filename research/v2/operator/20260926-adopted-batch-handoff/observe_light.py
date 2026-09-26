"""Bounded existing-file status observer only; never launches or changes a host."""
import ast,hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
assert len(sys.argv)==3 and sys.argv[1] in ('trn21','trn22')
host,tag=sys.argv[1:]
choice,pid,start,ino={
 'trn21':('trn21-batch-recursive-stack','3283774','95946276','596823'),
 'trn22':('trn22-batch-lr02625','3400752','103355943','832589')}[host]
old=HERE.parent/'20260926-k2-runtimegroups-handoff/observe_boundary.py';raw=old.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='0ef6a7a2f16bcd608f224ca6fe1e2c1097ae7cc4d5baddb40ac3eb2483b915c4'
text=raw.decode()
for a,b in [('trn22-runtimegroups-native201-bound-packet.json',choice+'-bound-packet.json'),
 ("b['host']=='trn22'", "b['host']=="+repr(host)),
 ("'trn22-boundary-'",repr(host+'-status-')),
 ("m.load_transport('trn22')","m.load_transport("+repr(host)+")"),
 ('3399167',pid),('103271087',start),('832589',ino)]:
 assert a in text,a;text=text.replace(a,b)
old_files="('outer-flock-exit.json',8192)]:"
new_files="('outer-flock-exit.json',8192),('fresh-public20m-launch.json',8192),('fresh-public20m.log',16384),('eval/rehearsal.json',131072),('eval-exit-code.txt',128)]:"
assert text.count(old_files)==1;text=text.replace(old_files,new_files)
old_print="print(json.dumps({'returncode':r.returncode,'error':receipt['stderr'],'remote':receipt['remote'],'receipt_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))"
new_print="""remote=receipt['remote'] or {};ff=remote.get('files',{})
latest=[x for x in ff.get('training.log',{}).get('text','').splitlines() if x.startswith('step=') or x.startswith('step ') or 'TRAINING_END_UTC=' in x or 'Traceback' in x or 'RuntimeError' in x or 'COMPILATION FAILED' in x]
print(json.dumps({'host':b['host'],'returncode':r.returncode,'error':receipt['stderr'],'utc':remote.get('utc_after'),'boundaries':remote.get('complete_live_boundary_count'),'fifth':remote.get('fifth_complete'),'guardian':remote.get('guardian'),'latest_training_lines':latest[-8:],'terminal':ff.get('terminal.json',{}),'rehearsal':ff.get('train/rehearsal.json',{}),'eval_launch_present':'text' in ff.get('fresh-public20m-launch.json',{}),'receipt_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))"""
assert text.count(old_print)==1;text=text.replace(old_print,new_print)
ast.parse(text)
sys.argv=[sys.argv[0],tag]
exec(compile(text,str(old),'exec'),{'__file__':__file__,'__name__':'__main__'})
