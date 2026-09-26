"""Existing first-live observer plus bounded static runtime-version facts."""
import ast
import hashlib
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
raw=(HERE/'observe_bound.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='695c1e027e375eca0a60088b18d230bc82d56b118ff202c31a2ab6b755566209'
assert sys.argv[1:]==['trn22-on-parentcontrol']
text=raw.decode()
extra=r'''
runtime={'scope':'static metadata and interpreter version only; not installed SDK byte-identity or native import attestation','packages':[]}
site=Path('/home/ubuntu/workspace/native_venv/lib/python3.12/site-packages')
for dirname in ('torch-2.11.0+cpu.dist-info','torch_neuronx-2.11.3.0.0+unknown.dev.dist-info','neuronx_cc-2.26.6360.0+6f180f47.dist-info','nki-0.5.0+28631259367.ga768afa6.dist-info','neuron_agentic_development-1.0.dist-info'):
 p=site/dirname/'METADATA';item={'path':str(p)}
 try:
  fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
  try:
   a=os.fstat(fd)
   if not stat.S_ISREG(a.st_mode) or a.st_size>65536:raise ValueError('METADATA_TYPE_OR_SIZE')
   rawmeta=os.read(fd,65537);z=os.fstat(fd)
  finally:os.close(fd)
  item.update(bytes=len(rawmeta),sha256=hashlib.sha256(rawmeta).hexdigest(),stable=(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns),device=a.st_dev,inode=a.st_ino)
  for line in rawmeta.decode().splitlines():
   if line.startswith(('Name: ','Version: ')):
    k,vv=line.split(': ',1);item[k.lower()]=vv
 except (OSError,ValueError,UnicodeError) as exc:item['observation_error']=str(exc)
 runtime['packages'].append(item)
try:
 rr=subprocess.run(['/home/ubuntu/workspace/native_venv/bin/python','-I','-B','-S','--version'],capture_output=True,text=True,timeout=3)
 runtime['interpreter']={'path':'/home/ubuntu/workspace/native_venv/bin/python','resolved':str(Path('/home/ubuntu/workspace/native_venv/bin/python').resolve()),'returncode':rr.returncode,'stdout':rr.stdout,'stderr':rr.stderr}
except (OSError,subprocess.TimeoutExpired) as exc:runtime['interpreter_observation_error']=str(exc)
'''
needle="wire=('P='+repr({'binding':b})+'\\n'+body).encode()"
replacement="body=body.replace(\"v=os.statvfs('/')\", "+repr(extra)+"+\"\\nv=os.statvfs('/')\")\nbody=body.replace(\"'read_only':True\",\"'read_only':True,'runtime_version_observation':runtime\")\n"+needle
assert text.count(needle)==1
text=text.replace(needle,replacement)
ast.parse(extra);ast.parse(text)
exec(compile(text,str(HERE/'observe_bound.py'),'exec'),{'__file__':str(HERE/'observe_bound.py'),'__name__':'__main__'})
