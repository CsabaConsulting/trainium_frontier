"""Create frozen diff/spec/raw-only packet; reuse exact reviewed isolation operation."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium-architecture-recursive-stack-20260926')
COMMIT='d1ceed83f0cc38605fc262eecf3784e0d9d50fd0'
BASE='b4d83b5db66971f3c348837f04f53cba65aa6d33'
SCHEDULE='a4aa4351789115a2d0aa947d9ca00ff689f9bd0b'
SOURCE='workflow/proposals/research-v2/train.py'
def blob(ref,path):return subprocess.check_output(['git','-C',str(REPO),'show',ref+':'+path])
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
 if (HERE/'PREPARED.json').exists():raise RuntimeError('Do not create a duplicate review')
 sources={name:blob(ref,SOURCE) for name,ref in [('baseline',BASE),('candidate',COMMIT),('schedule_reference',SCHEDULE)]}
 hashes={k:sha(v) for k,v in sources.items()}
 expected={'baseline':'3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c','candidate':'75b944f7caf7f16a8392e720c135d7a6436f4c1ef980a6bb0c9ac15bafc9f70f','schedule_reference':'6ef7d09017de0045f1bcf854571ee3f9aeaffc1080c003c1806f7e20394ad3b3'}
 if hashes!=expected:raise RuntimeError('Frozen source mismatch')
 packet=Path(tempfile.mkdtemp(prefix='trainium-fresh-architecture-recursive-packet-20260926.'))
 run=Path(tempfile.mkdtemp(prefix='trainium-fresh-architecture-recursive-review-20260926.'))
 (packet/'raw').mkdir();(run/'output').mkdir()
 diff=''.join(difflib.unified_diff(sources['baseline'].decode().splitlines(True),sources['candidate'].decode().splitlines(True),fromfile='baseline/train.py',tofile='candidate/train.py',n=100000))
 (packet/'baseline-to-candidate.diff').write_text(diff)
 schedule=''.join(difflib.unified_diff([],sources['schedule_reference'].decode().splitlines(True),fromfile='/dev/null',tofile='schedule-reference/train.py',n=100000))
 (packet/'schedule-reference.diff').write_text(schedule)
 (packet/'lane-spec.md').write_bytes((HERE/'lane-spec.md').read_bytes())
 for arm in ('reference','default','off','on'):
  name='tiny-g2-'+arm+'.json'
  (packet/'raw'/name).write_bytes(blob(COMMIT,'research/v2/lanes/architecture-recursive-stack/'+name))
 included=[{'path':str(p.relative_to(packet)),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in sorted(packet.rglob('*')) if p.is_file()]
 manifest={'candidate_commit':'fe0969fd311d654e143dc6113614416c8970a424','raw_commit':COMMIT,'baseline_commit':BASE,'schedule_reference_commit':SCHEDULE,'source_sha256':hashes,'included':included,'prior_history':False,'shared_branch':False,'excluded':['implementer notes/tests/conclusions','prior reviews','repository','checkpoint files','full-size/native claims']}
 encoded=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode();(packet/'INPUT_MANIFEST.json').write_bytes(encoded);pin=sha(encoded)
 template=(HERE.parent/'20260926-fresh-astra-architecture-clean/session/operate.py').read_text()
 template=template.replace('/tmp/trainium-fresh-architecture-clean-packet-20260926.q9afftww',str(packet)).replace('802a246a3fc0d816e9801c5b73fccb8dc0cb1b8b5e71214246eb5fff33907d92',pin)
 (run/'operate.py').write_text(template)
 result={'packet':str(packet),'run':str(run),'input_manifest_sha256':pin,'candidate_sha256':hashes['candidate'],'candidate_commit':'fe0969fd311d654e143dc6113614416c8970a424','fresh_session_started':False,'operation_changes':['packet path','manifest pin'],'model_requested':'gpt-6-astra'}
 (HERE/'PREPARED.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
