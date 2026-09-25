"""Generate one input-only packet and reuse the reviewed isolation operation."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
REPO=Path('/home/poojith-devan/Desktop/Trainium')
BASE='077481d0476cebfd79ddc751f8f63e61c1fbf82f'
CANDIDATE='18b9365aabb63fa85e1720fdefc11177ca8e7853'
PREFIX='research/v2/lanes/k2-resident-consumer/'
def blob(ref,path):return subprocess.check_output(['git','-C',str(REPO),'show',ref+':'+path])
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
    if (HERE/'PREPARED.json').exists():raise RuntimeError('One prepared session only')
    old=blob(BASE,PREFIX+'check_native.py')
    sources={name:blob(CANDIDATE,PREFIX+name) for name in ('check_native.py','consumer_nki.py','reference.py')}
    expected={'check_native.py':'89189024c54735338f2ff69f9f26a6ee025f2b3c48f0fe5fbd67438ba3290850','consumer_nki.py':'57b5b118e86eb9d78e26a011b276904aed9065868b8a12f39169e1257c3a1843','reference.py':'14572563ca970754ca0dceb309c86d141916a91ceef26acc7a7eccd254d1953f'}
    if {k:sha(v) for k,v in sources.items()}!=expected or sha(old)!='17e2d2ad417407c12327e1446aed857f273d445946839357102e93dea46d4435':raise RuntimeError('Source pin mismatch')
    if any(blob(BASE,PREFIX+n)!=sources[n] for n in ('consumer_nki.py','reference.py')):raise RuntimeError('Unexpected arithmetic change')
    packet=Path(tempfile.mkdtemp(prefix='trainium-fresh-k2-reporting-packet-20260926.'))
    run=Path(tempfile.mkdtemp(prefix='trainium-fresh-k2-reporting-review-20260926.'))
    (packet/'raw').mkdir();(run/'output').mkdir()
    def diff(a,b,oldname,newname):return ''.join(difflib.unified_diff(a.decode().splitlines(True),b.decode().splitlines(True),fromfile=oldname,tofile=newname,n=100000))
    (packet/'checker.diff').write_text(diff(old,sources['check_native.py'],'baseline/check_native.py','candidate/check_native.py'))
    for name in ('consumer_nki.py','reference.py'):(packet/(name+'.diff')).write_text(diff(b'',sources[name],'/dev/null','candidate/'+name))
    (packet/'lane-spec.md').write_bytes((HERE/'lane-spec.md').read_bytes())
    for name in ('checks-normal.json','checks-optimized.json'):(packet/'raw'/name).write_bytes(blob(BASE,PREFIX+name))
    raw_path='research/v2/verification/20260926-fresh-astra-k2-consumer/session/reviewer-output/checker_failure_results.json'
    prior=blob('90db28e0',raw_path)
    values={'format':'IEEE754 binary32 bit pattern as big-endian hexadecimal spelling; each actual/expected tensor has shape [1,1]','cases':[{k:row[k] for k in ('name','actual_hex','expected_hex')} for row in json.loads(prior)['cases']]}
    (packet/'raw/scalar-inputs.json').write_text(json.dumps(values,indent=2,sort_keys=True)+'\n')
    included=[{'path':str(p.relative_to(packet)),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in sorted(packet.rglob('*')) if p.is_file()]
    manifest={'baseline_commit':BASE,'candidate_commit':CANDIDATE,'baseline_checker_sha256':sha(old),'candidate_source_sha256':expected,'included':included,'prior_history':False,'shared_branch':False,'excluded':['all prior review conclusions/reports','implementer tests/notes/README/proposal','repository/branch','model checkpoints','native results'],'raw_baseline_receipts_commit':BASE,'scalar_inputs_projection':{'source_commit':'90db28e0','source_file_sha256':sha(prior),'kept_fields':['name','actual_hex','expected_hex'],'prior_conclusions_included':False}}
    encoded=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode();(packet/'INPUT_MANIFEST.json').write_bytes(encoded);pin=sha(encoded)
    template=(HERE.parent/'20260926-fresh-astra-architecture-recursive/session/operate.py').read_text()
    template=template.replace('/tmp/trainium-fresh-architecture-recursive-packet-20260926.dsxaqcz_',str(packet)).replace('75a0dbd970b2397c8ad7e75ad564cf1deb28f1a1eed5e4a51e0a288d952895dd',pin)
    (run/'operate.py').write_text(template)
    result={'packet':str(packet),'run':str(run),'input_manifest_sha256':pin,'candidate_commit':CANDIDATE,'candidate_checker_sha256':expected['check_native.py'],'fresh_session_started':False,'model_requested':'gpt-6-astra','operation_changes':['packet path','manifest pin']}
    (HERE/'PREPARED.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
