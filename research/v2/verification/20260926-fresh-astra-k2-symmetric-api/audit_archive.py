"""Authenticate fresh review execution and raw CPU artifacts, never native acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
checks = []
def ck(name, ok):
    if not ok: raise RuntimeError(name)
    checks.append(name)
def sha(b): return hashlib.sha256(b).hexdigest()
def read(p): return json.loads(p.read_text())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    manifest=read(HERE/'ARCHIVE_MANIFEST.json')
    ck('terminal419files',manifest['terminal'] is True and manifest['count']==len(manifest['files'])==419)
    total=0
    for row in manifest['files']:
        p=HERE/row['path'];b=p.read_bytes();total+=len(b)
        ck('archive '+row['path'],not p.is_symlink() and len(b)==row['bytes'] and sha(b)==row['sha256'])
    ck('22858019 archived bytes',total==manifest['total_bytes']==22858019)
    terminal=read(HERE/'session/terminal.json');pre=read(HERE/'session/preflight.json')
    ck('actual fresh terminal0',terminal['exit_code']==0 and terminal['prior_history'] is False and terminal['repository_mounted'] is False and terminal['model_requested']=='gpt-6-astra' and terminal['elapsed_seconds']<1200)
    ck('isolation preflight passed',pre['exit']==0 and pre['manifest_sha256']==terminal['manifest_sha256']=='d55ae04d708571d010d8eff4b680d8748472032624f74e4c7636aa2470189433')
    packet=read(HERE/'packet/INPUT_MANIFEST.json')
    ck('new factual evidence only',len(packet['included'])==10 and packet['prior_history'] is False and packet['shared_branch'] is False)
    out=HERE/'session/reviewer-output';summary=read(out/'cpu_artifacts/summary.json')
    ck('exact reconstructed kernel',sha((out/'candidate_consumer_nki.py').read_bytes())==summary['source_sha256']=='2c8295a4b1820b04985859389f38470774b7ffe190baaff4d01c86956cb511ae')
    ck('independently authored test archived',sha((out/'independent_cpu_test.py').read_bytes())==summary['test_sha256']=='ff9f597817f843eb3b3ec22ac91c57b67b8da3ef1bfc0bedf4b71824a8335a76')
    ck('test exited0 no stderr',(out/'cpu_test.exitcode.txt').read_text().strip()=='0' and (out/'cpu_test.stderr.txt').read_bytes()==b'')
    ck('scope is CPU only',summary['status']=='PASS_BOUNDED_SOURCE_CPU_ONLY' and summary['native_verdict']=='REFUSE_UNPROVEN' and summary['native_executed'] is False)
    cases=sorted((out/'cpu_artifacts').glob('*/result.json'));counts={'independent':0,'supplied':0};nbytes=adds=0
    ck('48case files',len(cases)==summary['cases']==48)
    for p in cases:
        result=read(p);root=p.parent;counts[result['origin']]+=1
        for name, row in result['files'].items():
            b=(root/name).read_bytes();ck('case artifact '+str(root.name)+'/'+name,len(b)==row['bytes'] and sha(b)==row['sha256'])
        a=(root/'actual.fp32le').read_bytes();e=(root/'expected.fp32le').read_bytes();writers=(root/'writers.u8').read_bytes()
        ck('raw exact one writer '+root.name,a==e and len(a)==131072 and writers==bytes([1])*32768)
        ck('finite raw '+root.name,all((bits[0]&0x7f800000)!=0x7f800000 for bits in struct.iter_unpack('<I',a)))
        ck('reported exact scope '+root.name,result['comparison']=={'exact':True,'finite':True,'max_abs':0.0} and result['symmetric_program_traces'] and result['stable_per_row_token_order'] and result['unused_rows_positive_zero'])
        nbytes+=len(a);adds+=result['effective_fp32_additions']
    ck('case split and totals',counts=={'independent':28,'supplied':20} and nbytes==summary['full_output_bytes_compared']==6291456 and adds==summary['effective_fp32_additions']==1315072)
    negatives=read(out/'cpu_artifacts/negative_controls.json')
    ck('17 negatives detected',len(negatives)==summary['negative_controls_detected']==17 and all(r['detected'] is True for r in negatives))
    report=(out/'REPORT.md').read_text()
    ck('explicit bounded verdict and native refusal',report.startswith('**Verdict: PASS for the bounded source/CPU gate only. REFUSE native exactness, adoption, and any claim that the compiler failure is fixed.**'))
    result={'status':'AUTHENTICATED_FRESH_SOURCE_CPU_PASS_ONLY','checks':len(checks),'check_names':checks,'native_accepted':False,'launch_authorized':False,'cases':len(cases),'raw_output_bytes':nbytes,'elapsed_seconds':terminal['elapsed_seconds'],'completed_utc':terminal['completed_utc']}
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='check_names'}))

if __name__=='__main__':main()
