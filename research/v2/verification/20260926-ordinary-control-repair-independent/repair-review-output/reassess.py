from pathlib import Path
import json, hashlib, base64, struct, copy
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'repair-review-output'
# Execute the independently authored prior test without altering frozen evidence.
text=(ROOT/'review-output/review.py').read_text().replace("OUT=ROOT/'review-output'","OUT=ROOT/'repair-review-output'").replace("ROOT/'full-context.diff'","ROOT/'corrected-source.diff'")
(OUT/'independent-rerun.py').write_text(text)
exec(compile(text,str(OUT/'independent-rerun.py'),'exec'),globals())
def check(ok,msg):
    if not ok: raise RuntimeError(msg)
check(report['source_sha256']=='1cc6ee395f88125b9c9d280562e918d002a1e1a10dc88d50501d1fb4a09a551c','SOURCE_HASH')
# Confirm the neutral full source is exactly the corrective diff's added/context source.
lines=(ROOT/'corrective.diff').read_text().splitlines(keepends=True)
inside=False; material=[]
for line in lines:
    if line.startswith('diff --git') and inside: break
    if line.startswith('@@ '): inside=True; continue
    if inside and line[:1] in (' ','+'): material.append(line[1:])
check(''.join(material)==source,'CORRECTIVE_DIFF_NOT_IDENTICAL')
extra={}
def rejects(name,action):
    try: action()
    except Exception as e: extra[name]={'rejected':True,'error':type(e).__name__+': '+str(e)}
    else: raise RuntimeError('FALSE_ACCEPTANCE:'+name)
original=json.loads((ROOT/'review-output/counterexample-receipt.json').read_text())
rejects('retained_original_malformed_receipt',lambda:p.validate_receipt(original,0))
# Fresh forgery with updated summary metadata avoids rejecting merely an old schema.
forged=copy.deepcopy(receipt)
empty=dict(sha256=hashlib.sha256(b'').hexdigest(),nbytes=0,dtype='torch.float32',shape=[256,128],nonzero=0,finite=True,negative_zero=0,nonfinite_count=0,nonfinite=[])
forged['blobs'][empty['sha256']]='';forged['weight']=empty
for entry in forged['routes']:
    for key in ('expected_inputs','before_forward','before_backward','after_backward'):entry[key]['weight']=empty
    entry['expected_output']=empty;entry['output']=dict(exact=True,max_abs=0.0,actual=empty,expected=empty)
rejects('fresh_consistent_malformed_receipt',lambda:p.validate_receipt(forged,0))
(OUT/'fresh-malformed-receipt.json').write_text(json.dumps(forged,sort_keys=True))
for name,raw,dtype,shape in [
 ('empty',b'','torch.float32',(256,128)),('short',bytes(131068),'torch.float32',(256,128)),('long',bytes(131076),'torch.float32',(256,128)),
 ('unaligned_int',bytes(1023),'torch.int32',(256,)),('unsupported_dtype',bytes(4),'torch.float64',(1,)),('null_dtype',bytes(4),None,(1,)),
 ('bool_dim',bytes(4),'torch.float32',(True,)),('float_dim',bytes(4),'torch.float32',(1.0,)),('negative_dim',b'','torch.float32',(-1,)),
 ('zero_dim',b'','torch.float32',(0,)),('string_dim',bytes(4),'torch.float32',('1',)),('no_dims',bytes(4),'torch.float32',()),
 ('wrong_container',bytes(4),'torch.float32','1'),('mutable_raw',bytearray(4),'torch.float32',(1,))]:
    target={}
    rejects(name,lambda:p.blob(raw,dtype,shape,target))
    check(not target,'MUTATED_BEFORE_REJECTION:'+name)
nonfinite_raw=b''.join(struct.pack('<I',[0x7fc00001,0x7f800000,0xff800000][i%3]) for i in range(32768))
bs={}; rec=p.blob(nonfinite_raw,'torch.float32',(256,128),bs)
check(rec['nonfinite_count']==32768 and len(rec['nonfinite'])==16,'NONFINITE_SAMPLE_BOUND')
check(base64.b64decode(bs[rec['sha256']])==nonfinite_raw,'NONFINITE_RAW_RETENTION')
check([x['tag'] for x in rec['nonfinite'][:3]]==['NaN','Infinity','-Infinity'],'NONFINITE_TAGS')
json.dumps(rec,allow_nan=False)
report['repair_negatives']=extra
report['nonfinite_check']={'total':rec['nonfinite_count'],'sample':len(rec['nonfinite']),'raw_sha256':rec['sha256'],'raw_bytes':len(nonfinite_raw)}
report['status']='SCOPED_SOURCE_CPU_PASS'
(OUT/'repair-results.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print('REPAIR_CHECKS\n'+json.dumps(report,indent=2,sort_keys=True))
