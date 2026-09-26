"""Mechanical host/predecessor rebind of the already-executed control transport."""
import ast
import hashlib
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
raw=(HERE/'stage_next.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='62bc9f3437057034537d7f50b397b7bb8e2ba8e303b0feb892cf9891d6178318'
assert sys.argv[1]=='trn21-e1e2-control'
assert sys.argv[2:] in (['--local-check'],['--execute'])
text=raw.decode()
assert text.count('trn22-e1e2-control')==3
text=text.replace('trn22-e1e2-control','trn21-e1e2-control')
def replace_once(old,new):
    global text
    assert text.count(old)==1,old
    text=text.replace(old,new)
replace_once("'20260926-e1e2-prospective-controls/trn22-prepared.json','cf88de2158cff9ed7976630005147adbfd995baef433e977d7889933066fcaf6'",
             "'20260926-e1e2-prospective-controls/trn21-prepared.json','1f080e4d96c2677844c1e53ff94d03f9c89c222e96d8a3a5ae51df6ab07a0743'")
replace_once("predecessor=HERE/('trn22-off-completed/results-bundle.json' if choice=='trn22-on' else host+'-p0-completed/results-bundle.json')",
             "predecessor=HERE/'trn21-batch-lr14-completed/results-bundle.json'")
replace_once("assert sha(files['package/train.py'])==b['source_sha256']",
             """assert sha(files['package/train.py'])==b['source_sha256']
assert b['host']=='trn21' and b['label']=='fs-20260926-sprint-e1e2-control-trn21-r1'
assert sha(files['EXPERIMENT_PROPOSAL.json'])=='e635132fe0f9ba407ca789fabfc5011c77accab8da5d2bacacae5673feeb4695'
assert files['EXPERIMENT_PROPOSAL.json']==(REPO/'research/v2/proposals/20260926-e1e2-prospective-controls/EXPERIMENT_PROPOSAL.json').read_bytes()
""")
ast.parse(text)
exec(compile(text,str(HERE/'stage_next.py'),'exec'),{'__file__':str(HERE/'stage_next.py'),'__name__':'__main__'})
