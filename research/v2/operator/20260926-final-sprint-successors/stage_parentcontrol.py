"""Existing exact-byte transport, rebound only to released ON/parent comparison."""
import ast
import hashlib
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
raw=(HERE/'stage_next.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='62bc9f3437057034537d7f50b397b7bb8e2ba8e303b0feb892cf9891d6178318'
assert sys.argv[1]=='trn22-on-parentcontrol'
assert sys.argv[2:] in (['--local-check'],['--execute'])
text=raw.decode()
def replace_once(old,new):
    global text
    assert text.count(old)==1,old
    text=text.replace(old,new)
replace_once("choices={", "choices={\n 'trn22-on-parentcontrol':('20260926-k2-sprint-timing/trn22-on-parentcontrol-prepared.json','d3ab702d0839043ad6267d01a27a0aeaa02f81236a451e5155720f0255c5614e'),")
replace_once("assignment=REPO/'research/v2/assignments/20260926-sprint-next-dispatch.md';auth=assignment.read_bytes();assert b'First qualified free host' in auth and b'LR.021' in auth",
             "assignment=REPO/'research/v2/assignments/20260926-k2-use-live-parent-control.md';auth=assignment.read_bytes();assert b'fs-20260926-k2-sprint-on-parentcontrol-trn22-r1' in auth and b'14,495,514,624' in auth and b'35*131072/(t44-t9)' in auth")
replace_once("if choice in ('trn22-off','trn22-on'):","if choice in ('trn22-off','trn22-on','trn22-on-parentcontrol'):")
replace_once("'basis':'Unchanged prepared per-job13.5GiB initial/10.5GiB runtime floor; pair admission additional and explicit.'",
             "'basis':'Unchanged per-job13.5GiB initial/10.5GiB runtime floor; already allocated parent retained, extraOFF45 cancelled prospectively.'")
replace_once("predecessor=HERE/('trn22-off-completed/results-bundle.json' if choice=='trn22-on' else host+'-p0-completed/results-bundle.json')",
             "predecessor=HERE/'trn22-e1e2-control-completed/results-bundle.json'")
# These are the Supervisor's existing comparison conditions, not an added
# scientific test. The future Operator records actual receipt-bound acceptance
# after inspecting the unchanged parent result, timing, graphs and environment.
# Missing actual evidence cannot be made PASS by this staging tool.
needle="attempt=HERE/(choice+'-attempt.json');"
gate="""acceptance=json.loads((HERE/'trn22-parent-control-timing-acceptance.json').read_text())
assert acceptance['completed_bundle_sha256']==sha(predecessor.read_bytes()),'PARENT_RECEIPT_DRIFT'
assert acceptance['control_label']=='fs-20260926-sprint-e1e2-control-trn22-r1'
assert acceptance['control_source_sha256']=='e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43'
assert acceptance['candidate_source_sha256']==b['source_sha256']
assert acceptance['precommit_commit']=='d3da2359'
assert acceptance['primary_statistic']=='35*131072/(t44-t9)'
for condition in ('same_host_runtime_science','first45_sequential_finite','t9_t44_trustworthy','window_no_recompile_or_stall','parent_train_eval_complete_and_budget_pass','explicit_postflight_quiescent'):
 assert acceptance['conditions'][condition] is True,'PARENT_COMPARISON_REFUSED:'+condition
assert acceptance['t44']>acceptance['t9']
assert acceptance['control_tokens_per_second']==35*131072/(acceptance['t44']-acceptance['t9'])
"""
replace_once(needle,gate+needle)
ast.parse(text)
exec(compile(text,str(HERE/'stage_next.py'),'exec'),{'__file__':str(HERE/'stage_next.py'),'__name__':'__main__'})
