#!/usr/bin/env python3
"""Exact LR0.015 confirmation repeat using the hardened r1 full-run harness."""
import importlib.util,pathlib
from decimal import Decimal,InvalidOperation
ROOT=pathlib.Path(__file__).resolve().parents[4];HERE=pathlib.Path(__file__).resolve().parent
R1=ROOT/"research/submission-validation/20260924-c595-matrixlr015-trn21/retune_job.py"
spec=importlib.util.spec_from_file_location("matrixlr015_r1_harness",R1);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
r.HERE=HERE;r.LABEL="fs-20260924-c595-matrixlr015-confirm-trn21-r2"
FIRST=Decimal("0.9920722144854677");PRIOR_BEST=Decimal("0.9928513116311253")
def confirmation(serialized):
 try:repeat=Decimal(serialized)
 except (InvalidOperation,TypeError):raise SystemExit("repeat BPB is not decimal")
 if not repeat.is_finite() or repeat<=0:raise SystemExit("repeat BPB invalid")
 mean=(FIRST+repeat)/Decimal(2);confirmed=repeat<PRIOR_BEST and mean<PRIOR_BEST
 return {"prior_best_bpb":str(PRIOR_BEST),"first_lr015_bpb":str(FIRST),"repeat_full20m_bpb":str(repeat),"two_run_arithmetic_mean":str(mean),"repeat_below_prior_best":repeat<PRIOR_BEST,"mean_below_prior_best":mean<PRIOR_BEST,"decision_gate":"CONFIRMED" if confirmed else "NOT_CONFIRMED","replication_required":False,"first_arm_retroactively_relabelled":False}
r.decision_from_serialized=confirmation
if __name__=="__main__":r.main()
