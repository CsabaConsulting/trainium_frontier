#!/usr/bin/env python3
"""Unique r2 entrypoint for the retained LR0.025 evaluator-only recovery."""
import importlib.util,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[3]
R1=ROOT/"research/submission-validation/20260924-c595-matrixlr025-eval-recovery-trn21/eval_recovery.py"
spec=importlib.util.spec_from_file_location("lr_eval_recovery_r1",R1);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
r.m.LABEL="fs-20260924-c595-matrixlr025-eval-recovery-trn21-r2"
r.m.OUTPUT=ROOT/"out"/r.m.LABEL
if __name__=="__main__":r.m.main()
