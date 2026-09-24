#!/usr/bin/env python3
import importlib.util,json,pathlib
from decimal import Decimal
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("confirm",HERE/"confirmation_job.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert m.r.TRAIN_SHA=="116188cfb1b9b52ee9bb7222a683c899a32cb26ec7c1e13f1f20b2496229534d"
assert m.r.LABEL=="fs-20260924-c595-matrixlr015-confirm-trn21-r2" and m.r.HERE==HERE
prior=Decimal("0.9928513116311253");first=Decimal("0.9920722144854677")
assert m.confirmation(str(prior))["decision_gate"]=="NOT_CONFIRMED"
repeat_for_exact_mean=prior*2-first
assert m.confirmation(str(repeat_for_exact_mean))["decision_gate"]=="NOT_CONFIRMED"
assert m.confirmation(str(prior-Decimal("0.0000000000000001")))["decision_gate"]=="CONFIRMED"
assert m.confirmation("0.9920")["first_arm_retroactively_relabelled"] is False
proposal=json.loads((HERE/"EXPERIMENT_PROPOSAL.json").read_text());assert proposal["first_run_full20m_bpb"]=="0.9920722144854677" and proposal["prior_best_bpb"]=="0.9928513116311253"
assert proposal["launch"]["outer_timeout_seconds"]==4200 and proposal["launch"]["smoke"] is False
print("LR0.015 confirmation r2 static/Decimal gates: PASS")
