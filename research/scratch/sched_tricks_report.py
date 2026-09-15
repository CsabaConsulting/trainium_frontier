"""Decompose the plan-§11.1 schedule-tricks 900s block into throughput vs per-step terms.

Ten arms bracketed by two same-commit controls is too much arithmetic to do by hand without a
transcription error (this notebook has already lost four numbers that way once), so the decomposition
lives in code and reads the committed log rather than a pasted table.

For each arm, against the MEAN of the two bracketing controls:

    delta            = arm bpb - control mean bpb
    throughput term  = -0.077 * ln(arm tokens / control mean tokens)
                       i.e. what the step-count difference alone buys, at the measured
                       below-the-data-wall slope of -0.077 bpb per ln(steps)
    per-step residual = delta - throughput term
                       i.e. what the change did to the quality of a step, which is the only
                       part that transfers to a different budget

These three items are pure optimizer-schedule changes with no new ops, so the PREDICTION is that
the throughput term is exactly zero. research/scratch/sched_tricks_smoke.sh measured dt directly
(0.841 s in the control, 0.8406-0.8425 across all eight arms) and this is the independent check on
the same claim from the step counts.

The control PAIR's own spread is the noise band this block is read against -- not a cross-day
figure, and not the +-0.001 rule of thumb, which is a floor rather than this block's actual band.

Run: python research/scratch/sched_tricks_report.py [experiment-log.jsonl]
"""
import json
import math
import os
import sys

SLOPE = -0.077  # bpb per ln(tokens), measured below the data wall
LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment-log.jsonl")
PREFIX = "st-"

rows = []
with open(LOG) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("label", "").startswith(PREFIX):
            rows.append(r)
if not rows:
    sys.exit(f"no {PREFIX}* rows in {LOG}")

# Keep only the LAST occurrence of each label, so a re-run supersedes the row it replaces.
by_label = {}
for r in rows:
    by_label[r["label"]] = r
rows = list(by_label.values())

ctrls = [r for r in rows if "ctrl" in r["label"]]
arms = [r for r in rows if "ctrl" not in r["label"]]
if not ctrls:
    sys.exit("no control arm found")


def tokens(r):
    return r.get("total_tokens_M") or r.get("num_steps")


def show(key, label):
    have = [r for r in ctrls if r.get(key) is not None]
    if not have:
        print(f"\n=== {label}: no control has {key}, skipping ===")
        return
    c_bpb = sum(r[key] for r in have) / len(have)
    c_tok = sum(tokens(r) for r in have) / len(have)
    spread = max(r[key] for r in have) - min(r[key] for r in have)
    print(f"\n=== {label} ===")
    print(f"control mean {c_bpb:.6f} over {len(have)} arm(s), tokens {c_tok:.1f}M, "
          f"pair spread {spread:+.6f}  <- this block's noise band")
    print(f"{'arm':<22} {'steps':>6} {'tokM':>8} {'bpb':>10} {'delta':>9} "
          f"{'thruput':>9} {'per-step':>9}  verdict")
    for r in sorted(ctrls + arms, key=lambda r: r["label"]):
        if r.get(key) is None:
            print(f"{r['label']:<22} {'--':>6} {'--':>8} {'MISSING (crash?)':>10}")
            continue
        d = r[key] - c_bpb
        thr = SLOPE * math.log(tokens(r) / c_tok)
        res = d - thr
        if abs(res) < 0.001:
            v = "noise"
        elif abs(res) < 0.003:
            v = "needs a paired repeat"
        else:
            v = "real" + (" WIN" if res < 0 else " LOSS")
        if "ctrl" in r["label"]:
            v = "(control)"
        print(f"{r['label']:<22} {r.get('num_steps', 0):>6} {tokens(r):>8.1f} "
              f"{r[key]:>10.6f} {d:>+9.6f} {thr:>+9.6f} {res:>+9.6f}  {v}")


show("public_val_bpb", "eval_seq_len 2048 (the scored length)")
show("public_val_bpb_alt", "eval_seq_len 1024 (the cross-check length)")
print("\nA verdict must reproduce at BOTH lengths to within 0.0003 before it is a verdict.")
