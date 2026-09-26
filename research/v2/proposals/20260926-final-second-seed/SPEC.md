# Prospective final-stack second-seed confirmation

Preparation only; no host release or queue preemption. Current final candidate
is confirmed batch262144/LR.021/ordinary.75/momentum300. Rebind this plan to
the actually selected best stack before launch if a new arm wins.

Use seed43 chosen now, not selected after looking at its BPB. Same official
1800-second accounting and fresh public20M/context1024/B1. No data/model/eval
change other than the chosen stack and seed. Source and argv must be frozen
in a completed per-host proposal before launch; labels must be unique.

For the current fallback:

- Candidate research8c6 + batch262144/LR.021 --seed43, OR its exact clean port
  cd0c with unchanged accepted launch plus --seed43 for this research run only.
- To assess cross-seed gain, pair against accepted e1e2 at seed43 on the SAME
  host. Existing seed43 runs of older6-layer architectures are not controls.
- Keep seed42 as submission seed and compare each seed to its matched control.
  Do not select the better seed or advertise one favorable replicate as a mean.
- Final submission-cold replay uses the final clean source and exact accepted
  command, seed42. A seed43 warm run cannot substitute for that cold replay.

Candidate43 alone measures variation, not a new matched causal improvement.
If matched control43 is unavailable, state that limitation rather than invent
a delta. Existing same-host seed42 comparisons and unchanged decide.py remain.
Perform this after prioritized warmdown/momentum/LR work and before final freeze;
do not stop running jobs. Final package remains due Monday evening.
