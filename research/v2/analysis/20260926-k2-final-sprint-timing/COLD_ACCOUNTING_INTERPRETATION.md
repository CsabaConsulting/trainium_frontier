# Report cold-start cost alongside K2 speed; not a new gate

The precommitted native screen remains whole-window rate ratio>=1.03,
45 valid finite steps, exact source/runtime and official accounting. Do not
replace that rule or the unchanged final-public20M selector with this estimate.

For interpretation, also report measured candidate cold startup S and charge
max(0,S-900). Under a constant observed steady rate assumption, a rough ratio
of tokens available in an official1800s run relative to an uncharged control is:

rate_ratio * max(0,1800-max(0,S-900))/1800.

This is a projection, not measured full-run tokens or BPB. The existing genuine
e1e2 cold reference is859.473523823s (no excess charge); the immediately
preceding research control startup793.8651350670261s is not cold proof.
An observed3% speedup by itself only offsets cold startup up to approximately
952.427185s;5% up to985.714286s;10% up to1063.636364s. A faster steady step
therefore does not automatically improve the official cold-run opportunity.

Do not infer a BPB loss or gain from tokens alone: repeated epochs, schedule,
learning efficiency and table behavior can differ. Full fixed-budget BPB and
the eventual actual cold replay remain decisive. Record first-call compilation
plus execution honestly; do not call it pure compiler time. Keep cache provenance,
startup measurement, charged time, actual steps and final BPB separate.
