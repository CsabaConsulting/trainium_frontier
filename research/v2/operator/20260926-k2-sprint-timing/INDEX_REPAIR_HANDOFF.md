# Exact K2 repair, next after P3

Local-only packet: `trn22-indexrepair-prepared.json`, SHA
`e09012030dd7a98b82e160998665a1bee1bb66deea467994f86251e1e0e162e8`.
Exact source `a3181d0dd183ca04efdbf9dcf5dcd7e3b556a892a3c9a30a2761e73d37c2aa71`;
driver `77be68e6d4167180385063d7e3df10e5ceb868b15d6b0bedba61a938b38a31fa`.

`prepare_index_repair.py` verifies literal driver equivalence under label/root,
source pin, and derived provenance pin replacement; source is copied unchanged
from the independently reviewed canonical file. Scientific argv, launch text,
guardian, idle check, dispatcher, runtime budget and six-cache policy are unchanged.
Completed repair proposal SHA `0b2f56940734765e9a85e46b250fd3d1ed8a989fa701c5ef7bfa0d908bcb7e5c`;
independent review `b689fb6ed0dd86b73bb945da2fb6bb31a3e8a524b15dc3c984624a5dea08c6dc`.

Proven transport adapter `../20260926-final-sprint-successors/stage_indexrepair.py`
SHA `0c3d512f5f49712e7a4607388687f75f7e8b9e13f70caa44b759447af9a99368`
passes `trn22-indexrepair --local-check`. It requires the completed P3 terminal
bundle and explicit original-lock/device/descendant quiescence before dispatch.
Original parent acceptance remains untouched and bound to its historical639 source;
the new reviewed repair source is explicitly bound separately. Original failure
bundle `52c017298afa972fb5731e479f1c3fa4eb72ee5ae7ed8eb39c067c41de4976de`
also remains an exact prerequisite, never rewritten as a pass.

Run only after P3 natural terminal/retrieval and fresh original832589 lock checks:

```
python -B research/v2/operator/20260926-final-sprint-successors/stage_indexrepair.py trn22-indexrepair --execute
```

No remote staging or retry execution has occurred during this preparation.
This is ON45 only, with no extra OFF, evaluation, automatic full or successor.
