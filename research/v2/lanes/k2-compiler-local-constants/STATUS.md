# K2 local-constant scope fix — delivered

Source662c968a, full source delta in SOURCE.diff, details in SPEC.md.
Both-dtype CPU exactness/OFF3/ON3 pass with byte-identical existing interpreter.
Real exact-version NKI parser verifies exact production-shape candidate MLIR;
parser-result.json records PASS, empty stderr, no closure cells, no native call.
No independent approval or native/timing result. Parent a318 remains frozen and
its consumed-retry native variant stays parked. Root's fresh review/new proposal
and sole Operator gates remain pending.
