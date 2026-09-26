# PASS — disabled one-child packet mechanics only

Frozen driver SHA256: f15220ab890b4219c6eaa26b13f3e73fe6b6de1f169a413b77a420e873ba8f2a.
Frozen disabled plan SHA256: 3aee91719b04f95896639afb674240a23ad10b988a6bf79c11797f5f88a4ba4c.
Helper remains 1b3714e050e473b176a3e1f1fbd59387579867f3d06d14ecf71248a6d8a1714c, already independently reviewed.

Reviewed only supplied packet files and previous bounded input bundle. Executed own 57 stdlib checks in normal and optimized Python, both exit 0. No host, network, SDK, real model, checkpoint, or training actions. All packet inputs and own test bytes hashed in output receipts.

Verified all sixteen exact old readonly identities against original raw bundle request/checkpoint metadata, plus frozen support/helper/INPUTS hashes. Command has one synchronous checkpoint-helper child with 600-second TERM timeout and 60-second KILL grace. It does not invoke G2 or training. Separate new output root/private tmp and -B/PYTHONDONTWRITEBYTECODE avoid intended writes to old source/checkpoint tree. The helper's four checkpoint pins and separately labeled three-arm BF16 comparison are unchanged.

Verified original lock device66305/inode832589 checks, inherited FD pass-through, unchanged authenticated outer flock command, and three fresh idle-observation call sites (initial, launch, post-child). Driver waits synchronously without an explicit unlock/relock. Guard authenticates exact flock parent and exclusive inherited FD, checks ps and neuron-ls, and fails on busy/invalid observations. Own fake-observation tests reject compiler jobs, device holders, malformed ps, missing driver and command failure. Own lock tests reject replacement FD/path identity. Own tests reject missing/symlink/size/hash input discrepancies. Resource checks pass exact 64GiB RAM and10.5GiB disk floors and reject one-unit-below floors, wrong host/parent and STOP presence.

Result-envelope checks tested using explicitly synthetic data: status, arm completeness, scope, preserved original failure, repeated outputs, NaN causality, and invalid mask proof all fail closed. These are envelope checks, not independent tensor validation or real result evidence. The pinned helper supplies actual tensor validation and any returned result still requires an independent outcome audit.

Critical scope: timeout exit and driver post-guard do NOT prove cleanup. The unchanged ps guard does not enumerate arbitrary helper descendants, and early exceptions may bypass the driver's post-guard. Therefore this PASS includes the plan's explicit external Operator obligation: on EVERY outcome independently enumerate all actually observed dispatch/flock/driver/timeout/helper PIDs and descendants/compiler, retain PID/starttime/PPID/PGID/SID and raw ps/neuron-ls/device-holder/original-lock evidence. Alive, missing, or ambiguous evidence means HOLD and no next job. This is not optional and may not be replaced by timeout status or terminal.json. No cleanup/deletion/second retry is authorized.

The plan remains stage_authorized=false, launch_authorized=false, native_launch_authorized=false and host_reserved=false. Driver does not itself consume those authority flags; safety depends on the separate explicit Supervisor release and sole Operator assignment specified in the packet. This review does not grant that release. Original r2 FAIL remains. No native/cold/full training, model change, quality/adoption or full113M checkpoint success is implied.

Commands executed:

- /usr/bin/python -B /tmp/trainium-mask-helper-review-jUp3zA/fresh-review/packet_tests.py
- /usr/bin/python -B -O /tmp/trainium-mask-helper-review-jUp3zA/fresh-review/packet_tests.py

Own output hashes:

- packet_tests.py: 62ff01599858385a7bc13d398048cb6d90d02db5129a0885c4261906de8e3b8e
- packet-normal.json: 26ab7a556895139cbb19245b2ff966f26c68d2f76fcad80e743ebc80eb4dd0e5
- packet-optimized.json: 34f5eaf9769bb12b4f317b6e6ad02350e5e8070bf81191d41212f843963b7055
