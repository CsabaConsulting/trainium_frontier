# Scoped independent source-port acceptance

2026-09-26 02:55 UTC. Root read the fresh review, reconstructed-source method,
independent functional-reference test and boundary test in full. The reviewer
received only the full-context diffs, neutral specification, old refusal and
raw gate receipts; no Implementer branch, notes or test implementation.

Report SHA256 c4464dca54f48c430895faeac9d109ce4bdd600c8c8c794049d92cbf88186093.
Candidate e1e2f68629f3b7ff56c19c555cadef5b5e30868147d174ddc77a486dd2a74b43.
This accepts the report's source-port and tiny CPU arithmetic/clock scope ONLY.
The old route-preservation refusal remains preserved; the new specification
explicitly retains the proven ordinary loader, not research's row adapter.

Root replay from /tmp/trainium-confirmed36-fresh-review-giwDge/review-output:

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/poojith-devan/venv/bin/python independent_test.py
    /home/poojith-devan/venv/bin/python static_boundary_test.py

Both exited0. Actual stdout:

    {"losses": [5.542303562164307, 5.546473503112793, 5.543309211730957], "peak_live_tensor_storage_bytes": 28412552, "model_parameters": 1475892, "batches": [1, 2, 3], "context": 256, "result": "PASS"}
    {"clock_cases": 280, "result": "PASS"}
    {"unchanged_ordinary_loader": true, "unchanged_save": true, "identical_research_schedule_helper": true, "actual_boundary_cases": 7, "boundary_clock_calls": "one at fifth boundary only", "launch_hash": "PASS"}

The first root attempt used system python3 and failed before any test with
ModuleNotFoundError: No module named 'torch'. It is not a candidate numerical
failure; using the already-installed declared venv required no installation
or source/test change. Reviewer harness failures remain in archived raw files.

Archived scripts expect their original reconstructed sources and parent patch/
launch inputs. Those exact inputs remain in the neutral directory and the
candidate's committed evidence; this archive is not advertised as standalone.
Do not commit the tiny checkpoint or claim a full113M/native test occurred.

Full CPU/production trajectory evidence, native consistency, candidate cold
startup and full run, fresh public20M and Claude via Poojith remain pending.
No host launch, gate waiver, official score or submission readiness follows.

CPU packet delegation attempts to the old Implementer, new sessions and a
finished Researcher returned 'agent thread limit reached'. No new packet
builder is active. Existing sole Operator and detached host jobs are unaffected.
