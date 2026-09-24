# External A0 step-boundary observer

`research/a0_boundary_observer.py` observes one already authorized legacy-guarded run. It does
not launch or stop training, alter the guard, or write in the pinned A0 checkout. The Operator
starts it before the first optimizer boundary and stores its single JSON stdout object at a
fresh path **outside** the pinned checkout. It exits 0 only with a complete timing receipt;
exit 2 means timing or identity proof is degraded. A timeout is also exit 2. Do not treat this
receipt as training correctness, budget, checkpoint, or BPB proof by itself.

For the pending trn22 warm c5c A0 request, use the exact label and absolute paths from the
finalized request. The observer's four path arguments are the existing guard script and the
three guard/log artifacts under that request's remote root. In particular, the boundary file
has a `.steps.jsonl` name but contains one fsynced decimal integer line per completed step,
beginning with `0`; it is not JSONL. Start monitoring while the guard and its child are alive,
before the first boundary is written. The guard process file may appear after the observer
starts. The observer refuses to certify a boundary that was already present at its first read.

Example argument shape (replace only the exact reviewed root/label and receipt location):

```bash
python /path/to/separate-observer-checkout/research/a0_boundary_observer.py \
  --label fs-20260925-v2-c5c-a0-control-trn22-r1 \
  --guard-script /home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r1/research/fresh_start_guard.py \
  --process-json /home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r1/research/run-logs/fs-20260925-v2-c5c-a0-control-trn22-r1.process.json \
  --boundary-file /home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r1/research/run-logs/fs-20260925-v2-c5c-a0-control-trn22-r1.steps.jsonl \
  --postflight-json /home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn22_r1/research/run-logs/fs-20260925-v2-c5c-a0-control-trn22-r1.postflight.json \
  --poll-seconds 0.02 --timeout-seconds 4800 \
  > /path/to/separate-fresh-receipt/fs-20260925-v2-c5c-a0-control-trn22-r1.timing.json
```

Create the destination directory beforehand; use a new destination name and shell noclobber so
an earlier receipt cannot be overwritten. The receipt includes the guard/child PID and process
start ticks, file identities and hashes, UTC and monotonic observation timestamps, the
fifth-boundary startup estimate, terminal observation, postflight, and invalid reasons. A line
seen in the same poll as another line, a duplicate or skipped integer, a rewrite/truncation,
an incomplete final line, a wrong guard process, or a mismatched step-log path makes
`scored_timing_proof` false. The timestamps are the observer's first completed-read times;
they are polling observations, not a claim of the organizer's private clock readings.

Local fake-guard check: `python -m unittest research.v2.test_a0_boundary_observer -v`.
It exercises a clean five-boundary sequence and the late-start, duplicate, multi-boundary,
truncation, partial-line, and wrong-guard failure paths without Trainium access.
