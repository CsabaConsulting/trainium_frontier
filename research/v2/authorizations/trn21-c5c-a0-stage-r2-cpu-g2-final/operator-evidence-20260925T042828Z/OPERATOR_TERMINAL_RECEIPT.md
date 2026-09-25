# trn21 CPU-only G2 Operator terminal receipt

UTC date: 2026-09-25. This is the one assigned CPU-only, four-rank, three-step synthetic G2 attempt. It is **raw evidence pending independent numeric review**, not a numeric verdict and not authority for native training, evaluation, transfer, cleanup, or a same-label retry.

## Authority and local gate

- Clean isolated Operator branch: `evidence/20260925-trn21-c5c-a0-cpu-g2-operator-0425`, started from `41f5215c4f786ed8712787ee3b41176b91227d2e`. The canonical controller checkout was consulted for its breaker state; the reviewed wrapper was invoked from the isolated checkout.
- Exact scope: pinned trn21 endpoint `ubuntu@ec2-16-26-246-98.ap-southeast-4.compute.amazonaws.com`, source checkout `/home/ubuntu/trainium_freshstart_20260925_v2_c5c_a0_control_trn21_stage_r2`, new scratch root `/home/ubuntu/trainium-v2-g2-20260925-trn21-stage-r2`, and unique label `fs-20260925-g2-a0-c5c-exactenv-trn21-stage-r2`. The synthetic helper options were `--steps 3 --nproc 4 --timeout 900 --keep`; no real-data evaluation was requested.
- Reviewed wrapper: `research/v2/authorizations/trn21-c5c-a0-stage-r2-cpu-g2-final/cpu_g2_invocation_final.py`, SHA-256 `8518e47d259a7ca9ecd592328093b4332a4a79f1ca96548ef8fe74bf22cacf6a`. Remote adapter SHA-256 `966e1c67d909e8be0a63cdb5f18e3801e73cd199c3fda63890916131ca48355a`.
- Actual request SHA-256 `7cd7b7aaf34c58da76f1072dbe3e1609c9b6838ca6a975de6fa5c5a7ab85be15`; binding `2463f37e7ae97f5a37632fa3d20fa2fa5b7b5f5df2e07749eda6c9a7af72dcbf`; independent final PASS `c3e2be88d6a813c9aae95387ecb98f8f0b07771b82dfbaa7c4de618e43f76e55`; Supervisor assignment `24fbce895bd57340e91332e19a3c6705c76e209cf1b7e46341b249f41027b55a`. Both actual fixed-path authority JSON files were regular, UID 1000, link count 1, mode `0644`, and reauthenticated byte-for-byte before the attempt.
- Canonical controller state SHA-256 `cf51b6de1d9fcdb58633ec7dcd1b29deba3190fa0ace1385333c174191c8ffd5`; local STOP and breaker gate clear. The fixed attempt marker was absent before invocation. No source edit was made.
- The exact reviewed invocation was the imported `operator_once()` once. Its read-only audit argv SHA-256 was `df2d4bbeb40bd37d1fe198fab17e0816df3828588d4772d9107449c81bdd388b`; its one-attempt run argv SHA-256 was `ba86c93554e7249ba0eca5d1c4eb9a8091da79a4f213069b7106a7c0f6b86e14`; the stdin payload SHA-256 was `adcb345f603879e506e6dcbee29729edaabb60d27222a3fecec7ae000f4d97f8`. The reviewed transport used strict host-key checking. The archived journal redacts only the SSH identity-file path; it retains the original argv hashes and all raw remote stdout/stderr.

From the isolated checkout, the single local invocation was:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -c 'import importlib.util,pathlib,json; p=pathlib.Path("research/v2/authorizations/trn21-c5c-a0-stage-r2-cpu-g2-final/cpu_g2_invocation_final.py").resolve(); spec=importlib.util.spec_from_file_location("cpu_g2_exact",p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); result=m.operator_once(); print(json.dumps(result,sort_keys=True),flush=True)'
```

## Point-in-time gates and attempt

- A separate complete read-only host audit passed at `2026-09-25T04:28:09Z`, before marker consumption: return code 0, empty stderr, stdout SHA-256 `6ccc3f847de8607ce9a6cf3dcdcd798704a678a44a9782b42cfeecfa70f3d35c`.
- `operator_once()` durably consumed the private fixed marker at `04:28:28.417103Z` before host contact. Its fresh full preflight snapshot was taken at `04:28:35.694942Z` and validated: host `ip-172-31-34-85`, boot ID `1903f7f0-8e0d-4011-9566-346e4ba01a0f`, mount namespace `mnt:[4026531832]`, root free bytes `58,368,049,152`; clean one-commit shallow/no-remote source checkout at `e65a3097a5b00f6178188a935bb2acb9a5042c6c`; two protected checkpoints with pinned full SHA-256 and no holders; six pinned warm-cache identities; all host STOPs absent; no relevant processes, host-lock or Neuron holders; new G2 scratch and label absent. In-wrapper audit return code 0, empty stderr, stdout SHA-256 `fe601eee4b77d5d6d1a3e16a599a56ee303f66f9cd18037675a128d71f02869b`.
- The one run transport returned code 0 with empty stderr and stdout SHA-256 `bd1f56559737e019d8171b962bdaacb3fd52f5bf02f19ef525250422357be8d1`. The raw response reports `cpu_only=true`, `launch_attempted=true`, `scratch_created=true`, `steps=3`, `nproc=4`, four rank records, `torchrun_rc=0`, and `timed_out=false`. The on-host CPU receipt is 17,293 bytes, SHA-256 `1469ce9ad30258ed65919449e17119e613c72e21b1bc5d6c90f758598cbbccb7`.
- A **separate full read-only postflight** snapshot was taken at `04:29:25.325883Z`: return code 0, empty stderr, stdout SHA-256 `3da1492e67fffce8bb73b90bf31f813982c1e95d53dbbe63accd4b9ae0e76371`. Boot and mount identity held; both protected checkpoint hashes and all six cache identities matched preflight; STOPs remained absent; processes, host lock, checkpoint holders, and Neuron were clear. The new private scratch root and CPU receipt remain preserved; the postflight receipt SHA-256 equals the run receipt SHA-256. Root free bytes were `57,897,570,304`.
- Terminal wrapper event at `04:29:26.105091Z`: `CPU_G2_RAW_EVIDENCE_PENDING_INDEPENDENT_NUMERIC_REVIEW`, with `native_training_authorized=false` and `evaluation_authorized=false`. This Operator makes no numeric interpretation. The consumed marker prohibits another same-label attempt.

## Evidence custody

- Original fixed marker/raw journal: `/home/poojith-devan/.local/state/trainium-trn21-c5c-a0-stage-r2-cpu-g2-attempt-once.jsonl`, owner UID 1000, mode `0600`, 145,188 bytes, SHA-256 `325c757c1b5ce177142d8f58fee2f46371bc55db1067dbaf45380393294b6818`. It remains in place and was not committed.
- Committed redacted copy: `RAW_FIXED_ATTEMPT_JOURNAL.redacted.jsonl`, 145,137 bytes, SHA-256 `82362050aad2c432684d46e66594db658ab6eff671346e6612d0cf4f4e3f3232`. The copy differs only by replacing the three SSH identity-file path occurrences with `[REDACTED_SSH_IDENTITY_FILE]`; all seven events and the full remote stdout/stderr remain unchanged. No private key material, credentials, tokens, or `.env` file is included.

The complete raw CPU receipt, rank records, command outputs, and both validated host snapshots are embedded in the redacted journal for an independent Verifier. The only authorized next step is review of this evidence and a separate Supervisor decision.
