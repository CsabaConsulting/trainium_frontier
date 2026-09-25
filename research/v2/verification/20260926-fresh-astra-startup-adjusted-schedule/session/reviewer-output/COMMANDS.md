All commands ran locally and sequentially. No packet source, evaluator, repository, external data, device backend, or supplied harness was imported or executed. Paths printed inside receipts were treated as data. Only /packet, standard CPU/system libraries, and this review's own /work artifacts were read. Writes were confined to /work.

Initial inspection used `cat /packet/lane-spec.md`, `ls -la /packet`, `rg --files /packet`, `cat /packet/INPUT_MANIFEST.json`, `head`/`tail` on the full-context diff, and bounded `sed`/`rg` reads of the reconstructed files. A search for AGENTS.md under /packet and /work returned no matches (rg exit 1, normal no-match result).

The substantive executed commands, in order, were:

```bash
python -B /work/prepare_review.py > /work/prepare_review.stdout 2> /work/prepare_review.stderr
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -B /work/reference_test_v1.py --attempt attempt1 > /work/attempt1.stdout 2> /work/attempt1.stderr
python -B /work/diagnostics/receipt_probe_failed.py
python -B /work/receipt_audit_v1.py > /work/receipt_audit.stdout 2> /work/receipt_audit.stderr
python -B /work/receipt_audit.py > /work/receipt_audit_attempt2.stdout 2> /work/receipt_audit_attempt2.stderr
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -B /work/reference_test.py --attempt attempt2 > /work/attempt2.stdout 2> /work/attempt2.stderr
python -B /work/source_audit.py > /work/source_audit.stdout 2> /work/source_audit.stderr
```

Version provenance: the first reference run invoked `reference_test.py` before that exact file was copied to `reference_test_v1.py`. Likewise, the first receipt audit invoked `receipt_audit.py` before it was copied to `receipt_audit_v1.py`. The failed schema probe was originally an inline `python -B -` script; its exact script body is preserved as `diagnostics/receipt_probe_failed.py` and its error in `diagnostics/receipt_probe_failure.txt`. The two `_v1.py` names above identify the preserved executed versions. These are not claims of separate additional runs.

Exit results: reconstruction 0; reference attempt1 0; schema probe 1 (KeyError); receipt audit v1 1 (KeyError, saved REFUSE); corrected receipt audit 0; strengthened reference attempt2 0; source audit 0. Some diagnostic shell calls displayed stdout after Python returned, so their shell exit status was the display command's 0; the Python scripts' own summaries and explicit exit logic preserve the failures.

An initial CPU dependency probe imported only torch and sys under the same thread environment. It reported Python 3.14.4, torch 2.13.0+cpu, and one intra-op thread. The numerical script additionally sets and asserts one inter-op thread. It does not construct a model or call autograd/backpropagation: gradients are complete explicit fixtures.

For a fresh repeat of the final numerical test, use a new output name, for example:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -B /work/reference_test.py --attempt reproduction1 > /work/reproduction1.stdout 2> /work/reproduction1.stderr
```

The test refuses to overwrite an existing attempt directory. Receipt audit directories also refuse overwriting. Every supplied receipt, including failures and the two-boundary attempt, is preserved byte for byte in `packet_copy/`. `input_integrity.json` records every packet hash and reconstructed-source hash. `artifact_manifest.json` inventories final review artifacts. Auxiliary JSON inspection commands did not execute source or change inputs.
