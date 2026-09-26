The files in `/work` are independent source/CPU review artifacts. None is a native success receipt.

| File | Meaning |
| --- | --- |
| `REPORT.md` | Findings, limited verdict, API assessment, original failure interpretation, and native blockers. |
| `independent_cpu_test.py` | Independently written bounded standard-library test. |
| `candidate_consumer_nki.py`, `baseline_consumer_nki.py` | Exact full-context-diff reconstructions; hashes match the packet. |
| `packet_integrity.json` | Ten packet-file checks, two source checks, eighteen documentation-excerpt checks. |
| `cpu_test.stdout.jsonl` | Raw stdout: one default-disabled checker message, 48 case records, one summary. |
| `cpu_test.stderr.txt`, `cpu_test.exitcode.txt` | Raw stderr and process return code. |
| `cpu_artifacts/summary.json` | Aggregate results, source/test hashes, resource limits and measured local CPU usage. |
| `cpu_artifacts/source_audit.json` | Static AST/model binding checks. |
| `cpu_artifacts/negative_controls.json` | Detected mutants, numerical counterexamples, and nonfinite comparator outputs. |
| `ARTIFACT_SHA256SUMS.txt` | SHA-256 inventory of every final artifact except the inventory itself. |

Each numbered case directory under `cpu_artifacts/` contains:

| File | Format |
| --- | --- |
| `gradient.fp32le` | 256×128 row-major little-endian FP32 words. BF16 cases store exact expanded BF16 values, so the low 16 bits of every word are zero. Logical-empty/short cases retain fixed storage, with unused rows filled by finite poison. |
| `expected.fp32le` | 256×128 little-endian FP32 output from the independent integer oracle. |
| `actual.fp32le` | Same shape/format, from the hand-written candidate CPU model. The name does not mean native output. |
| `writers.u8` | One byte per output element, row-major. Every successful byte is 1. |
| `ids.i32le` | Logical token IDs as little-endian signed 32-bit integers. Length is the logical token count, including zero for empty cases. |
| `metadata.json` | Full counts, starts, packed stable positions, and group maxima. |
| `traces.json` | Two program traces. Each group records its bound, active gather events `[iteration, row, position_address, token]`, skipped lanes, and inactive retained element count. |
| `result.json` | Case checks and SHA-256 hashes of raw binary files. |

The invocation was:

```sh
python3 -I -S /work/independent_cpu_test.py > /work/cpu_test.stdout.jsonl 2> /work/cpu_test.stderr.txt
```

The wrapper saved its return code in `cpu_test.exitcode.txt`. The test refuses to overwrite an existing `cpu_artifacts` directory. To repeat, first archive/move the existing artifacts and logs, then run the command with `/packet` and the exact source reconstruction present. Repeated runs produce the same case content; elapsed time, process resource use, and Python version fields may differ. The test enforces 120 CPU seconds and 256 MiB address space, uses 48 fixed cases (maximum declared 64), and accesses only local packet/review files and the standard library.

The source reconstruction procedure strips one diff prefix character from the single full-context hunk: retain context and removed lines for the baseline, context and added lines for the candidate. The resulting files contain 72 and 75 lines respectively. Both were SHA-256-verified against `INPUT_MANIFEST.json` before review.
