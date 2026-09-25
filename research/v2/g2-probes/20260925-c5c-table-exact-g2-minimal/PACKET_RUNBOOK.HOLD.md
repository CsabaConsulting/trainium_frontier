# Exact c5c A1/A2 CPU G2 packet — disabled

These two packets in `PACKETS.HOLD.json` are an Implementer draft only. They do
not authorize SSH, staging, CPU G2, Neuron training, evaluation, deletion, or
submission. The separate Verifier must review exact source/tool/prepare/queue
bindings and the Supervisor must issue a new one-use **CPU-only** assignment.
Neither old `r1-g2` label is reused. Host absence and uniqueness are still a
fresh Operator preflight, not established by this file.

The first eligible CPU slot is after the corresponding host's A0 training,
fresh public-20M evaluation, and 120-second cache-discriminator smoke are
terminal, with an idle/quiescent postflight and no STOP/breaker. Do not let a
CPU G2 overlap a train/eval/smoke. A1/trn21 and A2/trn22 are separate
predeclared profiles; no A2-vs-A1 layer-1 claim follows from cross-host results.

## Operator procedure after separate review and assignment

1. Recheck controller state and all `research/experiment-protocol.md` gates.
   Inspect the **fresh** process/Neuron device/host-lock state, root disk
   bytes, `MemAvailable`, inodes, STOP/breaker, prior A0/smoke terminal
   receipts, and label/root/path absence. Minimum planning floor is 32 GiB
   free root and 64 GiB `MemAvailable`; the separate 6.5-GiB operational
   floor remains a live abort condition. The simple table-tensor geometry in
   the JSON is **not** a measured peak: A1 at least 0.75 GiB/rank, A2 at
   least 1.5 GiB/rank, before model/activations/trace copies/optimizer
   temporaries/compiler/checkpoint. Insufficient capacity means HOLD, not a
   smaller multiplier or different scratch filesystem.
2. Atomically establish the packet's previously absent private `run_root`
   (`0700`) and private `source/`, `tool/`, `tmp/` children. Stage only the
   exact local c5c source and exact G2 tool by plain checked transfer; verify
   their host SHA-256 against `PACKETS.HOLD.json`. Verify the packet's
   pre-existing trusted `prepare.py` is one regular unchanged file with its
   pinned SHA-256; never copy or modify it. Verify the pinned Python
   interpreter exists and imports the expected Torch. Record the exact
   environment/argv. Do not use an old e65 checkout's untracked tool.
3. Bind `TMPDIR` to that pre-created `tmp/` and prove its resolved
   `tempfile.gettempdir()` is exactly the intended path under a clean
   environment. The unchanged G2 tool uses unqualified `mkdtemp(prefix="g2-")`;
   silent fallback to `/tmp` or `/dev/shm` is a failed gate. Set
   `PYTHONDONTWRITEBYTECODE=1`, seed 42, LNC2, the pinned compiler flags
   including `--optlevel=1`, and 1800-second science budget. Do not pass
   `--stub-prepare` or `--real-data`: the real prepare module is imported,
   while training inputs are synthetic.
4. Run the exact array below from a clean environment with a single
   externally bounded invocation and an atomically exclusive (`O_EXCL`)
   `attempt_marker` recorded before spawning workers. The absolute source,
   tool, prepare directory, `TMPDIR`, output, and flags are taken from the
   selected JSON packet. The G2 tool spawns four CPU/gloo workers itself.
   Preserve launcher stdout/stderr, the JSON receipt, and its `--keep`
   `g2-*` directory even on failure. A timeout/kill triggers full
   quiescence and postflight; do not delete partial evidence or retry the
   label.
5. Do a separate read-only postflight and hash-verified retrieval of raw
   `torchrun.log`, all four `rankN.json`, checkpoint, aggregate G2 JSON,
   command/environment record and pre/postflight. A different Verifier
   computes the numeric result before any native-smoke request is prepared.

The exact command form, with only the JSON-selected absolute packet fields
substituted by the Operator after fresh verification, is:

```text
env -i PATH=/home/ubuntu/workspace/native_venv/bin:/usr/bin:/bin HOME=/home/ubuntu TMPDIR=<packet.tmpdir> PYTHONDONTWRITEBYTECODE=1 NEURON_LOGICAL_NC_CONFIG=2 NEURON_CC_FLAGS='--optlevel=1 --auto-cast matmult --auto-cast-type bf16' NEURON_COMPETITION_R1_SEED=42 NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS=1800 /home/ubuntu/workspace/native_venv/bin/python -B <packet.tool_host_path> --train-py <packet.source_host_path> --prepare-dir <packet.trusted_prepare_dir> --steps 2 --nproc 4 --micro-rows 2 --micro-tokens 256 --timeout 3600 --trace-rmsprop --keep --out <packet.g2_out> -- <packet.argv tokens in exact JSON order>
```

This is a command **template for a separately assigned Operator**, not a
dispatch instruction. The exact argv arrays are in the held JSON rather than
duplicated in prose. `--trace-rmsprop --keep` and `--out` are mandatory;
there is no BPB evaluation in this CPU gate.

## Numeric interpretation of unnamed RMS checks

The tool's `rmsprop_trace.checks` entries do not contain parameter names.
Independent raw-order attribution is possible only after the reviewer
rechecks exact c5c bytes and all rank records: `GPT.__init__` inserts
`ngram_embeds` into an ordered `ModuleDict` by increasing layer, bigram
before trigram (`train.py` around lines 1919–1927); `setup_optimizer`
passes `list(self.ngram_embeds.parameters())` into the sole RMSProp group
(around lines 2189–2196); and the frozen tool iterates `group['params']`
and appends one check for each non-null gradient on each optimizer call
(`preflight_dryrun.py` around lines 371–405). With exactly 2 calls and
4/8 checks per rank, the expected order in the packet is therefore
`b5,t5` repeated twice (A1) or `b1,t1,b5,t5` repeated twice (A2).
Verify the raw `tables` list, every `numel`, touched-row count, error and
unchanged-row bit, plus four-rank agreement. If a source/tool mismatch,
missing gradient, changed group order or extra optimizer group breaks the
proof, mark HOLD and request a separate additive tool patch with independent
review; do not call aggregate `PASS` an attributed RMS numeric PASS.

Only after this CPU gate passes may the distinct 30-step Neuron smoke be
proposed/reviewed/assigned. Only after that smoke and a valid independently
reviewed same-host c5c A0 control may a separate 1800-second public-20M
table experiment be considered. No model/evaluator/tool code was changed.
