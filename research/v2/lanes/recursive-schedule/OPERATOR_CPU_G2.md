# Exact-size CPU G2 handoff (not yet run)

Only a distinct Operator may run this on an available large-memory CPU host.
Do not run full-size table G2 on the laptop or overlap the existing Neuron job.
Candidate train SHA is
`f9b6a9cbb1a1760fb8db422ffd79ffb91e9280899c40aa86028d935acf355f67`;
unchanged G2 tool SHA is
`6a54589127a31beb5bde39fec8f58db370c2af5198ca4d6bca2bac684278d0b8`;
organizer prepare SHA is
`5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb`.

Perform source/idle/disk/STOP/available-RAM checks, use a new unique receipt
label, and create an isolated CPU output directory. Do not install packages,
change prepare.py, use real training data, or include local test_support in
PYTHONPATH. The real organizer prepare module is imported for compatibility;
the unchanged G2 tool supplies deterministic synthetic inputs and its recorded
CPU shims. Use the proven host Python with Torch 2.11, not the laptop Torch
2.6 compatibility shim.

The exact argv for each possible selected profile is in `PROFILES.json`.
The command template below uses explicit placeholders resolved by the Operator
to previously staged and verified absolute paths; it is not a runnable launcher.

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NEURON_LOGICAL_NC_CONFIG=2 NEURON_CC_FLAGS="--optlevel=1 --auto-cast matmult --auto-cast-type bf16" NEURON_COMPETITION_R1_SEED=42 NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS=1800 /home/ubuntu/workspace/native_venv/bin/python -B <staged-tool>/preflight_dryrun.py --train-py <staged-candidate>/train.py --prepare-dir <verified-organizer-repo> --steps 3 --nproc 4 --micro-rows 2 --micro-tokens 256 --timeout 3600 --trace-rmsprop --keep --out <isolated-receipt-dir>/g2.json -- <selected-profile-exact-argv>
```

This runs the candidate's real loop on CPU/gloo. The tool itself appends
`--device-type cpu --no-compile --num-steps 3 --out-dir <private-g2-dir>`;
it does not launch Neuron training, alter persistent experiment checkpoints,
invoke public evaluation or compute BPB. It does allocate full-size model and
table parameters on CPU, and writes temporary CPU checkpoints. Preserve raw
rank reports/logs and aggregate JSON; separately review exit status, three
finite step losses, rank consistency, touched tables and strict loading.

Dense and whichever table profile is selected need their own exact argv
receipt. A1 and A2 G2 receipts are not interchangeable. Current evidence is
only the bounded tiny synthetic test; no exact-size PASS is claimed.
