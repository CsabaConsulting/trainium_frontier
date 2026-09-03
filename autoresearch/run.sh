#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"

# Setup data/tokenizer if needed
python3 ../prepare.py setup --build-tokenizer-if-missing

# Single-core 5-min autoresearch run (LNC=2 for trn2.48xlarge)
NEURON_LOGICAL_NC_CONFIG=2 python3 train.py
