"""Standalone diagnostic (not part of train.py/prepare.py): checks whether
USE_STOCHASTIC_ROUNDING's causality_check failure (max_abs_diff=0.021134) is a real
causality leak or a false positive from stochastic rounding's own forward-to-forward
noise. Runs model(x) TWICE on the *same* input under NEURON_RT_STOCHASTIC_ROUNDING_EN=1
and measures the diff -- if that self-consistency diff is comparable to the observed
causality_check failure, the failure is noise, not a real bug. Does not modify
prepare.py or train.py.
"""
import os

os.environ.setdefault("NEURON_RT_STOCHASTIC_ROUNDING_EN", "1")

import torch
import torch_neuronx  # noqa: F401 -- registers the neuron backend

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from train import GPT, GPTConfig, DEPTH, ASPECT_RATIO, HEAD_DIM, SEQ_LEN

device = torch.device("neuron")

base_dim = DEPTH * ASPECT_RATIO
n_embd = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
n_head = n_embd // HEAD_DIM
cfg = GPTConfig(vocab_size=8192, n_layer=DEPTH, n_head=n_head, n_embd=n_embd, sequence_len=SEQ_LEN)
model = GPT(cfg).to(device).eval()

torch.manual_seed(0)
x = torch.randint(0, cfg.vocab_size, (1, SEQ_LEN), device=device)

with torch.no_grad():
    a = model(x)[:, : SEQ_LEN // 2 - 1].float()
    b = model(x)[:, : SEQ_LEN // 2 - 1].float()

diff = (a - b).abs().max().item()
print(f"self-consistency max_abs_diff under SR (same input, two forward passes): {diff}")
print(f"compare to the causality_check failure this session: 0.021134")
