# First recovered live observation

At 10:59:42 UTC both exact original guardians and lock identities were intact.
Both transport calls returned 0, with empty stderr and no terminal/eval file.

- trn21: 494 complete boundaries; logged step 490, loss 3.1521,
  280497 tokens/s. `train/rehearsal.json` remains absent. Exact official
  startup and charged seconds are therefore not yet available.
- trn22: 70 complete boundaries; logged step 60, loss 4.6656,
  280867 tokens/s. Fifth boundary was after the prior observation beginning
  10:58:05.060526 UTC and by 10:59:42.822064 UTC. This polling bracket is
  not an exact official startup duration or a cold-cache measurement.

Raw recovery receipt hashes:

| Receipt | SHA-256 |
| --- | --- |
| trn21-status-105937.json | 58367188babdf518c3262d91a201e8c9f3ac884baca5d484382ceda0b79f6af1 |
| trn22-status-105937.json | 4ef2801024d15825aba5adb46a250eb1f984be4a82c215aa6e3e4382bd168146 |

The assertion of exact stack startup 893.5 seconds and approximately 24
charged seconds is not established by these receipts. Exact terminal
observer fields will be reported when they are emitted. No job was altered.
