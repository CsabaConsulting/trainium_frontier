# Actual dual-host progress at 03:18:38 UTC

One bounded read-only observation; no staging, launch, kill, cleanup, retry, source/driver/cache change or state mutation.

- **trn21 LR0.012:** step3280, loss2.7232,266,017 tokens/s, source LR multiplier0.183,188 seconds remaining, epoch2. Original four workers3152652–3152655 active; original66305:596823 lock held. Disk15,723,196,416 bytes free.
- **trn22 Recursive full:** step440, loss3.3857,266,799 tokens/s, source LR multiplier0.931,1585 seconds remaining, epoch1. Original four workers3252248–3252251 active; original66305:832589 lock held. Disk20,691,812,352 bytes free.

Both runs were training, with no terminal or attached-eval launch yet. No current compiler descendants, traceback/runtime/assertion failure or retired queue PID was observed. Neither host is free. All current worker identities, source hashes and PID/starttime/PPID/PGID/SID are preserved in raw receipts.

First five printed losses are finite on both. Child Recursive full printed9.0101,8.7779,8.4047,8.0711,7.7681, matching the cold run at printed precision only. Its first printed step took799.09 seconds. The exact uncapped first-five harness measurement is not available until finalization; do not substitute first-step duration or source capped allowance for it. trn21 likewise retains its first five logs but awaits the final harness measurement.

Raw SHA-256: trn21 `e2e56e1f672bf04adfde97fadb0bbe1bbaff34740296219af1bdc2c1977f8fdb`; trn22 `de4955e643f65bb8f3559d0a6ee8ad564a7b90ed81412e9ebead89eca1d59ced`.
