# 04:00 IST single-launcher self-audit

Read-only snapshot2026-09-25 22:30:17UTC /Sep26 04:00:17IST. Existing work untouched.

trn21: exact75b9 Recursive architecture cold-five, label fs-20260926-ct-arch75b9-recursive-cold-trn21-r1. Driver3069190, torchrun3069219, four worker/device holders3069234–3069237. Original common flock inode596823/dev66305 is held; competitor refused.26.640GiBfree. OldqueuePID2983496 absent and no host_queue.py command lines. Single active launcher; no train/eval overlap.

trn22: exact3e79 LR0.012 full training finished3629steps at22:28:32.631553UTC; attached fresh20M evaluator3174649 active under original driver3151669/common flock inode832589/dev66305. Competitor refused.30.270GiBfree. OldqueuePIDs3050509/3053905 absent and no host_queue.py command lines. Single active launcher; no extra queued native job.

trn21's bounded gap after CPU ended22:23:25.132545 to cold launch22:29:03.775502 was338.642957s, under15min. No old queue was restarted; no secret, AWS, source, evaluator, data or cache edit. Snapshot includes timestamped ps, neuron-ls, lock, disk and active identities. Job outcomes are pending; compilation is not a quality result.
