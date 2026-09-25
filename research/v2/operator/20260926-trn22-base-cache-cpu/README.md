# Exact bcbd full-parameter CPU qualification — raw receipts

Assignment c43a10d3; frozen packet2cf504d9. All four serial probes completed
with child/torchrun/outer exit0, no timeout,4 ranks/3 updates/full117,453,580
parameters and62/62 changed tensors. Reference/default/OFF/ON elapsed20.5,
20.3,20.3,20.6 seconds respectively. All three unchanged G2 comparisons
returned identical:true for12 losses/62 final digests/ownership plan;
all four whole-checkpoint SHA/size pairs are identical. Checkpoints retained
on host; none downloaded. This is CPU qualification, not native performance,
BPB, intermediate-gradient/optimizer-state proof, adoption or launch authority.

Each run held original common lock inode832589 and passed fresh ps/neuron-ls
under that lock. Descendants/device quiescence were checked between runs and
at completion. The last ON arm ended21:06:19.514347UTC; final comparison and
quiescence receipt21:06:51.748554UTC. No automatic next job ran.

Authoritative retrieved raw is **results-complete/**:88 whitelisted small files,
258,412 artifact bytes, including all16 raw rank reports, all4 aggregate G2
reports, full torchrun logs, requests/environments/preflights/locks/exits and
checkpoint identities. All SHA256 values were checked before and after local
extraction. RETRIEVAL_MANIFEST.json maps every file to its original host path.
The unchanged packet auditor --results passed numeric_results_checked:true.
Independent numeric acceptance remains required before a separate cold release.

Artifact-only transfer failure: multi-source SCP hit
subprocess.TimeoutExpired after60 seconds per group. Partial files remain in
results/ locally and are NOT authoritative; no native job failed or reran.
The successful second retrieval used one read-only size-bounded in-memory
stream and exact SHA verification; no host writes, checkpoint transfer,
cleanup, source/evaluator changes, retries of consumed job labels, or AWS actions.

The pure3e79 trn21 replicate is prepared separately in e6437df9; seed43's active
fresh20M remains untouched. trn22 has no native launch authority from this receipt.

