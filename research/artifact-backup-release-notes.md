Recovery artifacts for the Trainium Frontier research campaign. Kept as a draft
to back up checkpoints, original EC2 working-directory snapshots and raw profiler
captures independently of the laptop and provisioned instances.

Every artifact has a separate manifest containing source-file SHA256 values,
payload encoding and ordered chunk hashes. Filenames include content hashes;
uploads never replace an existing asset. Tracked receipts under
`research/run-logs/fs-artifact-backup-*.json` record GitHub's asset digests.

Restore by downloading an artifact's manifest and listed assets. Concatenate
split chunks in manifest order, verify the payload hash, then extract a tar.gz
archive or restore the original checkpoint file. Verify restored files against
the source inventory. Source and ongoing logs are on
`research/fresh-start-20260918` and `backup/fresh-start-20260918`.
