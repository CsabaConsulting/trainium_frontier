# Five selected training shards — metadata refresh

At 11:06:32 UTC on both hosts, all four actual current worker identities
matched their recorded PID/start ticks and exposed the expected effective
`NEURON_COMPETITION_R1_CACHE_DIR=/home/ubuntu/.cache/neuron_competition_r1`.
The corresponding data directory is that path plus `/data`.

Exactly `shard_00000.parquet` through `shard_00004.parquet` were statted.
All five are present, regular files, totaling459442478 bytes on each host.
Raw receipts retain each size, inode/device, and exact integer nanosecond
mtime/ctime. No contents, shard hashes, other data files, validation files,
setup/download or new Neuron workload were accessed by this check.

| Host | Observation UTC | Raw receipt SHA-256 |
| --- | --- | --- |
| trn21 | 2026-09-26T11:06:32.889186+00:00 | 59a7850ae92c3e1e2f934b35892c3b43aef7afa39c1428461f298ae26d701756 |
| trn22 | 2026-09-26T11:06:32.823349+00:00 | 0dfc0579cbb8b817e29a46b5afb8eae55da9883119b2ae3db21f8797997c7642 |

Prepare SHA5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb
comes from the authenticated existing training-launch records and packet
bindings; it was not freshly hashed by this metadata-only call. Historical
same-content proof remains the separate existing training-shard-provenance
comparison. This refresh does not claim visibility into official runner
overrides or inspect private data.
