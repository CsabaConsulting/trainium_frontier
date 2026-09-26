# Read-only trn22 checkout sizing

Observed2026-09-26 06:32:02.750449 UTC under the original exclusive
device66305/inode832589 lock, with fresh successful ps, neuron-ls and fuser
idle checks. Remote STOP checks and local trn22/global breakers were clear.
This is not staging, creation, reservation or launch authorization.

Original checkout HEAD remains65155c668efaa215a2136f67617b44ed32d4c9f6,
with exactly the six prior untracked Sep24 control logs. No tracked changes
were reported. The worktree registry still lists only the original checkout.
No old file, ignore rule, index, registry or checkout was intentionally changed.

The exact proposed owned path
`/home/ubuntu/trainium_k2_f0f2_trn22_evidence_20260926_r1` is absent.
Its parents are plain directories on device66305; findmnt reports ext4,
/dev/nvme0n1p1,259:1. Parent and lexical resolved metadata are retained.

The bound git ls-tree metadata lists3798 regular blobs,86770800 logical bytes,
zero symlinks and zero submodules. Rounding every regular file to4KiB gives
95301632 bytes;23 directory paths are represented. That estimate omits directory
allocation, worktree administration, index, filesystem overhead and job output.

Fresh free disk is17296412672 bytes. The16GiB floor is17179869184, leaving
116543488 bytes before a checkout. Subtracting only the rounded regular files
leaves an estimated21241856 bytes above that floor. Thus the regular files
appear to fit, but the complete checkout and diagnostic footprint are NOT
proven to fit. Do not lower the floor or treat these observations as permission.
Available RAM is129103466496 bytes, above the24GiB diagnostic floor.

The raw NUL-delimited tree listing is archived as base64 in trn22.json, not
printed broadly. Its SHA256 is
c84fe7a3ee1e09e9c9aa6db769103738db023499c608b3dc6a007dc25bcf8933.
Local decoding independently recomputed the listing hash,3798-file count,
logical total and4KiB allocation arithmetic. This is Operator receipt checking,
not independent verification or launch approval.

No SDK import, .pth execution, source/data execution or data-content read was
needed. Existing static SDK mapping evidence3605794a was referenced rather
than reread. No trn21 interaction occurred; its parked guardian was untouched.

Raw receipt SHA256:
bad64373f2a2660708ad11cc0008d91324cbb8d2d39cbbb58752d60b5e8a594f.
