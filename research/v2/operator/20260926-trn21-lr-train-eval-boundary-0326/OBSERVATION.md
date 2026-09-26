# trn21 LR012 training → fresh20M boundary

One read-only snapshot at 2026-09-26T03:25:58.298636+00:00. No host changes,
staging, launch, stop, restart, source/cache edits or new job. trn22 was not contacted.
Raw receipt trn21.json SHA256
fc48525fee3d026bd2fdda02daeb5ca8d1ef78369d321158711b978b30d67ccf.
The directory's 0326 label is approximate; the raw UTC is authoritative.

Training ended naturally at03:21:33.346937UTC, exit0,3,632 steps/476.1M tokens.
The full training log, all3,632 harness observations, source run summary and
checkpoint identity were retrieved completely. First-five observed seconds were
796.9385078420164,797.6911712669535,798.1927826069295,798.6943702140125,
799.1958145109238. Actual process wall2588.663411580026s minus observed fifth
799.1958145109238s gives1789.4675970691023s through process exit, budget true.
The separate source estimates are startup797.1s/charged1785.2s; do not substitute
these estimates for the actual harness accounting or claim this warm run cold.

Retained checkpoint453,048,594B, inode3186290 ondevice66305, recorded SHA256
3d2a0663fa285c4d2be9f3214aee2ace58e07fd4311b98370f87c45f3ac80fbc.
Its hash comes from the completed harness/checkpoint record. No checkpoint download
or fresh rehash while live eval was performed; post-eval final hash remains pending.

Attached fresh public20M started03:21:33.415409UTC and is still active:
timeout3175131(start93327987), harness3175132(start93327987), evaluator3175146
(start93328241), all PGID/SID3152603. Evaluator actual argv is prepare.py eval-public
with --batch-size1 --seq-len1024 --eval-tokens20971520 and the exact retained
checkpoint. neuron-ls reports that evaluator as the sole device process.
Dispatch3152603/flock3152604/driver3152605 remain alive. Original66305:596823
flock remains held by3152604. The host is NOT free; no terminal/final BPB exists
in this observation. Earlier workers and observed compiler IDs are preserved in
known_identity_floor and older raw observations for eventual explicit postflight.

Disk free15,269,175,296B; retired queue2983496 absent. Source/error logs include
ordinary Neuron warnings, no observed fatal error. A local pretty-print of the
3,632-observation rehearsal exceeded display output length; the immutable JSON
receipt is complete and untruncated (366,693 captured source-file bytes).

Optional same-host read-only parent-path feasibility data was captured inside the
same observation, without touching active LR source. Completed exact36 parent
train/train.py is regular mode0444,uid1000,device66305,inode3185544,size324457,
SHA36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb.
Colocated prepare.py is regular mode0600,uid1000,inode3185545,size35467,
SHA5807c46e08ce3c4ac1394405377e57ea79acd5db18c8d7be8d853460a2bd3bbb.
Both before/after stats were stable; exact mtime_ns/ctime_ns are in raw. All six
immediate cache roots from the pinned parent binding exist. These facts establish
neither filesystem immutability nor cache hits, and authorize no reuse/change.

Pending final retrieval: all eval output/terminal/exits, post-eval checkpoint hash,
closed graph CSVs at their actual names, and complete actual descendant/compiler/
device/original-lock postflight before any separate next-job release.
