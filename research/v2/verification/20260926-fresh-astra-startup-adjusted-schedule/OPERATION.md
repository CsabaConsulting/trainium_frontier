# Fresh startup-adjusted-schedule source review — live session

One new isolated `gpt-6-astra`/high review started at
**2026-09-25T22:36:36.086050+00:00**, thread
`01a0dab6-9e1b-7fc1-b0c5-d7f59cfe0c0d`, child PID814577, outer session98983.
This operator is not the fresh reviewer. Its verdict and own tests remain
**PENDING** at this launch snapshot. No source approval follows from this record.

Exact candidate7419927c7417b5666a984ec9c2b849feadabaaa9, SHA256
`36a86274c20a9dabce9fc7c34878896c7664f38026cf740b09ca12eef93d4ffb`;
baseline b4d83b5db66971f3c348837f04f53cba65aa6d33, SHA256
`3e79a7c1af5f72b68b7b8bd73fbbb458a04b12e5475a528ab28a49dda02d6f2c`.
Raw evidence frozen at225f27d13c25c9160f5fd2758cf016fb4c70bf0f.

Input packet `/tmp/trainium-fresh-startup-schedule-packet-20260926.loz9jxr3`;
manifest `fef50271f26087f3a6ed286f217a05e27b96abc40eb1960a1d385813a13f11f4`.
Run `/tmp/trainium-fresh-startup-schedule-review-20260926.hp4v6ff1`.
The17 task files are one complete baseline→candidate diff, one neutral lane
specification, and15 raw receipt/command files. Corrected five-boundary receipts,
the prior two-boundary receipts, and the original observer failure are retained.
No implementation tests, notes, conclusions, README, prior review, shared branch,
repository history or model checkpoint is supplied. The source is reconstructed
from the full-context diff, not imported from a shared worktree.

The previously reviewed operation is unchanged except packet path and manifest
pin. Outer bubblewrap exposes read-only system/CPU libraries and `/packet`,
writable `/work`, an empty home and no repository, SSH or process environment
filesystem. Authentication is memory-only for the CLI; tools receive only the
explicit credential-free allowlist. Provider network is available; tool network
is forbidden by the specification, not claimed separately OS-blocked. The
redundant inner sandbox is disabled while actual outer containment is retained.
Ephemeral/ignore-user-config/ignore-rules, no fork/resume, no Opus/Fable call or
automatic retry. One1200-second cap, one single-thread tiny CPU slot, no tensors
over64MiB, no full models or hardware. No other fresh session ran at launch.

The reviewer must default REFUSE, inspect the clock/flag plumbing, and author
and run its own fake-clock and actual tiny numerical reference. It must keep
default/OFF identity separate from intentional ON schedule change and distinguish
source/reduced CPU suitability from unexecuted full-size/native/quality gates.
No conclusions from this operator's separate static/raw audit are sent back.

The32-check prelaunch and35-check postlaunch input/isolation checks are retained.
The launch archive freezes all packet bytes and five non-secret operation files;
live events/output are deliberately not represented as terminal evidence.
`PREPARED.json` records the earlier prelaunch state, not current status.

OpenAI Docs was used for model/CLI verification:
[GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra) and
[CLI reference](https://learn.chatgpt.com/docs/developer-commands?surface=cli).
Local CLI help verified the ephemeral and configuration-isolation flags. Actual
provider identity is not invented beyond the recorded CLI/model evidence.
