# P r2 local preparation only

Exact99d7c261 packet was read; binding3c08ce4b and manifest4e6bc2f3 are pinned.
The helper, source profile, original lifetime lock, dispatcher and guard remain
unchanged. This transport derives from the consumed r1 transport, but targets
only the new r2 root. It verifies r1's exact refusal and explicit postflight.

No host staging or launch has occurred. The script deliberately refuses before
any transport until the Supervisor's independently reviewed one-shot release
commit and file path are bound. It is not launch authority. K2 priority remains.
Original r1 is preserved and may never be retried or overwritten.
