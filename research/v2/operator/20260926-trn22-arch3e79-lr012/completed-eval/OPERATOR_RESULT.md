# LR0.012 architecture result — raw Operator archive

Fresh public-20M/context1024/batch1 BPB: **0.9883603019664198**.
Raw totals: 57,178,204 nats / 83,462,186 bytes, 20,971,520 tokens.
Same-host3e79 LR0.015 reference:0.9863403052402583; delta **+0.0020199967261615**.
This is not an LR improvement; the parent architecture also failed its cross-host confirmation.

Training ended22:28:32.631553UTC after3629steps; the independent observer measured807.8952129699755s to fifth boundary,2597.3018926839577s process wall,1789.4066797139822s charged through exit.
Eval exited0 at22:40:14.201351UTC; outer driver also exited0. At22:41:39UTC no job process remained and the original lock was free.
The checkpoint remained453048594bytes, SHA4fc42f95dd6f597bb1b1f1c3854464a3db2088f1a5d151063d0dd28dd3356f26; post-eval remote rehash22:42:18UTC matched. No checkpoint was downloaded or removed.

`retrieval.json` authenticates10 closed remote files. Full training receipts are in the sibling `completed-training/` archive. This is an Operator observation, not independent acceptance, adoption, or submission authority.
