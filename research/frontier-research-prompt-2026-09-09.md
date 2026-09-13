There's a competition to train a NanoGPT like model on AWS Trainium 2 chip and achieving the lovest val_bpb score within 30 minutes of training. I've spent 4 days of agentic exploration various hyper parameters (such as number of model layers / DEPTH, model width, data batch parameters). I've learned that very short training time favors smaller DEPTH than what is optimal at 1800 seconds training and a ton of more things. I've tried special solution such as space state models like Mamba 2 (pure or hybrid) RWKV, QRWKV, or YAT / SLAY kernels. The main repository is in @trainium-frontier, and you can find the rules in @trainium-frontier/README.md, original plan in @trainium-frontier/research/experiment-rollout-plan and then exploration notes in @trainium-frontier/research/trn2-optimization-research and other files and setp-by-step experiemtn results in @trainium-frontier/research/trn2-experiment-results.md and other files. There are detailed run-logs too in @trainium-frontier/research/run-logs and scratch scripts, and related articles in @trainium-frontier/articles. There are related repositories besides the main repository (some of the notes also refers to, these include neuron kernel writing skills, NanoGPT related, autoresearch, RWKV, and other repositories). You are the most advanced AI model with deep thinking and research capabilities I was not able to get into the top 10 yet and I'm looking for 1. any gaps we might have missed (you will see that sometimes we found contradictions along the line of experiments). 2. Any potential ideas to stand oout from the crowd, you have Tavily web search and research skills to reach out to web or you can look at websites directly as well. Perform the most thorough deep research possible so I can minimize the val_bpb. Your deliverable is a detailed research plan markdown file which will serve as instructions for the other agentic loop, be as specific and detailed as possible and provide supporting thoughts.

Note: we'll try 3 top frontier models:
1. https://openrouter.ai/anthropic/claude-fable-5.1 (claude --resume 6457426e-cefc-4e59-a176-ef2ad789fab0)
1.1. https://openrouter.ai/anthropic/claude-fable-5
2. https://openrouter.ai/openai/gpt-6-astra-pro (claude --resume 4876f9d9-71fa-40c1-bc58-82adb950db4e)
3. https://openrouter.ai/meta/muse-spark-1.3-contributor (claude --resume e94fd67c-12cd-475f-8768-abcfa68f28c0)

In a CCR (Claude Code Router) + OpenRouter proxied Claude Code harness


Bedrock
===
1. us.anthropic.claude-fable-5-1 (claude --resume da46ad08-dcac-4b8f-96ec-8aee625202c9)
2. us.openai.gpt-6-astra
3. us.xai.grok-4.6

