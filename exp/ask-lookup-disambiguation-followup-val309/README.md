# Post-lookup ambiguity prompt follow-up

This follow-up tests a concise rule for selecting IDs after search/list results. It reuses the exact baseline predictions from ask-ambiguity-prompt-val309 and runs the focused prompt once on val309.

Primary paired analysis is the 262-task complement of the 47 gold ask tasks used to form the hypothesis. Those 47 are reported separately as a diagnostic subgroup, not as confirmation. The baseline is reused; model, PhoASR/QUAN transcript inputs, history ASR, ToolEnv runner, seed, decoding, and action cap are locked to the prior prompt experiment.

Run scripts/benchmark.py --prepare, then --run and --analyze with native system Python so the runner can reach local Ollama. The script refuses virtualenv execution, checks model and prompt fingerprints, and does not modify the production prompt or create a submission.
