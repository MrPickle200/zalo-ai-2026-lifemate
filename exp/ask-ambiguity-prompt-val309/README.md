# Prompt ambiguity experiment

`scripts/run_prompt.py` runs the exact English + CoT baseline from the latest public-test submission and a single-sentence replacement on the same val309 inputs. It locks model digest, transcript hash, and both prompt hashes before the first inference. The baseline hash must match the submission manifest. Runs use native system Python to reach local Ollama, one ToolEnv per task per branch, and alternating 15-task blocks. The production prompt is not edited.

`scripts/analyze_prompt.py` replays gold calls through official ToolEnv and reports BTC, ask errors, timing, and paired differences. A score difference on val309 is a local diagnostic; no submission is created.
