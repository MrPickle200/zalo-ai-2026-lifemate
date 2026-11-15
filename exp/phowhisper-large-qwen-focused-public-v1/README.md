# PhoWhisper Large + Qwen epoch 3 + focused lookup prompt

Generate a 634-row public-test candidate using the locked PhoWhisper Large v2 transcript cache for all main and history audio, Qwen 3.5 epoch 3 English CoT, and the focused post-lookup disambiguation prompt evaluated on val309. The original PhoWhisper baseline candidate is preserved.

Use native system Python so the runner can reach local Ollama. Run `scripts/make_submission.py --prepare`, then `--run`. The run reuses the 1,933 cached audio transcripts, writes a separate submission candidate, validates it against the official public-test schema, and records a prediction sidecar and manifest. It does not upload the candidate.
