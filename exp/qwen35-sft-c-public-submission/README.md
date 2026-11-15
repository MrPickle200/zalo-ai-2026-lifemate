# Qwen SFT + CoT branch C public-test submission

Generate a local `public_test` JSONL candidate using the branch C setup measured in `qwen35-sft-benchmark-v1`: the exact merged SFT Qwen tag, English production prompt with the unchanged TASK CoT checklist, QUAN LifeMate fine-tuned ASR with KenLM beam32 transcripts, PhoASR history transcription, JSON actions, thinking enabled, temperature 0, and a 768-token request cap. This task creates a local artifact only; it does not submit to the Public LB.

The script verifies the completed benchmark config, model/version/digest, exact C prompt hash, QUAN checkpoint and transcript manifest, and public-test ID order before inference. Each sample gets one official ToolEnv. The prediction sidecar takes calls and state hashes from that ToolEnv, including any calls made before a caught runtime error. The official schema/order validator must accept the full candidate before its files are moved into `outputs/submissions/`.

## Scripts

- `scripts/make_submission_c.py --prepare`: verify and lock the branch, transcripts, public-test manifest, source hashes, and local Ollama fingerprint.
- `scripts/make_submission_c.py --run`: run the full public-test set using system Python outside `.venv`; write the submission and matching sidecars only after validation.

## Artifacts

- Final JSONL, prediction sidecar, and manifest: `outputs/submissions/qwen35-sft-cot-quan-beam32-num-predict-768-num-batch-512-public-test*`.
- Lock, generation log, request telemetry, schema-validation log, and summary: `output/`.
- No raw chain-of-thought, secrets, labels, or `*_environment/` files are stored or read.

## Completed run

- The candidate contains 634 rows and 634 unique IDs in public-test manifest order. The official schema/order validator reported zero errors. The generation pipeline did not submit to the Public LB; the user later reported a score of 39.16 for this candidate, which Codex did not independently verify.
- There was one recorded Ollama CUDA out-of-memory error on `public_0527`. The per-sample sidecar retains the ToolEnv calls and state hash. Do not treat this candidate as runtime-failure-free.
- Local RTX 4050 latency was mean/p50/p95/max 22.70/19.62/46.74/86.73 seconds; 158 samples exceeded 30 seconds. This is local diagnostic data only and does not establish A30 timing.
- Final SHA-256: submission `37237f3651686e040dea4ad7334df763ef36d7069e661f94b8c2304f613f9e90`; predictions sidecar `51db84d76b5a1e3a615afda068b11f69a32131fd365f7f48befee080b17ef004`.
