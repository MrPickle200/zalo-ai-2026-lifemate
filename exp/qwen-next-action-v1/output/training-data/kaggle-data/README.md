# LifeMate Qwen next-action SFT input

This private Kaggle dataset contains the fixed train-only trajectory timestep JSONL, its local replay manifest, and the QLoRA training runner. The 309 fixed validation IDs are excluded. Each row has the production agent chat format and a single next action target; observations were recorded by replaying calls through the official ToolEnv.

Training base model: `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0). Kaggle downloads this public model during the GPU notebook run. The input dataset contains no environment files or direct state snapshots.
