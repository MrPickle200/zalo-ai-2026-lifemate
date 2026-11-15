# PhoWhisper large + Qwen prompt, public-test candidate

Transcribe all 1,933 public-test audio clips with `scripts/transcribe_public.py` in the project .venv. This covers 634 main clips and 1,299 history clips with PhoWhisper large v2. Then run `scripts/make_submission.py --prepare` and `--run` using native system Python for local Ollama.

Qwen English-prompt epoch-3 SFT was the strongest saved val60 Qwen branch by BTC. The val309 prompt variant gained only 0.07 local 30-second-gated BTC points (paired 95% CI -3.07 to +3.08), lost 1.00 ungated point, and exact gold ask calls fell from 10 to 1. The baseline prompt is selected. The output under outputs/submissions/ is schema validated but not uploaded.

Model parameter count: 4,205,751,296 Qwen + 1,543,304,960 PhoWhisper = 5,749,056,256 total.
