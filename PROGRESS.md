# Progress

| Tên | Nội dung | Local CV | Public LB |
|---|---|---:|---:|
| baseline-v0 | Whisper Small + Qwen3-4B-Instruct-2507 (pending local weights) | - | - |
| ollama-q4 | Whisper Small + Ollama Qwen3-4B-Instruct-2507 Q4_K_M | tool-call F1 0.0247 (20 samples) | 7.85 |
| baseline-v1-val309 | Whisper Small + Ollama Qwen3-4B-Instruct-2507 Q4_K_M; fixed grouped validation | proxy 0.1919; tool-call F1 0.1155; state match 0.2492 | - |
| oracle-asr-v1-val309 | Ground-truth transcript + Ollama Qwen3-4B-Instruct-2507 Q4_K_M | proxy 0.2920; tool-call F1 0.2429; state match 0.3301 | - |
| retrieval-v1-val309 | TF-IDF top-3 + max-5 tool shortlist + few-shot + missing-info demos; 634-row public submission created | proxy 0.2432; tool-call F1 0.2299; state match 0.2880 | 6.66 |
| retrieval-v2-wide-val309 | TF-IDF top-5 + max-8 tool shortlist + few-shot + missing-info demos | proxy 0.2392; tool-call F1 0.2287; state match 0.2783 | - |
| soft-retrieval-v1-val309 | TF-IDF top-3 demos + full tool schemas + missing-info demos | proxy 0.2558; tool-call F1 0.2435; state match 0.2977 | - |
| soft-retrieval-v1 | TF-IDF top-3 demos + full tool schemas; 634-row submission | proxy 0.2558; tool-call F1 0.2435; state match 0.2977 | 7.51 |
| phoasr-small-baseline-v1 | PhoASR-whisper-small + baseline Qwen3-4B, full tool set, no retrieval; 634-row submission candidate validated | proxy 0.2286; tool-call F1 0.1693; state match 0.2751; WER 0.2804 | 9.79 |
