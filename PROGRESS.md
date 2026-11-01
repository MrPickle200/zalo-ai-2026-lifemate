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
| phoasr-lifemate-ft-v1 | PhoASR-whisper-small LifeMate fine-tune + baseline Qwen3-4B, full tool set, no retrieval; 634-row public candidate validated | proxy 0.2222; tool-call F1 0.1634; state match 0.2686; WER 0.2757; CER 0.1966 | 9.78 |
| phoasr-hybrid-v1 | PhoASR public + LifeMate FT dual-transcript prompt + baseline Qwen3-4B, full tool set, no retrieval | proxy 0.2064; tool-call F1 0.1509; state match 0.2718 | not submitted |
| qwen-lora-v1-phoasr | PhoASR public + Qwen3-4B LoRA SFT trên LifeMate | dataset audited; training not run (PEFT backend unavailable) | - |
| iterative-agent-architecture-audit-v1-val30 | PhoASR public + Qwen3-4B iterative ToolEnv, 30 complex held-out samples; compared caps 5/6/7 | cap5 proxy 0.0518 / F1 0.1073 / state 0.00; cap6 0.0512 / 0.1058 / 0.00; cap7 0.0512 / 0.1053 / 0.00 | - |
| phoasr-nbest-v1-hard30 | PhoASR public; five decode configs; cached alternatives + WER-oracle and consensus selector diagnostics | current: WER 0.3048, proxy 0.0518, F1 0.1073; WER oracle: WER 0.2890, proxy 0.0672, F1 0.1441; consensus proxy 0.0518; all state match 0/30 | - |
| phoasr-nbest-v1-val100 | Same five PhoASR decode configs on the first 100 fixed val IDs outside hard-30; cached alternatives, WER oracle and consensus selector | BTC formula estimate: current 6.107; WER oracle 7.274; consensus 7.274 (0–100); corresponding local proxies 0.2464/0.2567/0.2542 | - |
