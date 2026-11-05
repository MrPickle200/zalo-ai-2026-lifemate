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
| qwen-next-action-lora-v1-val309 | PhoASR public + Qwen3-4B QLoRA next-action adapter; fixed train-only validation | BTC formula 2.3193/100 vs baseline 4.9145; call-F1 diagnostic 0.0611; proxy 0.1635 | not submitted |
| iterative-agent-architecture-audit-v1-val30 | PhoASR public + Qwen3-4B iterative ToolEnv, 30 complex held-out samples; compared caps 5/6/7 | cap5 proxy 0.0518 / F1 0.1073 / state 0.00; cap6 0.0512 / 0.1058 / 0.00; cap7 0.0512 / 0.1053 / 0.00 | - |
| phoasr-nbest-v1-hard30 | PhoASR public; five decode configs; cached alternatives + WER-oracle and consensus selector diagnostics | current: WER 0.3048, proxy 0.0518, F1 0.1073; WER oracle: WER 0.2890, proxy 0.0672, F1 0.1441; consensus proxy 0.0518; all state match 0/30 | - |
| phoasr-nbest-v1-val100 | Same five PhoASR decode configs on the first 100 fixed val IDs outside hard-30; cached alternatives, WER oracle and consensus selector | BTC formula estimate: current 6.107; WER oracle 7.274; consensus 7.274 (0–100); corresponding local proxies 0.2464/0.2567/0.2542 | - |
| phoasr-nbest-consensus-v1-public | PhoASR five-candidate N-best consensus medoid + Qwen3-4B-Instruct-2507 via Ollama (context 8192); full 634-row public-test submission | - | **9.49** |
| tool-router-v1 | BamiBERT next-tool classification on gold-replayed train/val trajectories (512/1024 tokens) | top-1 0.2963; top-3 0.3950; first/second 0/0; STOP-only collapse | - |
| session4-prompt-ab-val309 | PhoASR + Qwen3/Qwen3.5 Q4_K_M, prompt v2 vi/en; cap5/6 paired, fixed train-only val309 | BTC cap5 3.0868; cap6 3.1702 vs baseline 4.9145; 2 context-overflow failures/run | not submitted |
| session4-qwen3-vs-qwen35-val309 | Full fixed train-only val309; PhoASR, prompt v2 English, cap5, context 8192 | BTC Qwen3 3.0868 vs Qwen3.5 1.3916; Qwen3.5 0 runtime failures vs 2 | not submitted |
| qwen35-B-nothink-native-public | Qwen3.5 4B Q4_K_M; PhoASR production-greedy; prompt v2 English; no thinking; native Ollama tools/role=tool; validated 634-row public submission | BTC val309 6.3484/100 (6.4779 before local runtime gate) | **15.01** |
| qwen35-C-think-json-public | Qwen3.5 4B Q4_K_M; PhoASR production-greedy; prompt v2 English; thinking + JSON protocol; validated full 634-row submission | BTC val309 5.6095 (9.2079 before local runtime mask); mean latency 33.04s, 318 samples >30s | **19.09** |
| qwen35-track-c-asr-ablation-pilot60 | Track C fixed; PhoASR FT vs QUAN FT + KenLM beam32, same stratified 60 val IDs | BTC before runtime mask 10.000 vs 15.730; paired +5.730 (95% bootstrap CI −1.444 to +13.405); WER/CER 0.2597/0.1945 vs 0.1786/0.1522; 8 improved, 4 worsened, 48 tied | - |
| qwen35-4b-promptv2-en-phoasr-v1 | Tạo file local cho public_test, PhoASR, prompt v2 English, cap5 | Qwen3.5 1.3916 | 8.85000 |
| qwen3-4b-promptv2-en-phoasr-v1 | Advance prompt for Qwen3 4B | BTC Qwen3 3.0868 | 8.53000 |
| qwen35-thinking-tool-protocol-val309 | Fixed PhoASR/prompt-v2 2×2 val309: A no-think JSON **1.3916**; B no-think native **6.3484** (6.4779 pre-mask); C think JSON **5.6095** (9.2079 pre-mask); D think native **7.2168** (7.4326 pre-mask) | Complete; paired bootstrap at `exp/paired-analysis/qwen35-thinking-tool-protocol-val309/output/paired-bootstrap-report.json`. Runtime mask uses local RTX 4050 timing, not A30 | - |
| asr-phoasr-vs-quan-val309 | ASR-only; PhoASR-whisper-small vs QUAN-Transcribe-1.0 + KenLM beam32 (alpha=0.3, beta=1.0), same 309 validation IDs | PhoASR WER 0.2804 / CER 0.2013 / 0.911 s per sample; QUAN WER 0.2915 / CER 0.2317 / 0.204 s per sample; per-sample WER wins: PhoASR 158, QUAN 99, tie 52. Sequential RTX 4050 timing; numbers are not digit-normalized. | - |
| qwen35-D-think-native-public | Qwen3.5 4B Q4_K_M; PhoASR production-greedy; prompt v2 English; thinking + native Ollama tools; validated full 634-row submission | BTC val309 7.2168 (7.4326 before local runtime mask); mean latency 16.43s, 35 samples >30s | **14.11** |
| asr-phoasr-vs-quan-train1545 | ASR-only trên all 1.545 train examples (in-sample); PhoASR-whisper-small vs QUAN-Transcribe-1.0 + KenLM beam32 (alpha=0.3, beta=1.0) | PhoASR WER 0.2855 / CER 0.2017 / 1.358 s per sample; QUAN WER 0.2882 / CER 0.2291 / 0.288 s per sample; paired WER wins: PhoASR 667, QUAN 592, tie 286. Timings sequential, RTX 4050; no digit-to-word normalization. | - |
| phoasr-domain-v1-val309 | PhoASR Whisper Small decoder LoRA (r=8), 2 epochs on 1,236 train IDs; fixed final checkpoint, no val selection | WER 0.2804→0.2576 (-0.0229; -8.1%); CER 0.2013→0.1838 (-0.0174; -8.7%); mean latency 0.911→1.276 s/sample | - |
| quan-domain-v1-val309 | QUAN CTC reader 1 epoch + reader/top-2 encoder 1 epoch on 1,236 train IDs; KenLM beam32 (α=0.3, β=1.0), fixed final checkpoint | WER 0.2915→0.1631 (-0.1284; -44.1%); CER 0.2317→0.1298 (-0.1019; -44.0%); mean latency 0.204→0.292 s/sample | - |
| phoasr-vs-quan-domain-v1-val309 | Fine-tuned models, paired on the same 309 validation IDs | QUAN WER/CER 0.1631/0.1298 vs PhoASR 0.2576/0.1838; per-utterance WER wins QUAN 224, PhoASR 45, ties 40 | - |
| qwen35-track-c-quan-ft-beam32-public-test | Qwen3.5 Track C + QUAN LifeMate FT + KenLM CTC beam32; full 634-row public-test submission | validation 634/634, 634 unique, 0 schema errors; 0 runtime failures; 5.93h | **26.50** |
