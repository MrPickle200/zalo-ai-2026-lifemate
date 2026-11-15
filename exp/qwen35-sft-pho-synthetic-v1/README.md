# Qwen3.5 4B — PhoWhisper + synthetic direct-text SFT

## Mục tiêu

Tạo tập SFT gấp đôi số dòng gốc bằng paraphrase tiếng Việt được viết trực tiếp. Mỗi trajectory train BTC có một utterance synthetic đầy đủ; không sinh câu bằng code và không chỉ thêm tiền tố lịch sự. Các nhãn tool, history và observation được giữ nguyên.

## Kết quả

- 1.236 utterance synthetic phủ đủ fixed BTC train split.
- 3.873 dòng gốc dùng transcript PhoWhisper Large.
- 3.873 dòng synthetic cùng nhãn, context và observations.
- `output/data/combined_sft_7746.jsonl`: tập trộn 7.746 dòng, tỷ lệ synthetic:gốc 1:1.
- ToolEnv BTC replay: 1.236/1.236 trajectory chạy được, final state khớp nhãn; không có lỗi.

## Artifact chính

- `output/data/synthetic_utterances_1236.jsonl`: câu gốc PhoWhisper và câu synthetic theo sample ID.
- `output/data/btc_train_1236.jsonl`: trajectory train BTC.
- `output/data/original_phowhisper_sft_3873.jsonl`
- `output/data/synthetic_direct_sft_3873.jsonl`
- `output/data/combined_sft_7746.jsonl`
- `output/data/toolenv_replay_train_1236.jsonl`: audit replay ToolEnv.
- `output/data/dataset_manifest.json`: số lượng, phương pháp và checksum.
- `output/colab-resources/lifemate-qwen35-pho-synthetic-direct-sft-v1.zip`: gói nhỏ để tải lên Colab, không kèm audio/model weights.

## Dựng lại

PhoWhisper train inference đã hoàn thành và lưu ở `output/asr/train_audio_transcripts.jsonl`. Từ thư mục gốc dự án, chạy:

```powershell
.venv\Scripts\python.exe exp/qwen35-sft-pho-synthetic-v1/scripts/build_direct_dataset.py
.venv\Scripts\python.exe exp/qwen35-sft-pho-synthetic-v1/scripts/build_sft_bundle.py
```

`build_direct_dataset.py` dùng fixed 1.236 train IDs, không lấy validation/test vào training; sau khi tạo dataset, script replay mỗi trajectory đúng một lần qua ToolEnv chính thức. Không chạy fine-tune trong bước chuẩn bị này.

## Giới hạn kiểm định

ToolEnv xác nhận các gold tool calls kế thừa hợp lệ và kết thúc đúng state BTC; nó không đánh giá độ tương đương ngữ nghĩa giữa câu tự nhiên và nhãn. Câu synthetic được viết và rà soát trực tiếp theo transcript/context train. Nên đánh giá riêng trên validation cố định sau khi SFT.

## Benchmark adapter targeted 1 epoch và Public LB

- Adapter 1 epoch đạt **50.5423/100** theo BTC formula trên fixed val309 (baseline 42.7067; checkpoint 120 bước 46.6729).
- Tạo candidate `outputs/submissions/phowhisper-large-v2-qwen35-val309-benchmark-winner-focused-public-test.jsonl`: 634 ID duy nhất, 0 schema/runtime errors. Người dùng báo candidate đạt **51.19 Public LB**; chưa xác minh độc lập qua leaderboard.
- Submission dùng thinking bật cố định cho mọi request (`thinking: true`), không dùng adaptive switching; JSON, temperature 0, cap768, context8192, batch512, seed17, tối đa 5 action turns. Chi tiết khóa cấu hình và checksum nằm trong `output/submission-adapter-benchmark-winner/` và submission manifest.

## Benchmark adapter 3 epochs trên PhoWhisper val309

- Adapter đạt **50.1293/100 sau runtime gate 30 giây** (56.8124 trước gate), so với 50.5423 của adapter 1 epoch trên cùng val309.
- Chênh lệch 3 epoch − 1 epoch là **−0.4130 điểm**, paired bootstrap 95% CI **[−4.2690, +3.3449]**; chưa có bằng chứng 3 epoch tốt hơn. Adapter 3 epoch có 30/47 gold ask chính xác so với 23/47, nhưng điểm BTC tổng thể thấp hơn nhẹ.
- So với baseline tốt nhất, adapter 3 epoch tăng **+7.4226 điểm** (95% CI [+3.0690, +11.8231]). Độ trễ trung bình local 18.50 giây so với 18.15 giây của adapter 1 epoch; đo trên RTX 4050, không phải A30.
- Cả bốn nhánh dùng prompt focused post-lookup, PhoWhisper Large v2 transcript đã cache, thinking bật cố định, JSON, temperature 0, `num_predict=768`, `num_ctx=8192`, batch 512, seed 17 và tối đa 5 action turns. Mỗi nhánh có một lỗi context overflow ở `train_1725` (8.265 > 8.192 tokens).
- Đây là benchmark val309; không tạo submission. Báo cáo và paired details: `output/adapter-final-3epoch-benchmark/val309/summary.md`, `summary.json`, `paired_details.jsonl`.

## Adaptive thinking: screen hai adapter trên cohort 120

- Cùng cohort stress 120 mẫu của adaptive screen trước (42 mẫu step120 chậm + 78 control), cùng prompt và cấu hình. Thinking bật ở request đầu; sau khi tổng latency request của mẫu vượt 20 giây, tắt thinking cho các request tiếp theo. Fixed-thinking predictions được lấy từ benchmark val309 tương ứng.
- Adapter 1 epoch: BTC sau gate cục bộ 30 giây **46.07 → 52.88** (+6.81; paired 95% CI [+2.81,+11.37]); mẫu >30 giây **30 → 13**; mean latency **23.88 → 22.47s**. BTC trước gate giảm 56.74 → 54.54.
- Adapter 3 epoch: BTC sau gate **41.50 → 47.12** (+5.62; paired 95% CI [+2.20,+9.63]); mẫu >30 giây **37 → 17**; mean latency **25.35 → 22.49s**. BTC trước gate giảm 57.32 → 53.59.
- Không có lỗi runtime. Thinking được tắt cho 70/498 request ở adapter 1 epoch và 71/502 request ở adapter 3 epoch.
- Đây là cohort stress có chủ đích, không đại diện val309/public. Không dùng kết quả này để thay điểm full val309; nếu chọn adaptive làm cấu hình mới, cần chạy full val309 trước khi tạo submission.
- Báo cáo: `output/runtime-budget-adaptive-adapters/summary.md`; predictions, paired details và lock ở cùng thư mục. Script: `scripts/runtime_budget_adaptive_adapters.py`.

## Benchmark adapter 4 epochs vs 3 epochs

- Adapter adapter_final-4-epochs.zip was merged into the Qwen3.5-4B base, converted to Q4_K_M GGUF, registered and served with Ollama. Tag: lifemate-qwen35-phosynth-targeted-4epoch-q4km:latest; digest: 528597f6f48cae627a54057b424b48cc479330d5b0721d2a840578caf20f40bf.
- Fixed-thinking BTC val309: final4 scored **60.1475 before** and **53.7904 after** the local 30-second gate; final3 scored **56.8124 / 50.1293**. Final4−final3 was +3.3351 raw points (paired 95% CI [+0.1577,+6.5020], wins/ties/losses 39/248/22) and +3.6611 after gate (CI [+0.2368,+7.1146]). Exact gold asks: 38/47 vs 30/47. One shared 8,265-token context overflow occurred on train_1725 in each branch using num_ctx=8192.
- Report and locked inputs: output/adapter-final-4epoch-benchmark/val309/summary.md, summary.json, paired_details.jsonl, comparison_lock.json. Runtime gate and latency use local RTX 4050, not competition A30.

## Final4 public-test candidate

- Chosen over final3 by full-thinking, pre-gate val309 score. Candidate: outputs/submissions/phowhisper-large-v2-qwen35-phosynth-final4-val309-winner-focused-public-test.jsonl; SHA-256 4f0bc5953cd79c76b6444c297d845927eb8debdac8ea3fc6096456a006b6c2f5.
- Validated 634/634 rows, 634 unique IDs, 0 schema errors and 0 runtime errors. Used Q4_K_M Ollama tag above, PhoWhisper Large v2 for main/history, focused prompt (SHA-256 88b0264bd512a90368d0a64aa2241ff59da2660790c0725781358986bb140785), fixed thinking-on, JSON, temperature 0, cap768, batch512, seed17, max five action turns.
- Submission context was set to 16,384 (the BTC limit) to avoid the shared 8,265-token val309 overflow; all other generation settings stayed fixed. Local Qwen+ToolEnv mean latency was 20.90s and 117/634 exceeded 30s; estimated ASR+Qwen mean was 26.50s and 216/634 exceeded 30s. These are RTX 4050 diagnostics, not A30 measurements. The user later reported Public LB **53.47** for this final4 candidate (not independently verified), above the earlier user-reported 51.19 from the 1-epoch adapter.
- Lock, selection record, manifest and provenance: output/submission-final3-vs-final4/.
## Prompt-only policy screen after ask-stop

- Ran three additive system-prompt instructions on the same enriched pilot60 with the targeted final4 adapter, fixed thinking, and the successful sys.ask_missing_info stop guard. The arms targeted ask precision, completion after lookup, and grounding IDs/numbers/optional arguments.
- BTC before/after local 30s gate: ask precision 63.89/62.22, completion audit 60.78/60.78, argument grounding 62.22/62.22. Archived ask-stop baseline: 63.33/63.33. Paired gated deltas were −1.11 [95% CI −10.00,+7.78], −2.56 [−10.00,+5.00], and −1.11 [−9.44,+7.22], respectively. All intervals include zero; keep the existing prompt/policy.
- Descriptive gold-ask/no-gold-ask deltas were +5.33/−5.71, +0.53/−4.76, and +9.33/−8.57. The pilot is enriched, so these subgroup results are not full-val or Public LB estimates.
- All 180 inference samples completed with zero runtime errors. Latency is local RTX 4050, not A30. No candidate was generated. Report and logs: output/policy-prompt-screen-val60/; script: scripts/run_policy_prompt_screen_val60.py.

## Final4 Q8_0 merge and pilot60

- Built Q8_0 from the locally verified Qwen3.5-4B base plus targeted adapter_final-4-epochs.zip. GGUF: output/adapter-final-4epoch-q8-benchmark/gguf/adapter-final-4epoch-Q8_0.gguf (SHA-256 ed0f2be012e71e94775be5dfa7375bfb6205d29c336e7f8ebb9ccfd99f2f231a). Ollama tag: lifemate-qwen35-phosynth-targeted-4epoch-q8-0:latest.
- Entry points: scripts/build_final4_q8.py, scripts/register_ollama_final4_q8.py, scripts/benchmark_final4_q8_pilot60.py. Benchmark used cached PhoWhisper Large v2, focused prompt, fixed thinking, ask-stop, JSON, temperature 0, cap768, context8192, batch512, seed17, max five turns.
- Official BTC enriched pilot60: Q8 59.22 before / 38.89 after local 30s gate; archived Q4 ask-stop control 63.33/63.33. Q8−Q4 raw −4.11 points (95% paired bootstrap CI [−14.00,+5.33]); gated −24.44 (CI [−36.67,−12.78]). Q8 60/60 with zero runtime errors. On local RTX 4050, Ollama reported 74% GPU / 26% CPU residency; mean latency 29.38s, 23/60 above 30s. Local timings do not represent A30.
- Keep Q4 public baseline; no Q8 submission. The 60-task cohort is enriched and the Q4 control archived. Report, predictions, paired details, comparison lock and provenance: output/adapter-final-4epoch-q8-benchmark/.
## Ask-stop with a six-turn cap — pilot60

- Ran final4 Q4 on the enriched pilot60 with fixed thinking, focused prompt, and the successful sys.ask_missing_info stop guard. Raised the action-turn cap from five to six.
- All 60 completed without runtime errors, but **none used a sixth turn**. Thus the test does not estimate the benefit of turn six.
- Official BTC: archived max-five ask-stop 63.33/63.33 before/after local 30s gate; fresh max-six run 61.67/61.67. Paired delta −1.67 (95% CI [−5.00,0.00]); 0 wins, 59 ties, 1 loss. Calls matched on 57/60 and hashes on 59/60; this reflects fresh-run variation.
- Do not promote based on this screen. Script and outputs: scripts/benchmark_final4_askstop_max6_pilot60.py and output/ask-stop-max6-pilot60/.
## Ask-stop max-six follow-up on eligible val309 samples

- The enriched pilot60 used no sixth turn. Follow-up selected the 25 archived final4 val309 samples whose first five actions were successful tool calls without a sys.ask_missing_info call; all 25 fresh max-six runs actually used turn six.
- Official BTC before/after local 30s gate: max-five baseline 67.84/35.09; max-six 73.66/13.58. Paired raw +5.82 points (95% CI [−6.40,+18.04]); gated −21.51 (CI [−37.51,−7.20]); 3 wins, 21 ties, 1 loss. Zero runtime errors.
- RTX 4050 mean latency rose 33.07s→38.88s, with 21/25 over 30s. The cohort is targeted and local timing is not A30 timing; keep max-five for now.
- Entry point: scripts/benchmark_final4_askstop_max6_eligible_val309.py
- Report: output/ask-stop-max6-eligible-val309/summary.md
## Adaptive thinking combined with max-six ask-stop

- Screened the final4 Q4 adapter on the same 25 eligible val309 continuation samples as the fixed max-six run. Ask-stop remains active; cap six; adaptive thinking starts on and turns off for later requests after cumulative request latency reaches 20 seconds.
- Official BTC: adaptive max6 scored 67.08 before / 40.69 after the local 30s gate. Archived max5 fixed-thinking scored 67.84 / 35.09; fresh fixed max6 scored 73.66 / 13.58. Adaptive−max5 was −0.76 raw (95% CI [−12.00,+10.31]) and +5.60 gated (CI [−6.40,+18.40]). Adaptive−fixed6 was −6.58 raw (CI [−20.38,+6.33]) and +27.11 gated (CI [+11.56,+44.89]).
- Adaptive mean latency 27.91s; 9/25 over 30s (versus 33.07s and 14/25 for max5; 38.88s and 21/25 for fixed max6). Thinking switched off in 24 samples / 44 requests; 23 samples used the sixth turn; 0 runtime errors.
- Targeted continuation cohort, cached ASR and local RTX 4050 timing; not full-val, Public LB, or A30 evidence. Keep max5 and the public candidate unchanged; adaptive vs max5 quality CI includes zero.
- Script: `scripts/benchmark_final4_askstop_max6_adaptive20_eligible_val309.py`. Full report and artifacts: `output/ask-stop-max6-adaptive20-eligible-val309/`.
## Full val309: adaptive20 + max6 + successful ask-stop

- Compared a fresh final4 Q4 adaptive20/max6 run with the archived fixed-thinking max5 run reconstructed to stop after the first successful `sys.ask_missing_info`. Replayed truncated control calls through official ToolEnv; 47/309 control rows were truncated. Both branches used the same val309 IDs, focused prompt, cached PhoWhisper, and generation settings.
- BTC raw / after local 30s gate: control 63.80/57.88; adaptive 63.11/60.34. Adaptive−control raw −0.69 (paired 95% CI [−1.94,+0.45]); gated +2.46 ([+0.96,+4.27]). Over 30s: 31→15; mean latency 16.49s→15.85s. One shared context overflow; remaining rows completed without runtime errors.
- Adaptive thinking switched off in 54 samples / 83 requests; 7/309 used six tool calls. Supports this as a runtime-oriented option, not as raw-score winner; local RTX 4050 timings with cached ASR are not A30 evidence. Selected fixed max5 ask-stop for submission by pre-gate BTC.
- Selected public candidate is schema-valid at 634/634 and byte-identical to the existing user-reported 55.92 candidate. Script: `scripts/benchmark_final4_askstop_max6_adaptive20_val309.py`; report/artifacts: `output/ask-stop-max6-adaptive20-val309/`; selected candidate manifest: `../../outputs/submissions/phowhisper-large-v2-qwen35-phosynth-final4-val309-max5-fixed-askstop-public-test_manifest.json`.

## Adaptive total ASR+Qwen budget and max-turn full val309

- Compare 15s and 20s end-to-end adaptive budgets on all 309 validation IDs; PhoWhisper per-task timing is included. Score nested action-turn caps 4–7 from the same trajectory; stop after successful `sys.ask_missing_info`.
- Entry point: `scripts/run_adaptive_totalbudget_maxturn_fullval309.py`.
- Result: 15s/max6 scored 61.64 raw / 60.56 after the local 30s gate; 20s/max6 scored 61.83 / 58.30. Max7 added no score. Timing is local RTX 4050 and excludes ASR model-load startup.
- Outputs: `output/adaptive-totalbudget-maxturn-fullval309/`.

## Final5 20k-row adapter — merge and adaptive15/max6 cohort60

- Merged `adapter_final-5-epochs-20k-rows.zip` with the verified Qwen3.5-4B base; converted the result to Q4_K_M GGUF (2,708,804,096 bytes, SHA-256 `ae5c220611b6641307aa10c0173ef7431ccbf2e214ad1886710c7bfbf8543365`) and registered `lifemate-qwen35-phosynth-final5-20k-q4km:latest`.
- Paired official BTC ToolEnv benchmark on the same enriched pilot60 as the final4 control. Both used cached PhoWhisper timing, adaptive 15s total ASR+Qwen budget, max6 action turns, and successful ask-stop. Final4 scored 63.00 raw / 61.33 after the local ASR+Qwen 30s gate; final5 scored 59.67 / 59.67. Paired final5−final4: −3.33 raw [95% CI −11.67,+5.00], −1.67 gated [−10.00,+6.67], 3 wins / 53 ties / 4 losses. Both completed 60/60 without runtime errors; hash matches 50/60 each.
- This is an enriched diagnostic cohort, not full-val or Public LB evidence. Inference ran on local RTX 4050, not A100/A30; do not promote this adapter from the pilot alone. Keep the final4 baseline and user-reported 56.16 public candidate unchanged.
- Benchmark script: `scripts/benchmark_adapter_final5_20k_adaptive15_pilot60.py`. Report and per-task artifacts: `output/adapter-final5-20k-adaptive15-pilot60/`.
