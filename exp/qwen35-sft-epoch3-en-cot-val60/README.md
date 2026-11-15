# Qwen3.5 SFT epoch 3 (English prompt) + CoT — cohort60

## Mục tiêu

Merge adapter fine-tune thêm một epoch với English system prompt, sau đó so sánh nhánh **adapter epoch 3 tiếng Anh + CoT** trên đúng cohort 60 với hai kết quả đã có: **SFT epoch 2 + CoT (nhánh C)** và **adapter epoch 3 tiếng Việt + CoT**. Hai baseline được xác thực và tái sử dụng từ artifact; không chạy lại chúng.

Cohort là 60 ID cố định thuộc val309, cùng thứ tự/hash `87b925a9671e6131c31ee33c1288372289bda525037cd09f5796c9b1b7fc18e9`. Mỗi prediction tạo đúng một ToolEnv. Không đọc file `*_environment/`; gold được tính bằng scorer/ToolEnv chính thức. Không dùng public/private test.

## Provenance đầu vào

- Adapter: `exp/qwen35-sft-next-action-v1/adapter-3-epochs-en.zip`
- Dataset SFT: `exp/qwen35-sft-next-action-v1/kaggle-upload-prompt-en-v1/` (3.873 hàng; hash được lưu trong manifest)
- Base: Kaggle Qwen3.5-4B version 1, kiểm tra bằng manifest SHA-256 dùng chung với nhánh C
- Adapter report: continuation từ epoch-2 adapter, 1 epoch, LR `1e-5`; report không ghi hash chính xác của base weights dùng trên máy train

## Entry points

1. `scripts/audit_extract.py` — kiểm tra archive/checkpoint, hash adapter nguồn, dataset, base manifest và tính toàn vẹn của cả hai baseline đã lưu. Trích adapter và audit vào `output/model-provenance/`.
2. `scripts/merge_adapter.py --base-dir models/kaggle-qwen3-5-4b-v1/Qwen3.5-4B` — merge adapter bằng PEFT từ base local đã xác minh, kiểm tra logit trước/sau merge. Chạy bằng `.venv\Scripts\python.exe`.
3. `scripts/convert_model.py` — chuyển HF sang GGUF BF16 rồi quantize Q4_K_M với toolchain đã pin từ baseline.
4. `scripts/register_ollama.py` — đăng ký tag riêng `lifemate-qwen35-kagglev1-sft-epoch3-en-q4km:latest`. Ollama phải gọi bằng system Python/host runtime, không chạy trong `.venv`.
5. `scripts/run_epoch3_cot.py --prepare`, `--smoke`, `--run` — khóa config/cohort, chạy một smoke check qua ToolEnv, sau đó inference đúng 60 mẫu với thinking, JSON, temperature 0, cap 768, context 8192, batch 512. Dùng system Python để truy cập Ollama localhost.
6. `scripts/analyze_three_way.py` — replay gold bằng ToolEnv chính thức cho prediction mới, tính BTC và paired bootstrap so với hai baseline đã lưu.
7. `scripts/make_submission_en.py --prepare`, sau đó `--run` — tạo candidate JSONL cho `public_test` bằng model tiếng Anh + CoT, dùng QUAN transcripts beam32 đã khóa. Chạy system Python ngoài `.venv`; script tự kiểm tra Ollama digest, input lock, đúng một ToolEnv/mẫu và schema chính thức. Sau khi bị gián đoạn, `--resume` xác minh checkpoint JSONL/prediction/telemetry/log rồi nối tiếp ID còn thiếu. Các lệnh này chỉ tạo file cục bộ, không gửi lên Public LB.

## Artifact

- Adapter đã audit, training report, merge/conversion/Ollama manifests: `output/model-provenance/`
- HF merged, GGUF BF16/Q4_K_M, conversion log: `output/merged-hf/`, `output/gguf/`, `output/conversion-logs/`
- Prediction, telemetry, ToolEnv trace, per-sample BTC, paired rows, summary: `output/inference/` và `output/summary.md`
- Submission JSONL, prediction sidecar và manifest cuối: `outputs/submissions/qwen35-sft-epoch3-en-cot-quan-beam32-num-predict-768-num-batch-512-public-test*`
- Preflight lock, requests telemetry, execution log và schema validation log: `output/submission/`
- Hai kết quả baseline giữ nguyên ở `exp/qwen35-sft-benchmark-v1/output/inference/` và `exp/qwen35-sft-epoch3-cot-val60/output/inference/`.

BTC chính được báo riêng với diagnostic gate 30 giây trên máy local. Kết quả cohort60 là sàng lọc cố định, không đại diện toàn bộ val309; baseline không được chạy lặp nên không ước lượng phương sai giữa các lượt.

## Kết quả cohort60

Điểm BTC (không áp runtime gate 30 giây local): epoch 2 C **26.00**, epoch 3 prompt tiếng Việt **23.28**, epoch 3 prompt tiếng Anh **27.51**. Chênh ghép cặp của nhánh tiếng Anh là **+1.51 điểm** so với epoch 2 C (CI 95% **[-6.64, +9.56]**) và **+4.23 điểm** so với epoch 3 tiếng Việt (CI 95% **[-2.82, +11.61]**); cả hai khoảng đều chứa 0, nên cohort này chưa xác nhận được cải thiện chắc chắn. Có 0 lỗi runtime và 39/60 hash ToolEnv khớp. Bảng đầy đủ, score diagnostics, paired rows và provenance caveat nằm trong `output/summary.md` và `output/inference/three-way-analysis.json`.
