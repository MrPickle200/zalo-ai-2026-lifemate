# TASK.md

## Mục tiêu session

Benchmark nhanh một số mô hình ngôn ngữ **<= 5B tham số** làm agent policy cho LifeMate, nhằm kiểm tra xem **Qwen3-4B có đang bị giới hạn bởi khả năng hiểu tiếng Việt / instruction following / tool calling hay không**.

Không fine-tune trong task này.

Mục tiêu chính:

> Giữ cố định ASR, prompt, tool schemas, ToolEnv, turn cap và evaluator; chỉ thay base LLM để so sánh zero-shot.

Constraint cuộc thi:

- model tối đa **5B parameters**;
- context sử dụng tối đa **16K tokens**.

---

# Models cần benchmark

Ưu tiên theo thứ tự:

1. `MiniCPM3-4B`
2. `Phi-4-mini-instruct`
3. `Gemma-3-4B-IT`
4. `Qwen2.5-3B-Instruct`
5. `Sailor2-1B-Chat`

Baseline bắt buộc:

6. `Qwen3-4B-Instruct`

Nếu thiếu thời gian, ưu tiên hoàn thành:

```text
Qwen3-4B
vs
MiniCPM3-4B
vs
Phi-4-mini-instruct
```

Không chạy model >5B.

---

# Dataset benchmark

Dùng đúng **30 hard samples** đã dùng trong first-action oracle diagnostic sáng nay.

Không thay sample IDs.

Không chọn lại sample theo kết quả model.

---

# Task 1 — Đọc project instructions

Trước khi sửa code:

1. đọc `AGENTS.md`;
2. đọc `COMPETITION.md`;
3. đọc `INSIGHTS.md`, `WORKLOG.md`, `PROGRESS.md`;
4. inspect:
   - `src/lifemate/pipeline.py`
   - `src/lifemate/agent.py`
   - PhoASR integration
   - ToolEnv
   - evaluator
   - scripts oracle diagnostic hiện có.

Ưu tiên reuse code hiện tại thay vì tạo pipeline riêng không tương thích.

---

# Task 2 — Sinh PhoASR transcript đúng một lần và lưu lại

Đây là yêu cầu bắt buộc.

Không chạy PhoASR lại riêng cho từng model.

Tạo script ví dụ:

```text
scripts/build_hard30_phoasr_cache.py
```

Script phải:

1. load đúng 30 hard sample IDs;
2. chạy **PhoASR public** trên audio tương ứng;
3. lưu transcript;
4. lưu metadata cần thiết để kiểm tra sample identity;
5. validate đủ 30 sample;
6. không overwrite cache hợp lệ nếu không có flag rõ ràng.

Output đề xuất:

```text
outputs/benchmarks/hard30_phoasr_cache.jsonl
```

Mỗi row tối thiểu chứa:

```json
{
  "id": "...",
  "transcript": "...",
  "asr_model": "...",
  "audio_ref": "..."
}
```

Nếu project đã có transcript PhoASR public chính xác của 30 sample này từ oracle diagnostic, ưu tiên **reuse và export thành cache canonical** thay vì chạy ASR lại.

Phải verify transcript identity với artifacts oracle trước đó nếu có.

---

# Task 3 — Cache là input canonical cho toàn bộ LLM benchmark

Sau khi cache PhoASR được tạo:

> Tất cả model inference bắt buộc đọc transcript từ cùng một file cache.

Không model nào được tự chạy lại ASR.

Không normalize text khác nhau giữa các model.

Không sửa dấu, punctuation, casing hoặc typo theo từng model.

Mục tiêu là isolate:

```text
LLM effect only
```

---

# Task 4 — Tạo model-agnostic benchmark runner

Tạo script ví dụ:

```text
scripts/run_llm_model_benchmark.py
```

Yêu cầu CLI dạng:

```bash
python scripts/run_llm_model_benchmark.py   --model <model_name_or_path>   --transcripts outputs/benchmarks/hard30_phoasr_cache.jsonl   --output outputs/benchmarks/<model_slug>/
```

Runner phải reuse production logic tối đa có thể.

Giữ cố định:

- system prompt;
- full tool schemas;
- transcript;
- ToolEnv;
- observation serialization;
- parser;
- turn cap = 5;
- decoding config, trừ khi model bắt buộc có format riêng;
- evaluator.

Không dùng:

- gold transcript;
- gold call injection;
- retrieval;
- few-shot examples mới;
- model-specific hints làm thay đổi semantic task.

---

# Task 5 — Adapter cho từng model

Một số model có chat template / function calling format khác nhau.

Được phép viết model adapter mỏng cho:

```text
chat template
tokenizer
special tokens
tool-call serialization
generation API
```

Nhưng semantic information phải giữ tương đương.

Không được cung cấp cho một model thêm thông tin mà model khác không có.

Nếu model không hỗ trợ native tool calling tốt, vẫn phải map output về parser/tool-call representation hiện tại.

Log riêng nếu output không parse được.

---

# Task 6 — Context constraint

Cuộc thi giới hạn context ở **16K**.

Mọi benchmark phải enforce:

```text
max context <= 16384 tokens
```

Không tận dụng context 32K / 128K của model.

Log:

- prompt token count;
- max prompt token count;
- number of samples vượt 16K nếu có.

Nếu prompt vượt 16K, dùng cùng một truncation policy cho mọi model.

---

# Task 7 — Decoding config

Bắt đầu bằng deterministic decoding gần production nhất có thể.

Ưu tiên:

```text
temperature = 0
do_sample = false
```

hoặc equivalent phù hợp model.

Không sweep hyperparameters trong benchmark chính.

Nếu một model yêu cầu generation config đặc biệt để chạy đúng chat template, ghi rõ.

---

# Task 8 — Metrics

Với mỗi model, báo cáo:

```text
tool-call F1
final-state match
proxy
exact trajectory
first full action accuracy
malformed output count
runtime failures
average trajectory length
```

Nếu evaluator hiện tại hỗ trợ, thêm:

```text
first tool accuracy
argument accuracy
second tool accuracy
premature STOP
extra tool calls
```

---

# Task 9 — So sánh trực tiếp với Qwen3-4B

Tạo summary table:

| Model | Params | Tool F1 | State | Proxy | First full call | Exact traj | Malformed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B-Instruct | 4B | | | | | | |
| MiniCPM3-4B | 4B | | | | | | |
| Phi-4-mini-instruct | 3.8B | | | | | | |
| Gemma-3-4B-IT | 4B | | | | | | |
| Qwen2.5-3B-Instruct | ~3B | | | | | | |
| Sailor2-1B-Chat | 1B | | | | | | |

Không kết luận chỉ từ train benchmark công khai.

Ưu tiên kết quả trên LifeMate hard-30.

---

# Task 10 — Vietnamese comprehension diagnostic nhẹ

Nếu còn thời gian, thêm một diagnostic nhỏ để phân biệt:

```text
Vietnamese comprehension
vs
tool-policy failure
```

Không cần tạo benchmark lớn.

Có thể inspect các sample mà:

- Qwen3-4B chọn sai intent/tool;
- model khác chọn đúng tool hoặc arguments;
- transcript chứa entity, time, quantity hoặc relation tiếng Việt khó.

Ghi qualitative examples ngắn vào `INSIGHTS.md`.

Không dùng diagnostic này để thay metric chính.

---

# Task 11 — Artifacts

Mỗi model lưu riêng:

```text
outputs/benchmarks/model-comparison/
  phoasr_cache/
  qwen3-4b/
  minicpm3-4b/
  phi-4-mini/
  gemma3-4b/
  qwen2.5-3b/
  sailor2-1b/
  summary.json
  summary.md
```

Mỗi model output nên giữ:

- per-sample trajectory;
- tool calls;
- observations;
- final state;
- metrics;
- malformed outputs;
- runtime errors;
- token counts;
- runtime / throughput.

---

# Task 12 — Resource management

GPU local hiện đang rảnh nhưng session chỉ khoảng 1–1.5 giờ.

Ưu tiên:

1. hoàn thành canonical PhoASR transcript cache;
2. benchmark Qwen3-4B baseline từ cache;
3. benchmark MiniCPM3-4B;
4. benchmark Phi-4-mini-instruct;
5. chạy các model còn lại nếu còn thời gian.

Không dành phần lớn thời gian để tối ưu dependency cho một model khó chạy.

Nếu model mất quá nhiều thời gian setup, log và skip sang model tiếp theo.

---

# Task 13 — Không ảnh hưởng fine-tune Kaggle đang chạy

Task này độc lập với trajectory SFT đang chạy trên Kaggle.

Không:

- sửa training dataset đang dùng;
- thay checkpoint đang train;
- thay production behavior;
- overwrite artifacts của `qwen-next-action-lora-v1`.

Chỉ reuse code/evaluator khi an toàn.

---

# Task 14 — Tracking

## WORKLOG.md

Ghi ngắn gọn:

- PhoASR cache creation / reuse;
- model download/setup;
- benchmark runs;
- runtime issues;
- outputs.

## INSIGHTS.md

Ghi:

- model nào cải thiện first full action;
- model nào cải thiện tool F1;
- model nào cải thiện final-state;
- malformed differences;
- dấu hiệu Vietnamese comprehension;
- dấu hiệu tool-policy vẫn là bottleneck.

Không thêm vào `PROGRESS.md` trừ khi experiment được coi là candidate chính thức cho submission.

---

# Không làm trong task này

- Không fine-tune model mới.
- Không chạy ASR lại cho từng model.
- Không dùng gold transcript.
- Không inject gold calls.
- Không retrieval.
- Không thay hard-30 IDs.
- Không tăng turn cap.
- Không sweep temperature rộng.
- Không dùng model >5B.
- Không vượt 16K context.
- Không sửa production pipeline ngoài adapter cần thiết cho benchmark.

---

# Báo cáo cuối task

```text
## PhoASR cache

- Samples:
- Cache path:
- Generated new / reused existing:
- Transcript identity verified:
- ASR runtime:

## Models

### Qwen3-4B-Instruct
- Tool F1:
- State:
- Proxy:
- First full call:
- Exact trajectory:
- Malformed:
- Runtime:

### MiniCPM3-4B
- Tool F1:
- State:
- Proxy:
- First full call:
- Exact trajectory:
- Malformed:
- Runtime:

### Phi-4-mini-instruct
- Tool F1:
- State:
- Proxy:
- First full call:
- Exact trajectory:
- Malformed:
- Runtime:

### Other models
...

## Comparison

- Best Tool F1:
- Best final-state:
- Best first-full-call:
- Lowest malformed rate:

## Diagnostic

- Evidence that Vietnamese comprehension is a bottleneck:
- Evidence that tool-policy is the main bottleneck:

## Conclusion

- Is Qwen3-4B still the best <=5B base model for this task?
- Which alternative model should be tested next with trajectory SFT?
```

Kết luận cuối phải trả lời rõ:

> Có model <=5B nào zero-shot tốt hơn Qwen3-4B trên cùng PhoASR transcript và cùng iterative ToolEnv hay không?

và:

> Kết quả có cho thấy khả năng hiểu tiếng Việt của Qwen3-4B là bottleneck đáng kể hay không?
