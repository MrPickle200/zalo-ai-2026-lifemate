# Zalo AI Challenge 2026 — LifeMate

## 1. Tổng quan

**LifeMate** là một track của Zalo AI Challenge 2026, tập trung vào bài toán xây dựng **AI agent có khả năng hiểu yêu cầu bằng giọng nói tiếng Việt và thực thi hành động thông qua các tool được cung cấp**.

Agent không chỉ cần nhận dạng lời nói hay phân loại intent, mà còn phải:

- hiểu ngữ cảnh người dùng;
- sử dụng conversation history khi cần;
- đọc schema của các tool khả dụng;
- lập kế hoạch nhiều bước;
- gọi tool theo đúng thứ tự;
- quan sát kết quả của tool trước để quyết định bước tiếp theo;
- xử lý trường hợp thiếu thông tin;
- đưa simulated environment về đúng trạng thái cuối cùng.

Có thể hình dung pipeline tổng quát:

```text
Audio
  ↓
Speech Recognition / Speech Understanding
  ↓
Context + Profile + Conversation History
  ↓
LLM / Agent Planner
  ↓
Tool Selection + Argument Grounding
  ↓
Tool Execution
  ↓
Observation
  ↓
Next Action / Final State
```

Trang cuộc thi:

- https://challenge.zalo.ai/portal/lifemate

---

## 2. Input và Output

Mỗi sample có thể bao gồm:

- audio tiếng Việt;
- user context, ví dụ thời gian, timezone, vị trí;
- user profile;
- conversation history;
- danh sách app/tool đang khả dụng;
- JSON schema của từng tool;
- simulated environment.

Agent cần sinh một **ordered list of tool calls** và thực thi chúng trong environment.

Ví dụ:

```json
{
  "id": "s_000123",
  "tool_calls": [
    {
      "name": "smart_home.get_status",
      "arguments": {}
    },
    {
      "name": "smart_home.climate_set",
      "arguments": {
        "device_id": "ac_living_room",
        "mode": "cool",
        "temperature": 26
      }
    }
  ],
  "state_hash": "..."
}
```

Thứ tự tool call có ý nghĩa vì output của một bước có thể cần cho bước sau.

---

## 3. Các hành vi chính cần hỗ trợ

### 3.1. Đủ thông tin

Nếu yêu cầu có đủ dữ kiện, agent cần thực thi tool phù hợp.

```text
"Bật điều hòa phòng khách 26 độ."
→ resolve thiết bị
→ gọi tool điều khiển điều hòa
```

### 3.2. Thiếu thông tin

Nếu thiếu tham số bắt buộc và không thể suy ra an toàn từ context, agent không được tự đoán.

Agent cần dùng cơ chế hỏi thêm thông tin, ví dụ:

```text
sys.ask_missing_info
```

### 3.3. Không có tool phù hợp

Nếu yêu cầu nằm ngoài khả năng của các app/tool được cấp, agent không được gọi một tool gần giống chỉ để “cố xử lý”.

Trong trường hợp đó, output tương ứng là không có tool call:

```json
[]
```

---

# 4. Technical Requirements

Đây là phần quan trọng nhất khi thiết kế solution.

## 4.1. Giới hạn kích thước mô hình

Toàn bộ hệ thống phải có **ít hơn 6B parameters**.

Tổng này bao gồm:

- ASR;
- LLM;
- các neural model phụ trợ;
- bất kỳ component học máy nào khác.

Riêng **LLM phải nhỏ hơn 5B parameters**.

Ví dụ, nếu dùng:

```text
LLM: 4.0B
ASR: 1.5B
```

thì tổng là 5.5B và vẫn nằm trong giới hạn.

Ngược lại:

```text
LLM: 4.5B
ASR: 1.8B
```

tổng 6.3B → không hợp lệ.

### Context length

LLM phải có context length tối đa:

```text
16K tokens = 16,384 tokens
```

Không được dựa vào context window lớn hơn trong solution chính thức.

---

## 4.2. Data access

Agent **không được đọc trực tiếp `state.json`**.

Mọi thao tác đọc hoặc ghi dữ liệu của environment đều phải đi qua execution environment do BTC cung cấp.

Không được:

```python
open("state.json")
```

hoặc parse/copy trực tiếp file state.

Mục đích là bắt buộc agent phải tương tác với thế giới thông qua tool interface, đúng bản chất của bài toán agent.

---

## 4.3. Không được đọc trực tiếp environment files

Solution hoặc agent **không được đọc, copy hoặc parse bất kỳ file nào dưới**:

```text
<split>_environment/
```

bao gồm cả:

```text
state.json
```

Toàn bộ app data phải được truy cập thông qua `zaic_tools` của BTC.

---

## 4.4. ToolEnv

Mỗi sample phải sử dụng:

```text
exactly one ToolEnv
```

Không được tạo nhiều environment độc lập cho cùng một sample để thử nhiều hướng rồi chọn kết quả tốt nhất.

Ngoài ra:

> Mọi tool call thực sự được gọi trên ToolEnv phải xuất hiện trong submission.

Không được gọi tool “ẩn” để thu thập thông tin rồi chỉ submit một phần sequence.

Ví dụ, nếu runtime thực hiện:

```text
1. smart_home.get_status
2. contacts.search
3. ride.book
```

thì cả ba call phải được phản ánh trong `tool_calls` của submission.

---

# 5. Quy định về dữ liệu và model

## 5.1. Public data và open-source model

Đội thi có thể sử dụng:

- public data;
- open-source models;

với điều kiện chúng đã được phát hành **trước ngày 26/10/2026** và license cho phép sử dụng.

Cần kiểm tra license trước khi đưa model/data vào solution.

---

## 5.2. Synthetic data

Trong giai đoạn development, đội thi có thể sử dụng model hoặc service khác để tạo synthetic data.

Ví dụ:

```text
LLM API
TTS service
external models
```

có thể được dùng để sinh thêm dữ liệu huấn luyện.

Tuy nhiên, các external service/model này **không được sử dụng tại inference**.

---

## 5.3. Cấm dùng labeled evaluation data để train/tune

Không được dùng labeled data từ:

- public test;
- private test;
- các tập test/evaluation tương tự;

để train hoặc tune solution.

Quy định này áp dụng cả khi label được tạo bằng:

- con người;
- external API;
- LLM khác;
- dịch vụ bên ngoài.

Nói cách khác, không được biến test set thành training set thông qua manual annotation hoặc pseudo-label từ dịch vụ ngoài.

---

# 6. Runtime và Hardware

Mỗi sample phải được xử lý trong:

```text
≤ 30 giây
```

Nếu sample:

- timeout;
- crash;
- không execute thành công;

thì sample đó nhận:

```text
score = 0
```

## Hardware chấm điểm

Solution được chạy trên:

```text
GPU: 1 × NVIDIA A30
CPU: 16-core Intel Xeon Gold 6442Y
RAM: 64 GB
```

Do đó solution cần được tối ưu cho **single-GPU inference**.

Không nên thiết kế pipeline giả định:

- multi-GPU;
- API cloud;
- cluster inference;
- GPU có VRAM lớn hơn đáng kể.

---

# 7. Offline inference

Solution phải chạy **hoàn toàn offline**.

Trong quá trình inference không được gọi:

- OpenAI API;
- Gemini API;
- Claude API;
- Hugging Face hosted inference;
- web service nội bộ hoặc bên ngoài;
- search engine;
- external database/service qua network.

Pipeline cuối cùng cần có dạng:

```text
Input
  ↓
Local ASR
  ↓
Local LLM / local logic
  ↓
BTC ToolEnv
  ↓
Output
```

Không có external network dependency.

---

# 8. Reproducibility

BTC yêu cầu solution có tính deterministic cao.

Cần cố định random seed cho các thư viện liên quan, ví dụ:

```python
random.seed(SEED)
numpy.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

Nên bật các tùy chọn deterministic phù hợp của PyTorch nếu không ảnh hưởng nghiêm trọng tới runtime.

---

## 8.1. Deterministic decoding

ASR và LLM phải sử dụng decoding deterministic.

Ví dụ:

```text
greedy decoding
temperature = 0
do_sample = False
```

Nên tránh:

```text
temperature > 0
top-p sampling
top-k sampling
random beam selection
```

nếu chúng làm output thay đổi giữa các lần chạy.

---

## 8.2. Re-run policy

Nếu BTC chạy lại solution và kết quả khác submission, BTC có thể chạy Docker image:

```text
5 lần
```

sau đó dùng **average score** làm official score.

Vì vậy nondeterminism có thể trực tiếp làm giảm điểm.

---

# 9. Tie-breaking

Nếu hai đội có cùng final score, BTC áp dụng tie-breaking theo thứ tự.

### Bước 1 — Tool-call F1

BTC re-score `tool_calls` bằng F1.

Trong bước này:

```text
state_hash bị bỏ qua
```

Đội có tool-call F1 cao hơn được xếp trên.

### Bước 2 — Runtime

Nếu vẫn hòa, đội có:

```text
average processing time per sample thấp hơn
```

sẽ được xếp hạng cao hơn.

Như vậy tốc độ không chỉ là điều kiện để tránh timeout mà còn có thể ảnh hưởng thứ hạng cuối.

---

# 10. Implication cho thiết kế hệ thống

Với các ràng buộc trên, một kiến trúc thực tế có thể là:

```text
                 ┌────────────────┐
Audio ──────────►│ Lightweight ASR│
                 └───────┬────────┘
                         │
                         ▼
                 ┌────────────────┐
Context ────────►│                │
History ────────►│  LLM < 5B      │
Tools ──────────►│  Agent / Planner│
                 │                │
                 └───────┬────────┘
                         │
                         ▼
                 ┌────────────────┐
                 │ Tool Executor  │
                 │   ToolEnv      │
                 └───────┬────────┘
                         │
                         ▼
                   Final State
```

Một cấu hình hợp lý để bắt đầu có thể là:

```text
LLM: ~3B–4B
ASR: lightweight / medium-size
Rule-based logic: không đáng kể về parameter count
```

Thay vì dùng LLM sát ngưỡng 5B, nên để lại budget cho ASR và các component khác, đồng thời giảm latency trên A30.

---

# 11. Những lỗi dễ phạm rule

Checklist cần tránh:

- đọc `state.json` trực tiếp;
- scan hoặc parse `<split>_environment/`;
- tạo nhiều `ToolEnv` cho một sample;
- gọi tool nhưng không ghi call đó vào submission;
- dùng tổng model ≥ 6B parameters;
- dùng LLM ≥ 5B parameters;
- dùng context > 16K;
- gọi external API khi inference;
- để sample chạy quá 30 giây;
- dùng stochastic decoding;
- không cố định seed;
- train/tune bằng labeled test data;
- dùng model/data phát hành sau 26/10/2026;
- dùng model/data có license không cho phép.

---

# 12. Checklist trước khi submit

```text
[ ] Total parameters < 6B
[ ] LLM parameters < 5B
[ ] Context length <= 16,384
[ ] Inference hoàn toàn offline
[ ] Không có network dependency
[ ] <= 30s / sample trên NVIDIA A30
[ ] Exactly one ToolEnv / sample
[ ] Không đọc state.json trực tiếp
[ ] Không đọc/copy/parse <split>_environment/
[ ] Mọi ToolEnv call đều xuất hiện trong submission
[ ] Fixed random seeds
[ ] Deterministic ASR decoding
[ ] Deterministic LLM decoding
[ ] Không dùng labeled test data để train/tune
[ ] Model/data hợp lệ theo cutoff 26/10/2026
[ ] Kiểm tra license của toàn bộ model và dataset
```

---

# 13. Tóm tắt

LifeMate là bài toán **Vietnamese voice agent + tool use + state management**.

Điểm khó không chỉ nằm ở ASR hay LLM mà ở khả năng:

```text
nghe đúng
→ hiểu đúng
→ chọn đúng tool
→ resolve đúng argument
→ gọi đúng thứ tự
→ quan sát đúng
→ đạt đúng final state
```

Trong khi đó, solution phải đồng thời thỏa các ràng buộc khá chặt:

```text
< 6B total parameters
< 5B LLM
<= 16K context
offline only
<= 30s/sample
1 × NVIDIA A30
deterministic
ToolEnv-only data access
```

Do đó mục tiêu hợp lý là xây một agent nhỏ, ổn định, deterministic và tối ưu tốt cho tool-use, thay vì chỉ ưu tiên model lớn.
