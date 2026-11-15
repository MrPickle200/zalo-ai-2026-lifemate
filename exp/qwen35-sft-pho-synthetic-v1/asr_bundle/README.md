# PhoWhisper Large train-only inference bundle

ZIP này chỉ phục vụ phiên âm 4.035 audio thuộc 1.236 ID train cố định bằng PhoWhisper Large v2. Gói gồm model weights, audio train, manifest tối thiểu và runner có checkpoint/resume. Không có nhãn BTC, validation, test hoặc dữ liệu SFT.

## Chạy trên Colab L4

1. Upload ZIP lên Drive, copy ZIP vào `/content`, rồi giải nén vào `/content/phowhisper-train` để model và audio được đọc từ ổ cục bộ.
2. Cài phụ thuộc nếu runtime chưa có: `%pip install -q transformers av safetensors` (Colab đã có PyTorch).
3. Chạy:

```bash
python /content/phowhisper-train/scripts/run_train_asr.py \
  --model-dir /content/phowhisper-train/model/pho-whisper-large-zaloai \
  --manifest /content/phowhisper-train/assets/train_audio_manifest.jsonl \
  --audio-root /content/phowhisper-train/assets/audio \
  --output /content/drive/MyDrive/lifemate/train_audio_transcripts.jsonl
```

Runner dùng CUDA, tiếng Việt, chế độ `transcribe`, greedy deterministic decode. Nó lưu checkpoint mỗi 25 audio và tiếp tục từ checkpoint khi chạy lại. Kết quả có `key`, `id`, `kind`, `history_index`, `audio`, `transcript`, `inference_seconds`; có thể dùng làm input ASR cho thí nghiệm SFT sau khi copy về project.

## Nội dung

- 1.236 ID train cố định; 1.236 audio chính và 2.799 audio lịch sử.
- PhoWhisper Large v2 model files và license đi kèm.
- Audio được giữ nguyên đường dẫn tương đối trong manifest.
- Không đóng gói `train.jsonl`, nhãn/tool trajectory, transcript val309, validation audio hoặc test data.
