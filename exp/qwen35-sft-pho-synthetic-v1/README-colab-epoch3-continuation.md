# Colab SFT bundle — epoch 3 continuation

ZIP gồm tập SFT 7.746 dòng, adapter epoch 3 English và provenance. Model nền không kèm theo; notebook tải Qwen/Qwen3.5-4B trực tiếp từ Hugging Face.

Upload ZIP vào MyDrive/lifemate, chọn Colab GPU L4, bật Internet, rồi chạy notebook. Mặc định: 1 epoch, LR 1e-5, QLoRA NF4, LoRA r=16/alpha=32, max length 8192, effective batch 16. Optimizer/scheduler mới.

SFT targets không chứa CoT. CoT checklist và focused post-lookup prompt chỉ thuộc inference pipeline.

Adapter SHA-256: a15c99d66b1bc2c14846428ef0efd84865110d084fbf17f7206517ad69977048
Combined SFT SHA-256: 6203ce35636f65a34b629b0398f12cc2bf1b8bdbd3c266a1abf5a5602ffc16d1

Training report epoch 3 không lưu hash snapshot model nền Kaggle; revision Hugging Face chỉnh được trong CFG.
