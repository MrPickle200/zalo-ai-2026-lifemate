# QUAN fine-tuning and transcript dictionary

## Result

Completed 10 sequential fine-tuning epochs from the same pretrained QUAN checkpoint. Epoch 1 updated the CTC reader; epochs 2–10 updated the reader and the top two EmbeddingGemma encoder layers. Training used seed 17, accumulation 4, reader LR 5e-5, encoder LR 1e-6. Each epoch was evaluated on the fixed 309-ID validation split with KenLM beam width 32 (alpha 0.3, beta 1.0).

| Epoch | Corpus WER | Corpus CER | Note |
|---:|---:|---:|---|
| 0 | 0.291464 | 0.231738 | Pretrained baseline; comparison only |
| 1 | 0.170426 | 0.136743 | |
| 2 | 0.163055 | 0.129808 | |
| 3 | 0.163202 | 0.130082 | |
| 4 | 0.162318 | 0.129945 | |
| 5 | 0.162612 | 0.128485 | |
| 6 | **0.162023** | 0.129215 | **Selected: lowest WER** |
| 7 | 0.162760 | 0.130036 | |
| 8 | 0.162612 | **0.127937** | Lowest CER; higher WER than epoch 6 |
| 9 | 0.162170 | 0.127983 | |
| 10 | 0.164382 | 0.129717 | |

The selected epoch reduces validation WER by 44.41% and CER by 44.24% relative to the pretrained baseline. Checkpoint: `output/checkpoints/epoch-06`; `model.safetensors` SHA-256: `6e12cf493cf0dea3805cefac6fb7bd4c086c48a0f00c4bafa60b10c77608deee`.

The CTC GPU backward kernel emitted PyTorch's nondeterminism warning during training. Results are recorded from the measured run; bitwise reproducibility is not guaranteed.

## Transcripts

The selected checkpoint produced transcripts for all 1,545 train-manifest rows and all 634 public_test rows. Both JSONL files contain only ID, prediction, audio duration, and inference time; references are not copied into transcript artifacts.

- `output/transcripts/train.jsonl`
- `output/transcripts/public_test.jsonl`

The public_test manifest was used for IDs and audio paths only. No public_test labels or environment files were read.

## Train-only dictionary

Dictionary extraction used references and predictions for the 1,236 IDs in `data/splits/train_ids.json`. The 309 validation IDs were excluded. It found 493 candidate pairs: 422 lexical substitutions, 38 number-rendering variants, 26 orthographic variants, and 7 spoken variants. There are 10 high-, 25 medium-, and 458 low-consistency candidates.

The confidence label measures how consistently an alignment produced a pair; it does not establish that the proposed correction is semantically right. Orthographic and number-rendering variants are marked as non-rules. Spoken variants require context review. Lexical candidates marked eligible still require a human decision before use.

Alignment details:

1. Normalize each text as lowercase Unicode `\w+` tokens; punctuation is removed and diacritics are retained.
2. Align reference and prediction with word-level Levenshtein distance. Ties prefer diagonal, then deletion, then insertion.
3. Emit only contiguous runs of diagonal substitutions with equal numbers of source and target tokens. Insertions and deletions are counted separately and never converted into substitution rules.
4. Save up to eight reference/transcript context examples per pair. Confidence is high at >=5 occurrences, >=3 train IDs, and wrong-phrase dominance >=0.8; medium at >=2 occurrences, >=2 IDs, and dominance >=0.6; otherwise low.

This run found 945 substitution operations, 381 insertions, and 328 deletions. Only substitutions enter the candidate list.

- `output/dictionary/train_candidates.json`
- Builder: `scripts/build_train_dictionary.py`

Training targets use the existing CTC normalization (lowercase, punctuation removed, numbers written in Vietnamese words, and common orthography normalized). Validation scoring uses the project's lowercase Unicode word-token normalization for WER and removes whitespace for CER. Dictionary extraction deliberately retains diacritics and does not merge number words with digits; those cases are labeled as formatting variants for review.

## Public-test review

Six context-based correction hypotheses are stored separately in `output/review/public_test_suggestions.jsonl`. Each row preserves the original transcript, identifies the suspicious phrase, gives a proposed correction and reason, and includes supporting train-only candidate counts where available. Every item has status `unaccepted_user_review` and `auto_applied: false`. No transcript or dictionary rule was changed by these suggestions.

## Decoder recovery

Epoch 7 validation initially stopped when a CTC frame contained only a non-decodable unknown unit and KenLM produced an empty beam. The decoder now falls back to the blank unit only when no valid unit remains. Normal frames keep the same candidate set. Epoch 7 validation was rerun in full and epochs 8–10 then completed; all checkpoints were evaluated with the same beam settings.
