from __future__ import annotations

import json
from pathlib import Path

import _bootstrap  # noqa: F401


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = _bootstrap.ROOT / "exp" / "quan-continuation-cot-v1" / "output" / "asr"
    baseline = read(root / "checkpoint-verification-rerun-val309" / "summary.json")
    epochs = [read(root / f"epoch-{i:02d}-val309" / "summary.json") for i in (1, 2)]
    analyses = [read(root / f"epoch-{i:02d}-paired.json") for i in (1, 2)]
    previous_wer = baseline["metrics"]["corpus_wer"]
    sequence = []
    for index, (summary, analysis) in enumerate(zip(epochs, analyses), 1):
        metrics = summary["metrics"]
        wer, cer = metrics["corpus_wer"], metrics["corpus_cer"]
        sequence.append({"epoch": index, "checkpoint": f"output/checkpoints/epoch-{index:02d}",
                         "model_sha256": summary["model_sha256"], "corpus_wer": wer, "corpus_cer": cer,
                         "wer_delta_vs_previous": wer - previous_wer,
                         "paired_utterance_wer_delta": analysis["paired"]["wer_delta_mean_candidate_minus_baseline"],
                         "paired_utterance_wer_95pct_ci": analysis["paired"]["wer_delta_95pct_bootstrap_ci"],
                         "paired_utterance_cer_delta": analysis["paired"]["cer_delta_mean_candidate_minus_baseline"],
                         "paired_utterance_cer_95pct_ci": analysis["paired"]["cer_delta_95pct_bootstrap_ci"],
                         "utterance_wer_wins": analysis["paired"]["utterance_wer_wins"],
                         "mean_latency_seconds": metrics["mean_inference_seconds"],
                         "numeric_time_negation_exactness": analysis["critical_slot_exactness"]})
        previous_wer = wer
    consecutive_worse = len(sequence) >= 2 and all(row["wer_delta_vs_previous"] > 0 for row in sequence[-2:])
    result = {
        "metric_priority": "corpus WER; corpus CER as tie-breaker when WER is near-equal",
        "baseline_checkpoint": "exp/quan-transcribe-lifemate-domain-v1/output/model",
        "baseline_model_sha256": baseline["model_sha256"],
        "baseline_corpus_wer": baseline["metrics"]["corpus_wer"],
        "baseline_corpus_cer": baseline["metrics"]["corpus_cer"],
        "evaluated_epochs": sequence,
        "early_stop": consecutive_worse,
        "early_stop_reason": "Two consecutive validation WER increases over the preceding checkpoint; CER also increased at both epochs." if consecutive_worse else None,
        "selected_checkpoint": "exp/quan-transcribe-lifemate-domain-v1/output/model",
        "selected_reason": "No added epoch improved WER/CER; use the reproducibly verified current checkpoint as the step-2 control.",
        "epoch_03_run": False,
        "validation_was_used_for_checkpoint_selection_only_after_each_epoch": True,
    }
    path = root.parent / "checkpoint-selection.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
