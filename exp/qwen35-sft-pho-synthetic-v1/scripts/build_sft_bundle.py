from __future__ import annotations

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "exp/qwen35-sft-pho-synthetic-v1"
DATA = EXP / "output/data"
DEST = EXP / "output/colab-resources/lifemate-qwen35-pho-synthetic-direct-sft-v1.zip"
FILES = (
    "README.md",
    "dataset_manifest.json",
    "synthetic_utterances_1236.jsonl",
    "btc_train_1236.jsonl",
    "original_phowhisper_sft_3873.jsonl",
    "synthetic_direct_sft_3873.jsonl",
    "combined_sft_7746.jsonl",
    "toolenv_replay_train_1236.jsonl",
    "synthetic_quality_audit.json",
)


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_suffix(DEST.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name in FILES:
            source = (EXP / name) if name == "README.md" else (DATA / name)
            if not source.is_file():
                raise FileNotFoundError(source)
            archive.write(source, f"lifemate-qwen35-pho-synthetic-direct-sft-v1/{name}")
    tmp.replace(DEST)
    with zipfile.ZipFile(DEST) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert len(names) == len(set(names))
        combined = [line for line in archive.read("lifemate-qwen35-pho-synthetic-direct-sft-v1/combined_sft_7746.jsonl").splitlines() if line]
        assert len(combined) == 7746
    manifest = json.loads((DATA / "dataset_manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({"archive": str(DEST), "bytes": DEST.stat().st_size, "combined_sft_rows": len(combined), "toolenv_replay_failures": manifest["toolenv_gold_replay"]["failures"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
