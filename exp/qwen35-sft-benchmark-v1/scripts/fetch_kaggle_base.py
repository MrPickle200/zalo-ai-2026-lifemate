from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
MODEL_REF = "limamateus/qwen3-5-4b/pytorch/default/1"
MODEL_DIR = ROOT / "models" / "kaggle-qwen3-5-4b-v1"
PROVENANCE_DIR = ROOT / "exp" / "qwen35-sft-benchmark-v1" / "output" / "model-provenance"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_kaggle(args: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["kaggle", *args], cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode:
        # Retain actionable CLI diagnostics while redacting the credential defensively.
        token = os.environ.get("KAGGLE_API_TOKEN", "")
        detail = (result.stderr or result.stdout)[-1500:]
        if token:
            detail = detail.replace(token, "<redacted>")
        raise RuntimeError(
            f"Kaggle CLI exited {result.returncode} for {args[0:5]}: "
            f"{detail}"
        )
    return result


def load_kaggle_token() -> str | None:
    env_file = ROOT / ".env"
    if env_file.is_file():
        for raw_line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, separator, value = line.partition("=")
            if separator and key.strip() == "KAGGLE_API_TOKEN":
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                    value = value[1:-1]
                if value:
                    os.environ["KAGGLE_API_TOKEN"] = value
                    return "project .env"
    if os.environ.get("KAGGLE_API_TOKEN"):
        return "process environment"
    return None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_extract_model_archive(archive_path: Path, destination: Path) -> int:
    """Extract only ordinary model files, excluding caches and environment trees."""
    destination = destination.resolve()
    extracted = 0
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            raw_name = member.name.replace("\\", "/")
            while raw_name.startswith("./"):
                raw_name = raw_name[2:]
            parts = PurePosixPath(raw_name).parts
            if not parts or raw_name.startswith("/") or any(part in {"", ".", ".."} for part in parts):
                raise RuntimeError(f"Unsafe path in Kaggle model archive: {member.name!r}")
            if ".cache" in parts or any(part.endswith("_environment") for part in parts):
                continue
            if not (member.isdir() or member.isfile()):
                raise RuntimeError(f"Unexpected non-file entry in Kaggle model archive: {member.name!r}")
            target = destination.joinpath(*parts)
            if not target.resolve().is_relative_to(destination):
                raise RuntimeError(f"Archive path escapes target directory: {member.name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read regular archive member: {member.name!r}")
            with source:
                if target.is_file() and target.stat().st_size == member.size:
                    while source.read(1024 * 1024):
                        pass
                    extracted += 1
                    print(f"verified-size-existing={extracted} file={target.relative_to(destination).as_posix()} bytes={member.size}", flush=True)
                    continue
                with target.open("wb") as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
            extracted += 1
            print(f"extracted={extracted} file={target.relative_to(destination).as_posix()} bytes={member.size}", flush=True)
    return extracted


def main() -> int:
    credential_source = load_kaggle_token()
    if not credential_source:
        raise RuntimeError("KAGGLE_API_TOKEN was not found in project .env or the process environment; token value is never logged.")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    all_files = []
    page_token = None
    page_count = 0
    seen_tokens: set[str] = set()
    while True:
        page_count += 1
        listing_args = ["models", "instances", "versions", "files", MODEL_REF,
                        "--format", "json", "--page-size", "200"]
        if page_token:
            listing_args += ["--page-token", page_token]
        listing = run_kaggle(listing_args)
        start = listing.stdout.find("[")
        try:
            page_files, _ = json.JSONDecoder().raw_decode(listing.stdout[start:])
        except (json.JSONDecodeError, ValueError) as exc:
            detail = listing.stdout[:1600] or listing.stderr[:1600]
            token = os.environ.get("KAGGLE_API_TOKEN", "")
            if token:
                detail = detail.replace(token, "<redacted>")
            raise RuntimeError(f"Kaggle CLI did not return parseable JSON file inventory: {detail}") from exc
        if not isinstance(page_files, list):
            raise RuntimeError("Kaggle CLI file inventory was not a JSON array")
        all_files.extend(page_files)
        match = re.search(r"Next Page Token\s*=\s*(\S+)", listing.stdout)
        next_token = match.group(1) if match else None
        if not next_token:
            break
        if next_token in seen_tokens:
            raise RuntimeError("Kaggle CLI repeated a pagination token; stopping instead of looping")
        seen_tokens.add(next_token)
        page_token = next_token
    if not all_files:
        raise RuntimeError("Kaggle API returned an empty version-1 file inventory")
    listing_data = {"model_instance_version": MODEL_REF, "pages": page_count, "files": all_files}
    write_json(PROVENANCE_DIR / "kaggle-version1-cli-files.json", listing_data)

    tar_path = MODEL_DIR / "qwen3-5-4b.tar.gz"
    run_kaggle(["models", "instances", "versions", "download", MODEL_REF,
                "--path", str(MODEL_DIR), "--quiet"])
    if not tar_path.is_file() or tar_path.stat().st_size == 0:
        raise RuntimeError("Kaggle CLI returned without completing the model archive download")
    partial_marker = Path(str(tar_path) + ".kaggle-partial")
    if partial_marker.is_file():
        marker_data = json.loads(partial_marker.read_text(encoding="utf-8"))
        expected_size = marker_data.get("size")
        if expected_size and tar_path.stat().st_size != expected_size:
            raise RuntimeError(f"Incomplete Kaggle model archive: {tar_path.stat().st_size} of {expected_size} bytes")

    extracted_files = safe_extract_model_archive(tar_path, MODEL_DIR)

    configs = []
    for path in MODEL_DIR.rglob("config.json"):
        if any(part.startswith(".") for part in path.relative_to(MODEL_DIR).parts):
            continue
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if config.get("model_type") == "qwen3_5":
            configs.append(path.parent)
    if len(configs) != 1:
        raise RuntimeError(f"Expected one Qwen3.5 model directory under {MODEL_DIR}, found {len(configs)}.")
    base_dir = configs[0]

    entries = []
    for path in sorted(p for p in base_dir.rglob("*") if p.is_file()
                        and ".cache" not in path.relative_to(base_dir).parts):
        entries.append({"path": path.relative_to(base_dir).as_posix(),
                        "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    base_prefix = base_dir.relative_to(MODEL_DIR).as_posix().rstrip("/") + "/"
    api_files = {}
    for row in all_files:
        name = str(row.get("name", "")).replace("\\", "/")
        if not name.startswith(base_prefix):
            continue
        relative = PurePosixPath(name).relative_to(base_prefix).as_posix()
        parts = PurePosixPath(relative).parts
        if ".cache" in parts or any(part.endswith("_environment") for part in parts):
            continue
        api_files[relative] = row.get("size")
    actual_files = {row["path"]: row["size_bytes"] for row in entries}
    missing = sorted(set(api_files) - set(actual_files))
    unexpected = sorted(set(actual_files) - set(api_files))
    size_mismatch = sorted(name for name in set(api_files) & set(actual_files)
                           if api_files[name] is not None and int(api_files[name]) != actual_files[name])
    if not api_files or missing or unexpected or size_mismatch:
        raise RuntimeError(f"Extracted files do not match Kaggle API inventory: missing={missing[:10]}, "
                           f"unexpected={unexpected[:10]}, size_mismatch={size_mismatch[:10]}")
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    present = {row["path"] for row in entries}
    if not required <= present or not any(name.endswith(".safetensors") for name in present):
        raise RuntimeError(f"Downloaded version 1 is missing required Transformers files: {sorted(required - present)}")

    manifest = {
        "kaggle_model_instance_version": MODEL_REF,
        "framework": "pytorch",
        "variant": "default",
        "version": 1,
        "archive": {"path": str(tar_path.relative_to(ROOT)), "size_bytes": tar_path.stat().st_size,
                    "sha256": sha256_file(tar_path)},
        "credential_source": credential_source,
        "base_directory": str(base_dir.relative_to(ROOT)),
        "model_type": "qwen3_5",
        "model_index_present": "model.safetensors.index.json" in present,
        "download_inventory_file_count": len(all_files),
        "download_inventory_pages": page_count,
        "download_inventory_model_file_count_excluding_cache": len(api_files),
        "download_inventory_matches_extracted_files": True,
        "sha256_scope": "all extracted base model files excluding local .cache files",
        "archive_extracted_model_file_count": extracted_files,
        "files": entries,
        "file_count": len(entries),
        "base_directory_tree_sha256": hashlib.sha256(
            "".join(f"{row['path']}\0{row['size_bytes']}\0{row['sha256']}\n" for row in entries).encode("utf-8")
        ).hexdigest(),
        "independent_sha256_verification": True,
    }
    write_json(PROVENANCE_DIR / "kaggle-version1-sha256.json", manifest)
    print(json.dumps({"model_instance_version": MODEL_REF, "base_directory": manifest["base_directory"],
                      "files": len(entries), "tree_sha256": manifest["base_directory_tree_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
