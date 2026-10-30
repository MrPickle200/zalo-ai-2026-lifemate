from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

_NVIDIA_DLL_HANDLES = []


class ASRModel:
    """Deterministic Whisper Small wrapper using local CTranslate2 weights."""

    def __init__(self, model_path: str | Path | None = None, cache_dir: str | Path = "cache/transcripts", device: str | None = None):
        self.model_path = Path(model_path or os.environ.get("LIFEMATE_ASR_MODEL", "models/faster-whisper-small")).expanduser()
        self.cache_dir = Path(cache_dir)
        self.device = device or ("cuda" if self._cuda_available() else "cpu")
        self._model = None

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load(self):
        if self._model is None:
            if not self.model_path.is_dir():
                raise FileNotFoundError(
                    f"Whisper weights must be available locally at {self.model_path}. "
                    "Set LIFEMATE_ASR_MODEL to a local faster-whisper model directory."
                )
            if os.name == "nt":
                # NVIDIA's Windows CUDA wheels keep DLLs in package-local bin folders,
                # which are not on the loader path by default.
                try:
                    import site
                    for site_dir in site.getsitepackages():
                        nvidia_dir = Path(site_dir) / "nvidia"
                        if nvidia_dir.is_dir():
                            for bin_dir in nvidia_dir.glob("*/bin"):
                                resolved = str(bin_dir.resolve())
                                _NVIDIA_DLL_HANDLES.append(os.add_dll_directory(resolved))
                                if resolved.lower() not in os.environ.get("PATH", "").lower().split(os.pathsep):
                                    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + resolved
                except (OSError, AttributeError):
                    pass
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError("Missing faster-whisper; install requirements.txt into .venv") from exc
            compute_type = "float16" if self.device == "cuda" else "int8"
            try:
                self._model = WhisperModel(str(self.model_path), device=self.device, compute_type=compute_type)
            except Exception as exc:
                if self.device != "cuda":
                    raise
                warnings.warn(f"Whisper CUDA runtime unavailable ({exc}); falling back to CPU inference.", RuntimeWarning)
                self.device = "cpu"
                self._model = WhisperModel(str(self.model_path), device="cpu", compute_type="int8")
        return self._model

    def transcribe(self, audio_path: str) -> str:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio not found: {path}")
        key = hashlib.sha256()
        key.update(str(self.model_path).encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                key.update(chunk)
        cache_path = self.cache_dir / f"{key.hexdigest()}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))["text"]
        def run_decode(model):
            segments, _ = model.transcribe(str(path), language="vi", task="transcribe", beam_size=1,
                                           best_of=1, temperature=0.0, vad_filter=False,
                                           condition_on_previous_text=False)
            return " ".join(segment.text.strip() for segment in segments).strip()

        model = self._load()
        try:
            transcript = run_decode(model)
        except Exception as exc:
            if self.device != "cuda":
                raise
            warnings.warn(f"Whisper CUDA inference failed ({exc}); falling back to CPU inference.", RuntimeWarning)
            self._model = None
            self.device = "cpu"
            transcript = run_decode(self._load())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"text": transcript}, ensure_ascii=False), encoding="utf-8")
        return transcript


class PhoASRModel(ASRModel):
    """PhoASR-whisper-small using the released Hugging Face Transformers weights."""

    def __init__(self, model_path: str | Path | None = None,
                 cache_dir: str | Path = "cache/transcripts", device: str | None = None):
        super().__init__(model_path or os.environ.get("LIFEMATE_ASR_MODEL", "models/pho-asr-whisper-small"), cache_dir, device)
        self.processor = None

    def _load(self):
        if self._model is None:
            if not self.model_path.is_dir():
                raise FileNotFoundError(f"PhoASR weights not found locally: {self.model_path}")
            try:
                import torch
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            except ImportError as exc:
                raise RuntimeError("PhoASR requires torch and transformers") from exc
            self.processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True)
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                str(self.model_path), local_files_only=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self._model.eval()
        return self._model

    @staticmethod
    def _read_audio(path: Path):
        # Decode/resample locally through PyAV, already installed with faster-whisper.
        import av
        import numpy as np
        container = av.open(str(path))
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        chunks = []
        for frame in container.decode(stream):
            chunks.extend(frame.to_ndarray() for frame in resampler.resample(frame))
        chunks.extend(frame.to_ndarray() for frame in resampler.resample(None))
        container.close()
        if not chunks:
            raise ValueError(f"No audio frames decoded from {path}")
        return np.concatenate(chunks, axis=-1).reshape(-1)

    def transcribe(self, audio_path: str) -> str:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio not found: {path}")
        key = hashlib.sha256()
        key.update(str(self.model_path).encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                key.update(chunk)
        cache_path = self.cache_dir / f"{key.hexdigest()}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))["text"]
        import torch
        model = self._load()
        waveform = self._read_audio(path)
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        inputs = {name: tensor.to(device=self.device,
                                  dtype=model.dtype if name == "input_features" else tensor.dtype)
                  for name, tensor in inputs.items()}
        with torch.inference_mode():
            tokens = model.generate(**inputs, language="vi", task="transcribe", do_sample=False,
                                    num_beams=1, return_timestamps=False)
        transcript = self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"text": transcript}, ensure_ascii=False), encoding="utf-8")
        return transcript

