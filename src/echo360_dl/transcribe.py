"""Optional local speech recognition.

Backends (both optional extras, both run entirely on this machine — audio is never uploaded):

* ``faster-whisper`` (extra ``transcribe``): CTranslate2. On macOS it runs
  on the **CPU** (int8). CTranslate2 has no Metal/GPU backend, so no Apple GPU acceleration.
* ``mlx-whisper`` (extra ``transcribe-mlx``, Apple Silicon only): runs on the
  Apple GPU via MLX/Metal. Heavier install.

``auto`` prefers mlx-whisper when it is installed on Apple Silicon, else faster-whisper.

Model downloads are never silent: if the chosen model is not in
``paths.models_dir()``, :func:`get_transcriber` asks ``confirm_download(message)`` (interactive)
or requires ``TranscribeOptions.allow_model_download`` (non-interactive) and otherwise raises
:class:`ModelDownloadRequiredError`.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from echo360_dl import paths
from echo360_dl.errors import (
    CancelledError,
    DependencyError,
    ModelDownloadRequiredError,
    UsageError,
)
from echo360_dl.models import (
    Cue,
    ProgressSink,
    TranscribeOptions,
    Transcriber,
    Transcript,
    TranscriptProvenance,
)

FASTER_WHISPER = "faster-whisper"
MLX_WHISPER = "mlx-whisper"
BACKENDS = (FASTER_WHISPER, MLX_WHISPER)

#: Approximate download sizes (MB) of the published model weights, for consent prompts.
MODEL_SIZES_MB = {
    "tiny": 75, "tiny.en": 75, "base": 145, "base.en": 145, "small": 480, "small.en": 480,
    "medium": 1500, "medium.en": 1500, "large-v3": 3100, "large-v2": 3100,
    "large-v3-turbo": 1600, "turbo": 1600, "distil-large-v3": 1500,
}

#: mlx-community repositories for model names (MLX conversions of OpenAI Whisper).
_MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx", "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "base": "mlx-community/whisper-base-mlx", "base.en": "mlx-community/whisper-base.en-mlx",
    "small": "mlx-community/whisper-small-mlx", "small.en": "mlx-community/whisper-small.en-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}

ConfirmFn = Callable[[str], bool]


@dataclass(frozen=True)
class BackendInfo:
    name: str
    available: bool
    accelerated: str | None  # e.g. "Apple GPU (Metal via MLX)"; None = CPU only
    detail: str


def _apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def available_backends() -> list[BackendInfo]:
    fw = _installed("faster_whisper")
    mlx = _installed("mlx_whisper")
    return [
        BackendInfo(
            FASTER_WHISPER, fw, None,
            "installed (CTranslate2, CPU int8)" if fw
            else "not installed — see README: Local transcription"),
        BackendInfo(
            MLX_WHISPER, mlx and _apple_silicon(),
            "Apple GPU (Metal via MLX)" if _apple_silicon() else None,
            ("installed (MLX)" if mlx else
             "not installed — see README: Local transcription")
            if _apple_silicon() else "requires Apple Silicon macOS"),
    ]


def resolve_backend(name: str) -> str:
    infos = {b.name: b for b in available_backends()}
    if name == "auto":
        for candidate in (MLX_WHISPER, FASTER_WHISPER):
            if infos[candidate].available:
                return candidate
        raise DependencyError(
            "No local transcription backend is installed.",
            hint="From the echo360-dl project folder run: "
                 "uv tool install --force '.[transcribe]' (see README: Local transcription).")
    if name not in infos:
        raise UsageError(f"Unknown transcription backend {name!r}.",
                         hint=f"Choose one of: auto, {', '.join(BACKENDS)}.")
    if not infos[name].available:
        raise DependencyError(f"Transcription backend {name} is unavailable: "
                              f"{infos[name].detail}.")
    return name


def _consent(options: TranscribeOptions, backend: str, confirm: ConfirmFn | None,
             target: Path) -> None:
    size = MODEL_SIZES_MB.get(options.model)
    approx = f"~{size} MB" if size else "size unknown"
    message = (f"Download speech model '{options.model}' for {backend} ({approx}) "
               f"to {target}?")
    if options.allow_model_download:
        return
    if confirm is not None and confirm(message):
        return
    raise ModelDownloadRequiredError(
        f"Speech model '{options.model}' ({approx}) is not downloaded yet.",
        hint="Re-run with --allow-model-download to fetch it once (stored in "
             f"{target}), or choose a smaller --transcribe-model such as 'base'.")


def get_transcriber(options: TranscribeOptions, *,
                    confirm_download: ConfirmFn | None = None) -> Transcriber:
    backend = resolve_backend(options.backend)
    if backend == MLX_WHISPER:
        return MlxWhisperTranscriber.create(options, confirm_download)
    return FasterWhisperTranscriber.create(options, confirm_download)


def _whisper_language(code: str | None) -> str | None:
    """Whisper takes bare ISO 639-1 codes: ``en-CA`` -> ``en``."""
    from echo360_dl.selection import normalize_language

    return normalize_language(code)


def _cues_from_segments(segments: list[tuple[float, float, str]]) -> list[Cue]:
    cues: list[Cue] = []
    for start, end, text in segments:
        text = " ".join(text.split())
        if not text:
            continue
        start = max(0.0, float(start))
        end = max(start + 0.001, float(end))
        cues.append(Cue(text=text, start=round(start, 3), end=round(end, 3)))
    return cues


class FasterWhisperTranscriber:
    """faster-whisper on CPU (int8)."""

    def __init__(self, model: Any, options: TranscribeOptions) -> None:
        self._model = model
        self._options = options
        #: Set by the pipeline; checked between segments so Ctrl-C stops transcription.
        self.cancel_event: threading.Event | None = None

    @classmethod
    def create(cls, options: TranscribeOptions,
               confirm: ConfirmFn | None) -> FasterWhisperTranscriber:
        from faster_whisper import WhisperModel, download_model

        root = paths.models_dir() / FASTER_WHISPER
        try:
            local = download_model(options.model, cache_dir=str(root), local_files_only=True)
        except ValueError as exc:  # unknown model name
            raise UsageError(str(exc)) from None
        except Exception:
            _consent(options, FASTER_WHISPER, confirm, root)
            paths.ensure_private_dir(root)
            local = download_model(options.model, cache_dir=str(root))
        model = WhisperModel(local, device="cpu", compute_type="int8")
        return cls(model, options)

    def transcribe(self, audio_path: Path, *, language: str | None,
                   progress: ProgressSink) -> Transcript:
        segments, info = self._model.transcribe(
            str(audio_path), language=_whisper_language(language or self._options.language),
            vad_filter=True, beam_size=5)
        task = "transcribe"
        progress.task_start(task, f"Transcribing locally ({FASTER_WHISPER} "
                            f"{self._options.model}, CPU)", total=info.duration, unit="seconds")
        collected: list[tuple[float, float, str]] = []
        for seg in segments:  # generator: decoding happens while iterating
            if self.cancel_event is not None and self.cancel_event.is_set():
                progress.task_done(task, ok=False, message="cancelled")
                raise CancelledError("Transcription cancelled.")
            collected.append((seg.start, seg.end, seg.text))
            progress.task_update(task, completed=min(seg.end, info.duration))
        progress.task_done(task, ok=True)
        return Transcript(
            language=info.language, provenance=TranscriptProvenance.GENERATED_LOCAL,
            cues=_cues_from_segments(collected), label="generated",
            source_detail="local speech recognition (audio not uploaded)",
            generator=f"{FASTER_WHISPER} {self._options.model} (int8, CPU)")


class MlxWhisperTranscriber:
    """mlx-whisper on the Apple GPU."""

    def __init__(self, model_path: str, options: TranscribeOptions) -> None:
        self._model_path = model_path
        self._options = options

    @classmethod
    def create(cls, options: TranscribeOptions,
               confirm: ConfirmFn | None) -> MlxWhisperTranscriber:
        from huggingface_hub import snapshot_download

        repo = _MLX_REPOS.get(options.model)
        if repo is None:
            raise UsageError(f"Model {options.model!r} is not available for {MLX_WHISPER}.",
                             hint=f"Choose one of: {', '.join(_MLX_REPOS)}.")
        root = paths.models_dir() / MLX_WHISPER
        try:
            local = snapshot_download(repo, cache_dir=str(root), local_files_only=True)
        except Exception:
            _consent(options, MLX_WHISPER, confirm, root)
            paths.ensure_private_dir(root)
            local = snapshot_download(repo, cache_dir=str(root))
        return cls(local, options)

    def transcribe(self, audio_path: Path, *, language: str | None,
                   progress: ProgressSink) -> Transcript:
        import mlx_whisper

        task = "transcribe"
        # mlx-whisper exposes no progress callback: report an indeterminate task.
        progress.task_start(task, f"Transcribing locally ({MLX_WHISPER} "
                            f"{self._options.model}, Apple GPU)", total=None, unit="seconds")
        result = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=self._model_path,
                                        language=_whisper_language(
                                            language or self._options.language),
                                        verbose=None)
        progress.task_done(task, ok=True)
        segments = [(s["start"], s["end"], s["text"]) for s in result.get("segments", [])]
        return Transcript(
            language=result.get("language"), provenance=TranscriptProvenance.GENERATED_LOCAL,
            cues=_cues_from_segments(segments), label="generated",
            source_detail="local speech recognition (audio not uploaded)",
            generator=f"{MLX_WHISPER} {self._options.model} (Metal)")
