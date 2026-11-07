"""Transcription backend selection, consent and (opt-in) a real run on synthetic speech."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from echo360_dl import transcribe
from echo360_dl.errors import DependencyError, ModelDownloadRequiredError, UsageError
from echo360_dl.models import NullProgress, TranscribeOptions, TranscriptProvenance


def _fake_backends(monkeypatch, fw: bool, mlx: bool, apple: bool = True):
    monkeypatch.setattr(transcribe, "_apple_silicon", lambda: apple)
    monkeypatch.setattr(transcribe, "_installed",
                        lambda m: {"faster_whisper": fw, "mlx_whisper": mlx}.get(m, False))


def test_auto_prefers_mlx_on_apple_silicon(monkeypatch):
    _fake_backends(monkeypatch, fw=True, mlx=True)
    assert transcribe.resolve_backend("auto") == transcribe.MLX_WHISPER
    _fake_backends(monkeypatch, fw=True, mlx=True, apple=False)
    assert transcribe.resolve_backend("auto") == transcribe.FASTER_WHISPER


def test_no_backend_is_actionable_dependency_error(monkeypatch):
    _fake_backends(monkeypatch, fw=False, mlx=False)
    with pytest.raises(DependencyError) as info:
        transcribe.resolve_backend("auto")
    assert ".[transcribe]" in (info.value.hint or "")
    with pytest.raises(UsageError):
        transcribe.resolve_backend("cloud-api")


def test_backend_info_never_claims_gpu_for_faster_whisper(monkeypatch):
    _fake_backends(monkeypatch, fw=True, mlx=False)
    info = {b.name: b for b in transcribe.available_backends()}
    assert info["faster-whisper"].accelerated is None
    assert "CPU" in info["faster-whisper"].detail


def test_consent_required_without_permission(tmp_path):
    opts = TranscribeOptions(enabled=True, model="small")
    with pytest.raises(ModelDownloadRequiredError) as info:
        transcribe._consent(opts, "faster-whisper", None, tmp_path)
    assert "~480 MB" in info.value.message and "--allow-model-download" in info.value.hint
    prompts: list[str] = []
    with pytest.raises(ModelDownloadRequiredError):
        transcribe._consent(opts, "faster-whisper", lambda m: prompts.append(m) or False, tmp_path)
    assert prompts and "small" in prompts[0] and str(tmp_path) in prompts[0]
    transcribe._consent(opts, "faster-whisper", lambda m: True, tmp_path)
    transcribe._consent(TranscribeOptions(allow_model_download=True), "x", None, tmp_path)


def test_cues_never_negative_or_empty():
    cues = transcribe._cues_from_segments([(-0.2, 1.0, "  hello  world "), (1.0, 1.0, "x"),
                                           (2.0, 3.0, "   ")])
    assert [(c.start, c.end, c.text) for c in cues] == [(0.0, 1.0, "hello world"),
                                                        (1.0, 1.001, "x")]


@pytest.mark.transcribe
@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("say"),
                    reason="needs macOS 'say' for synthetic speech")
def test_real_faster_whisper_on_synthetic_speech(tmp_path, monkeypatch):
    """Opt-in: set ECHO360_DL_TEST_MODEL_HOME to a home whose cache already holds 'tiny'."""
    pytest.importorskip("faster_whisper")
    home = os.environ.get("ECHO360_DL_TEST_MODEL_HOME")
    if not home:
        pytest.skip("set ECHO360_DL_TEST_MODEL_HOME (model cache with faster-whisper 'tiny')")
    monkeypatch.setenv("ECHO360_DL_HOME", home)
    audio = tmp_path / "speech.aiff"
    subprocess.run(["say", "-o", str(audio), "Today we discuss eigenvalues."], check=True)
    t = transcribe.get_transcriber(TranscribeOptions(enabled=True, backend="faster-whisper",
                                                     model="tiny"))
    result = t.transcribe(audio, language="en", progress=NullProgress())
    assert result.provenance is TranscriptProvenance.GENERATED_LOCAL and result.generated
    assert result.timed and "eigenvalue" in " ".join(c.text for c in result.cues).lower()
    assert "CPU" in (result.generator or "")


def test_region_codes_are_reduced_for_whisper():
    assert transcribe._whisper_language("en-CA") == "en"
    assert transcribe._whisper_language("fr_CA") == "fr"
    assert transcribe._whisper_language(None) is None


def test_faster_whisper_stops_between_segments_when_cancelled():
    import threading
    from types import SimpleNamespace

    from echo360_dl.errors import CancelledError

    event = threading.Event()

    def segments():
        yield SimpleNamespace(start=0.0, end=1.0, text="one")
        event.set()  # Ctrl-C arrives while decoding continues
        yield SimpleNamespace(start=1.0, end=2.0, text="two")
        raise AssertionError("decoding continued after cancellation")

    model = SimpleNamespace(transcribe=lambda *a, **k: (segments(),
                                                       SimpleNamespace(duration=3.0,
                                                                       language="en")))
    t = transcribe.FasterWhisperTranscriber(model, TranscribeOptions(enabled=True))
    t.cancel_event = event
    with pytest.raises(CancelledError):
        t.transcribe(Path("x.m4a"), language="en-CA", progress=NullProgress())
