from __future__ import annotations

import json
from pathlib import Path

import pytest

from threech0.models import (
    Chapter,
    Cue,
    SubtitleModeKind,
    Transcript,
    TranscriptBundle,
    TranscriptProvenance,
    VerifyReport,
)

from .conftest import URL, assert_clean, invoke

# -- transcript --------------------------------------------------------------------------


def test_transcript_language_vtt(fake, tmp_path):
    target = tmp_path / "out.vtt"
    result = invoke(["transcript", URL, "--language", "en", "--format", "vtt", "-o",
                     str(target)])
    assert result.exit_code == 0, result.stderr
    call = fake.transcript_calls[0]
    assert call["subtitles"].kind is SubtitleModeKind.LANGUAGES
    assert call["subtitles"].languages == ("en",)
    assert call["transcribe"].enabled is False
    assert call["transcribe"].language == "en"
    text = target.read_text()
    assert text.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:02.500\nWelcome back." in text
    assert "platform captions" in result.stderr
    assert_clean(result.stderr)


def test_transcript_stdout_and_default_name(fake, tmp_path):
    result = invoke(["transcript", URL, "--format", "srt", "--output", "-"])
    assert result.exit_code == 0, result.stderr
    assert result.stdout.startswith("1\n00:00:00,000 --> 00:00:02,500\nWelcome back.")
    folder = invoke(["transcript", URL, "--format", "txt", "-o", str(tmp_path)])
    assert folder.exit_code == 0, folder.stderr
    (written,) = list(tmp_path.glob("*.txt"))
    # Same naming scheme as download sidecars.
    assert written.name == (f"PHYS 201 - 2026-10-14 - Lecture 7 - Thermodynamics [draft] "
                            f"[{fake.recording.short_id}].en.txt")
    assert written.read_text() == "Welcome back.\nToday: entropy.\n"


def test_transcript_json_mode(fake, tmp_path):
    result = invoke(["transcript", URL, "--json", "-o", str(tmp_path / "t.json"),
                     "--format", "json"])
    assert result.exit_code == 0, result.stderr
    doc = json.loads(result.stdout)
    assert doc["command"] == "transcript" and doc["ok"] is True
    assert doc["transcript"]["provenance"] == "platform-captions"
    assert doc["transcript"]["cues"] == 2
    data = json.loads((tmp_path / "t.json").read_text())
    assert data["cues"][0]["text"] == "Welcome back."
    assert invoke(["transcript", URL, "--json", "-o", "-"]).exit_code == 2


def test_transcript_absent_suggests_transcribe(fake):
    fake.bundle = TranscriptBundle(absent=True)
    result = invoke(["transcript", URL, "-o", "-"])
    assert result.exit_code == 4
    assert "No platform transcript found" in result.stderr
    assert "--transcribe local" in result.stderr


def test_transcript_missing_language(fake):
    result = invoke(["transcript", URL, "--language", "fr", "-o", "-"])
    assert result.exit_code == 4
    assert "Available: en" in result.stderr


def test_untimed_transcript_refuses_vtt_but_writes_txt(fake):
    fake.bundle = TranscriptBundle(transcripts=[Transcript(
        "en", TranscriptProvenance.PLATFORM_TRANSCRIPT, cues=[Cue("No timing here.")])])
    vtt = invoke(["transcript", URL, "--format", "vtt", "-o", "-"])
    assert vtt.exit_code == 4
    assert "no timestamps" in vtt.stderr
    txt = invoke(["transcript", URL, "--format", "txt", "-o", "-"])
    assert txt.exit_code == 0
    assert txt.stdout == "No timing here.\n"


def test_generated_transcript_is_labelled(fake, tmp_path):
    fake.bundle = TranscriptBundle(transcripts=[Transcript(
        "en", TranscriptProvenance.GENERATED_LOCAL, cues=[Cue("Hello.", 0.0, 1.0)],
        generator="faster-whisper small (int8, cpu)")])
    result = invoke(["transcript", URL, "--transcribe", "local", "--allow-model-download",
                     "-o", str(tmp_path)])
    assert result.exit_code == 0, result.stderr
    call = fake.transcript_calls[0]
    assert call["transcribe"].enabled and call["transcribe"].allow_model_download
    assert call["confirm"]("model?") is True
    (written,) = list(tmp_path.glob("*.vtt"))
    assert ".generated." in written.name
    assert "NOTE generated locally" in written.read_text()
    assert "generated" in result.stderr


# -- chapters ----------------------------------------------------------------------------


def _chapter_file(tmp_path: Path, chapters: list[dict]) -> Path:
    path = tmp_path / "chapters.json"
    path.write_text(json.dumps({"version": 1, "chapters": chapters}))
    return path


GOOD = [{"start": "00:00:00", "title": "Introduction"},
        {"start": "00:08:30", "end": "00:20:00", "title": "Worked example"}]


@pytest.mark.parametrize("width", ["40", "100"])
def test_chapters_validate_good(fake, tmp_path, width):
    path = _chapter_file(tmp_path, GOOD)
    result = invoke(["chapters", "validate", str(path), "--duration", "1h2m"],
                    env={"COLUMNS": width})
    assert result.exit_code == 0, result.stderr
    err = result.stderr
    assert "Worked example" in err
    assert "00:08:30" in err and "00:20:00" in err and "00:08:30" in err
    assert "01:02:00" not in err or "Introduction" in err
    assert "[OK] 2 chapter(s) valid" in " ".join(err.split())
    for line in err.splitlines():
        assert len(line) <= int(width)


def test_chapters_validate_resolves_end_from_next_start(fake, tmp_path):
    path = _chapter_file(tmp_path, GOOD)
    result = invoke(["chapters", "validate", str(path), "--duration", "1h2m", "--json"])
    doc = json.loads(result.stdout)
    assert doc["ok"] is True
    assert doc["chapters"][0] == {"start": 0.0, "end": 510.0, "title": "Introduction"}
    assert doc["duration"] == 3720.0


def test_chapters_validate_bad(fake, tmp_path):
    path = _chapter_file(tmp_path, [{"start": "00:10:00", "title": "B"},
                                    {"start": "00:05:00", "title": "A"}])
    result = invoke(["chapters", "validate", str(path), "--media", str(path)])
    assert result.exit_code == 8
    assert "[FAIL] chapter starts must be strictly increasing" in result.stderr
    json_result = invoke(["chapters", "validate", str(path), "--json"])
    assert json_result.exit_code == 8
    doc = json.loads(json_result.stdout)
    assert doc["ok"] is False and doc["problems"]


def test_chapters_validate_invalid_json_file(fake, tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{nope")
    result = invoke(["chapters", "validate", str(path)])
    assert result.exit_code == 8
    assert "not valid JSON" in result.stderr


def test_chapters_validate_usage(fake, tmp_path):
    path = _chapter_file(tmp_path, GOOD)
    assert invoke(["chapters", "validate", str(tmp_path / "nope.json")]).exit_code == 2
    assert invoke(["chapters", "validate", str(path), "--duration", "x"]).exit_code == 2
    both = invoke(["chapters", "validate", str(path), "--duration", "1h", "--media", str(path)])
    assert both.exit_code == 2


def test_chapters_show(fake, tmp_path):
    media = tmp_path / "lecture.mp4"
    media.write_bytes(b"\0")
    result = invoke(["chapters", "show", str(media)])
    assert result.exit_code == 0
    assert "No embedded chapters" in result.stderr
    fake.probe_report = VerifyReport(ok=True, duration=600.0,
                                     chapters=[Chapter(0, "Intro"), Chapter(120.5, "Main")])
    shown = invoke(["chapters", "show", str(media), "--ascii"])
    assert shown.exit_code == 0
    assert "00:02:00.500" in shown.stderr and "Main" in shown.stderr
    assert shown.stderr.isascii()
    doc = json.loads(invoke(["chapters", "show", str(media), "--json"]).stdout)
    assert doc["chapters"][1] == {"start": 120.5, "end": 600.0, "title": "Main"}
    assert_clean(shown.stderr)


def test_json_format_always_uses_threech0_schema():
    from threech0.cli.output import serialise_transcript
    from threech0.models import Cue, Transcript, TranscriptProvenance

    t = Transcript(language="en", provenance=TranscriptProvenance.PLATFORM_TRANSCRIPT,
                   cues=[Cue("hi", 0.0, 1.0)], raw=b'{"platform": "shape?token=SECRET1"}',
                   raw_format="json")
    out = serialise_transcript(t, "json")
    assert '"version": 1' in out and "SECRET1" not in out and '"platform"' not in out


def test_unfetchable_platform_transcript_is_not_reported_as_absent():
    from threech0.cli.app import pick_transcript
    from threech0.errors import ExitCode, ForbiddenError
    from threech0.models import Failure, TranscriptBundle

    bundle = TranscriptBundle(failed=[Failure("subtitles", "captions:en", "forbidden",
                                              "Could not fetch English captions: HTTP 403",
                                              hint="Run `3ch0 auth login`")])
    with pytest.raises(ForbiddenError) as info:
        pick_transcript(bundle, "en", "vtt", transcribe_enabled=False)
    assert info.value.exit_code == ExitCode.AUTH and "403" in info.value.message
