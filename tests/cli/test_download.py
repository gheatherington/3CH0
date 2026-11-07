from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from echo360_dl.models import (
    ChapterModeKind,
    Container,
    DownloadResult,
    Encoder,
    Failure,
    IfExists,
    Layout,
    OutputFile,
    OutputKind,
    ResultStatus,
    SubtitleModeKind,
)

from .conftest import CONTENT, MODEL_MESSAGE, QUERY, URL, assert_clean, invoke, make_recording


def test_non_interactive_without_streams_is_usage_error(fake):
    result = invoke(["download", URL, "--non-interactive"])
    assert result.exit_code == 2
    assert "--streams all" in result.stderr
    assert "Download which sources" not in result.output
    assert fake.discover_calls == []
    assert fake.requests == []


def test_no_tty_counts_as_non_interactive(fake):
    # CliRunner streams are not terminals: no prompt even without --non-interactive.
    result = invoke(["download", URL], input="1\ny\n")
    assert result.exit_code == 2
    assert "Download which sources" not in result.output


def test_json_without_streams_is_usage_error_json(fake):
    result = invoke(["download", URL, "--json"])
    assert result.exit_code == 2
    doc = json.loads(result.stdout)
    assert doc["error"]["kind"] == "selection-required"
    assert "--streams" in doc["error"]["hint"]


@pytest.mark.parametrize("extra", [["--non-interactive"], ["--json"]])
def test_login_required_non_interactive_exit_3_without_login(fake, extra):
    fake.login_required = True
    result = invoke(["download", URL, "--streams", "all", *extra])
    assert result.exit_code == 3
    text = result.stdout + result.stderr
    assert "3ch0 auth login" in text
    assert fake.login_calls == []
    assert fake.discover_calls[0][1].interactive is False
    assert fake.requests == []


def test_example_all_best_auto_output(fake):
    result = invoke(["download", URL, "--streams", "all", "--quality", "best", "--subtitles",
                     "auto", "--chapters", "auto", "--output", "~/Downloads"])
    assert result.exit_code == 0, result.stderr
    (req,) = fake.requests
    # Source 2 of the fake recording is FORBIDDEN: "all" skips it with a warning.
    assert req.source_indexes == (1,)
    flat = " ".join(result.stderr.split())
    assert "Skipping Source 2: forbidden" in flat
    assert "--streams 2" in flat
    assert req.audio_only is False
    assert req.quality.kind == "best"
    assert req.subtitles.kind is SubtitleModeKind.AUTO
    assert req.chapters.kind is ChapterModeKind.AUTO
    assert req.output_dir == Path.home() / "Downloads"
    assert req.container is Container.MP4
    assert req.if_exists is IfExists.SKIP
    assert req.transcribe.enabled is False
    assert req.write_sidecars is True
    assert (req.retries, req.concurrency) == (3, 2)
    assert isinstance(fake.cancels[0], threading.Event)


def test_example_source2_720p_en_chapters_file(fake, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("chapters.json").write_text('{"version": 1, "chapters": []}')
    result = invoke(["download", URL, "--streams", "2", "--quality", "720p", "--subtitles",
                     "en", "--chapters-file", "chapters.json", "--non-interactive"])
    assert result.exit_code == 0, result.stderr
    (req,) = fake.requests
    assert req.source_indexes == (2,)
    assert (req.quality.kind, req.quality.max_height) == ("max_height", 720)
    assert req.subtitles.kind is SubtitleModeKind.LANGUAGES
    assert req.subtitles.languages == ("en",)
    assert req.chapters.kind is ChapterModeKind.FILE
    assert req.chapters.file == Path("chapters.json")
    assert req.output_dir.is_absolute()


def test_example_transcribe_local_non_interactive(fake):
    result = invoke(["download", URL, "--streams", "all", "--transcribe", "local",
                     "--non-interactive"])
    assert result.exit_code == 0, result.stderr
    (req,) = fake.requests
    assert req.transcribe.enabled is True
    assert req.transcribe.backend == "auto"
    assert req.transcribe.model == "small"
    assert req.transcribe.allow_model_download is False
    assert fake.confirm_answers == [False]  # no consent without --allow-model-download


def test_allow_model_download_flag(fake):
    result = invoke(["download", URL, "--streams", "1", "--transcribe", "local",
                     "--transcribe-backend", "mlx-whisper", "--transcribe-model", "medium",
                     "--allow-model-download", "--non-interactive"])
    assert result.exit_code == 0, result.stderr
    req = fake.requests[0]
    assert (req.transcribe.backend, req.transcribe.model) == ("mlx-whisper", "medium")
    assert fake.confirm_answers == [True]


def test_other_options_map(fake, tmp_path):
    result = invoke(["download", URL, "--streams", "audio", "--container", "mkv",
                     "--subtitles", "none", "--chapters", "interval", "--chapter-interval",
                     "5m", "--if-exists", "overwrite", "--no-sidecars", "--keep-work-dir",
                     "--retries", "1", "--concurrency", "4", "--default-subtitle", "fr",
                     "-o", str(tmp_path)])
    assert result.exit_code == 0, result.stderr
    req = fake.requests[0]
    assert req.audio_only is True and req.source_indexes == ()
    assert req.container is Container.MKV
    assert req.subtitles.kind is SubtitleModeKind.NONE
    assert (req.chapters.kind, req.chapters.interval) == (ChapterModeKind.INTERVAL, 300.0)
    assert req.if_exists is IfExists.OVERWRITE
    assert req.write_sidecars is False and req.keep_work_dir is True
    assert (req.retries, req.concurrency, req.default_subtitle) == (1, 4, "fr")
    assert req.output_dir == tmp_path


@pytest.mark.parametrize("args, fragment", [
    (["--streams", "x"], "--streams"),
    (["--streams", "1", "--quality", "huge"], "--quality"),
    (["--streams", "1", "--subtitles", "e!"], "--subtitles"),
    (["--streams", "1", "--chapter-interval", "5m"], "--chapters interval"),
    (["--streams", "1", "--chapters-file", "/nonexistent/ch.json"], "not found"),
    (["--streams", "1", "--container", "avi"], "avi"),
])
def test_usage_errors_exit_2(fake, args, fragment):
    result = invoke(["download", URL, *args])
    assert result.exit_code == 2
    assert fragment in result.output
    assert fake.requests == []


def test_unknown_source_and_quality_unavailable_exit_4(fake):
    assert invoke(["download", URL, "--streams", "3"]).exit_code == 4
    result = invoke(["download", URL, "--streams", "1", "--quality", "240p"])
    assert result.exit_code == 4
    assert "360p" in result.stderr
    assert fake.requests == []


def test_human_report_and_plan(fake):
    fake.recording = make_recording(sources=2, forbidden_second=False)
    result = invoke(["download", URL, "--streams", "all"])
    assert result.exit_code == 0
    err = result.stderr
    assert_clean(err)
    assert "DOWNLOAD PLAN" in err
    assert "720p (1280x720)" in err
    assert "shared track, downloaded once" in err
    assert "en (platform captions)" in err
    assert "3 native" in err
    assert "[OK] Download complete" in err
    assert "chapters ✓ 3 embedded" in err
    assert "h264 1280x720" in err
    assert "27.0 MB" in err


@pytest.mark.parametrize("value, expected", [
    ("Week 3", "Week 3"), ("Week 3.mp4", "Week 3"), ("  Lecture.MKV ", "Lecture"),
])
def test_name_maps_to_request_and_plan(fake, value, expected):
    result = invoke(["download", URL, "--streams", "1", "--name", value])
    assert result.exit_code == 0, result.stderr
    assert fake.requests[0].name == expected
    assert f"{expected}.mp4" in result.stderr
    assert f"sidecars: {expected}.<lang>" in result.stderr


def test_no_name_keeps_default(fake):
    assert invoke(["download", URL, "--streams", "1"]).exit_code == 0
    assert fake.requests[0].name is None


@pytest.mark.parametrize("value, fragment", [
    ("sub/dir", "not a path"), ("a\\b", "not a path"), ("..", "no usable"), ("  ", "no usable"),
])
def test_bad_name_is_usage_error(fake, value, fragment):
    result = invoke(["download", URL, "--streams", "1", "--name", value])
    assert result.exit_code == 2
    assert fragment in result.stderr
    assert fake.requests == []


def test_json_download_result(fake):
    result = invoke(["download", URL, "--streams", "1", "--json"])
    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["command"] == "download" and doc["ok"] is True
    assert doc["result"]["status"] == "ok"
    assert doc["result"]["outputs"][0]["kind"] == "video"
    assert_clean(result.stdout + result.stderr)
    assert "DOWNLOAD PLAN" not in result.stderr


def _partial(request):
    out = request.output_dir / "Lecture-7-s1.mp4"
    return DownloadResult(
        ResultStatus.PARTIAL, lesson_id="x",
        outputs=[OutputFile(out, OutputKind.VIDEO, source_index=1, size=1000)],
        failures=[Failure("download", "Source 2", "unauthorized",
                          f"HTTP 403 for {CONTENT}/s2q1.m3u8{QUERY}",
                          hint="Source 2 may not exist for this lesson.")],
        warnings=["Native chapters were overridden by chapters.json."])


def test_partial_result_exit_5_lists_outputs(fake, tmp_path):
    fake.download_result = _partial
    result = invoke(["download", URL, "--streams", "all", "-o", str(tmp_path)])
    assert result.exit_code == 5
    err = result.stderr
    flat = "".join(err.replace("│", "").split())  # undo wrapping inside box borders
    assert "TARGET" in err and "".join(str(tmp_path).split()) in flat  # destination
    assert "Lecture-7-s1.mp4" in flat
    assert "[PARTIAL]" in err and "1 output" in err
    assert "Source 2" in err and "may not exist" in err
    assert "Native chapters were overridden" in err
    assert_clean(err)
    json_result = invoke(["download", URL, "--streams", "all", "-o", str(tmp_path), "--json"])
    assert json_result.exit_code == 5
    doc = json.loads(json_result.stdout)
    assert doc["ok"] is False and doc["result"]["status"] == "partial"
    assert_clean(json_result.stdout)


def test_keyboard_interrupt_exit_130(fake):
    fake.download_error = KeyboardInterrupt()
    result = invoke(["download", URL, "--streams", "1"])
    assert result.exit_code == 130
    assert "cancelled — rerun to resume" in result.stderr


def test_cancelled_result_exit_130(fake):
    fake.download_result = DownloadResult(ResultStatus.CANCELLED)
    result = invoke(["download", URL, "--streams", "1"])
    assert result.exit_code == 130
    assert "rerun to resume" in result.stderr
    json_result = invoke(["download", URL, "--streams", "1", "--json"])
    assert json_result.exit_code == 130
    assert json.loads(json_result.stdout)["result"]["status"] == "cancelled"


def test_keyboard_interrupt_json(fake):
    fake.download_error = KeyboardInterrupt()
    result = invoke(["download", URL, "--streams", "1", "--json"])
    assert result.exit_code == 130
    assert json.loads(result.stdout)["error"]["kind"] == "cancelled"


def test_unexpected_error_is_scrubbed_one_line(fake):
    fake.download_error = RuntimeError(f"boom while fetching {CONTENT}/x.m3u8{QUERY}")
    result = invoke(["download", URL, "--streams", "1"])
    assert result.exit_code == 1
    assert "Unexpected error: RuntimeError: boom" in result.stderr
    assert "Traceback" not in result.stderr
    assert "--verbose" in result.stderr
    assert_clean(result.stderr)
    verbose = invoke(["download", URL, "--streams", "1", "--verbose"])
    assert verbose.exit_code == 1
    assert "Traceback" in verbose.stderr
    assert_clean(verbose.stderr)


def test_quiet_only_outputs(fake, tmp_path):
    result = invoke(["download", URL, "--streams", "1", "-q", "-o", str(tmp_path)])
    assert result.exit_code == 0
    lines = [line for line in result.stderr.splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0].endswith(".mp4")
    assert result.stdout == ""


def test_verbose_debug_lines_scrubbed(fake):
    result = invoke(["download", URL, "--streams", "1", "-v"])
    assert result.exit_code == 0
    assert "debug:" in result.stderr
    assert_clean(result.stderr)


def test_progress_stages_in_order(fake):
    result = invoke(["download", URL, "--streams", "1"])
    err = result.stderr
    positions = [err.index(f"STAGE     {n}/6 {stage}") for n, stage in (
        (1, "discover"), (2, "select"), (3, "download"), (6, "verify"))]
    assert positions == sorted(positions)
    assert "\r" not in err and "\x1b" not in err  # redirected: plain sequential lines


# -- interactive flow ---------------------------------------------------------------------


@pytest.fixture
def two_available(fake):
    fake.recording = make_recording(sources=2, forbidden_second=False)
    return fake


@pytest.mark.parametrize("answer, indexes, audio_only", [
    ("2", (2,), False),
    ("all", (1, 2), False),
    ("audio", (), True),
    ("", (1, 2), False),
])
def test_interactive_pick(two_available, tty, answer, indexes, audio_only):
    # Picking two sources adds one "Layout?" prompt; Enter keeps the default (separate).
    layout_answer = "\n" if len(indexes) == 2 else ""
    result = invoke(["download", URL], input=f"{answer}\n{layout_answer}y\n")
    assert result.exit_code == 0, result.stderr
    assert "Download which sources?" in result.stderr
    assert "DOWNLOAD PLAN" in result.stderr
    assert "Start download?" in result.stderr
    (req,) = two_available.requests
    assert req.source_indexes == indexes
    assert req.audio_only is audio_only
    assert two_available.discover_calls[0][1].interactive is True
    assert result.stdout.strip() == ""


def test_interactive_default_skips_forbidden_source(fake, tty):
    result = invoke(["download", URL], input="\n\n")
    assert result.exit_code == 0, result.stderr
    assert fake.requests[0].source_indexes == (1,)


def test_interactive_reprompts_on_invalid(two_available, tty):
    result = invoke(["download", URL], input="7\n1\ny\n")
    assert result.exit_code == 0, result.stderr
    assert "Source 7 not found" in result.stderr
    assert two_available.requests[0].source_indexes == (1,)


def test_interactive_decline_downloads_nothing(two_available, tty):
    result = invoke(["download", URL], input="1\nn\n")
    assert result.exit_code == 0
    assert "Nothing downloaded" in result.stderr
    assert two_available.requests == []


def test_interactive_eof_cancels(two_available, tty):
    result = invoke(["download", URL], input="")
    assert result.exit_code == 130
    assert two_available.requests == []


def test_interactive_with_streams_does_not_prompt(two_available, tty):
    result = invoke(["download", URL, "--streams", "1"], input="")
    assert result.exit_code == 0, result.stderr
    assert "Start download?" not in result.stderr


def test_interactive_model_consent(two_available, tty):
    result = invoke(["download", URL, "--transcribe", "local"], input="1\ny\ny\n")
    assert result.exit_code == 0, result.stderr
    assert two_available.confirm_answers == [True]
    assert "about 460 MB" in result.stderr and MODEL_MESSAGE[:20] in result.stderr
    declined = invoke(["download", URL, "--transcribe", "local"], input="1\ny\nn\n")
    assert declined.exit_code == 0
    assert two_available.confirm_answers == [True, False]


@pytest.mark.parametrize(("answer", "layout"), [
    ("", Layout.SEPARATE), ("side-by-side", Layout.SIDE_BY_SIDE), ("tb", Layout.TOP_BOTTOM)])
def test_interactive_layout_prompt(two_available, tty, answer, layout):
    result = invoke(["download", URL], input=f"all\n{answer}\ny\n")
    assert result.exit_code == 0, result.stderr
    assert "Layout?" in result.stderr
    (req,) = two_available.requests
    assert req.layout is layout
    if layout is not Layout.SEPARATE:
        flat = " ".join(result.stderr.replace("│", " ").split())  # undo panel wrapping
        assert "re-encoded" in flat and "Source 1 + Source 2" in flat


def test_layout_flags_map_to_request(two_available):
    result = invoke(["download", URL, "--streams", "all", "--layout", "top-bottom",
                     "--layout-order", "2,1", "--keep-separate", "--encoder", "x264",
                     "--non-interactive"])
    assert result.exit_code == 0, result.stderr
    (req,) = two_available.requests
    assert (req.layout, req.layout_order, req.keep_separate, req.encoder) == \
        (Layout.TOP_BOTTOM, (2, 1), True, Encoder.X264)
    assert "Layout?" not in result.stderr  # never prompts with explicit flags


@pytest.mark.parametrize("args, message", [
    (["--streams", "1", "--layout", "side-by-side"], "exactly two"),
    (["--streams", "audio", "--layout", "side-by-side"], "cannot be combined"),
    (["--streams", "all", "--layout-order", "2,1"], "only apply with"),
    (["--streams", "all", "--layout", "side-by-side", "--layout-order", "1"],
     "--layout-order"),
])
def test_layout_usage_errors(two_available, args, message):
    result = invoke(["download", URL, *args, "--non-interactive"])
    assert result.exit_code == 2, result.stderr
    assert message in " ".join(result.stderr.split())
    assert not two_available.requests  # nothing downloaded
