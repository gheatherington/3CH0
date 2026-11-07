"""3CH0 console: telemetry derivations, live layout, keys, summary and branding."""

from __future__ import annotations

import io
import os
import time
import tomllib
from datetime import datetime
from pathlib import Path

import pytest

from echo360_dl.cli import dashboard, telemetry
from echo360_dl.cli.keys import DETAILS, DOWN, ENTER, UP, KeyReader, parse_keys
from echo360_dl.cli.output import UI, UIOptions, fmt_clock, fmt_pair, fmt_rate, fmt_span
from echo360_dl.cli.progress import LineProgress, LiveProgress, make_progress
from echo360_dl.cli.render import result_view, stream_details
from echo360_dl.cli.telemetry import Telemetry
from echo360_dl.cli.theme import ASCII, UNICODE
from echo360_dl.models import (
    DownloadResult,
    Failure,
    OutputFile,
    OutputKind,
    ResultStatus,
    Stage,
)

from .conftest import CONTENT, QUERY, SECRET_SIG, invoke, make_recording

ROOT = Path(__file__).resolve().parents[2]


class Utf8Buffer(io.StringIO):
    encoding = "utf-8"


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_tel() -> tuple[Telemetry, Clock]:
    clock = Clock()
    return Telemetry(clock=clock, wall=lambda: datetime(2026, 11, 7, 14, 32, 8)), clock


def steady(tel: Telemetry, clock: Clock, task: str, *, seconds: float, rate: float,
           start: float = 0.0, step: float = 0.25) -> float:
    done = start
    for _ in range(int(seconds / step)):
        clock.now += step
        done += rate * step
        tel.task_update(task, completed=done)
    return done


# -- telemetry ----------------------------------------------------------------------------


def test_rates_from_measured_deltas():
    tel, clock = make_tel()
    tel.task_start("download:video-s1q1", "Source 1 video", None, "bytes")
    tel.task_update("download:video-s1q1", completed=0)
    clock.now += 0.5
    tel.task_update("download:video-s1q1", completed=500_000)
    assert tel.snapshot().rates.measuring  # under a second of data: no rate yet
    assert tel.snapshot().rates.current is None
    steady(tel, clock, "download:video-s1q1", seconds=6, rate=1_000_000, start=500_000)
    rates = tel.snapshot().rates
    assert rates.current == pytest.approx(1_000_000, rel=0.05)
    assert rates.average == pytest.approx(1_000_000, rel=0.05)
    assert rates.peak is not None and rates.peak >= rates.current * 0.99
    assert 6 <= len(rates.history) <= 8 and max(rates.history) > 0


def test_resumed_and_reused_bytes_are_not_network_traffic():
    tel, clock = make_tel()
    tel.task_start("download:audio-s0q1", "Audio", None, "bytes")
    tel.task_update("download:audio-s0q1", total=10_000_000)
    clock.now += 1
    tel.task_update("download:audio-s0q1", completed=8_000_000)  # resumed part file
    clock.now += 1
    tel.task_update("download:audio-s0q1", completed=9_000_000)
    tel.task_start("download:video-s1q1", "Source 1 video", None, "bytes")
    tel.task_update("download:video-s1q1", completed=5_000_000, total=5_000_000)
    tel.task_done("download:video-s1q1", ok=True, message="reused from an earlier run")
    snap = tel.snapshot()
    assert snap.rates.transferred == 1_000_000
    assert snap.aggregate.done == 14_000_000 and snap.aggregate.total == 15_000_000
    assert snap.aggregate.reused == 5_000_000
    assert snap.task("download:video-s1q1").state == "reused"


def test_restart_resets_baseline():
    tel, clock = make_tel()
    tel.task_start("download:a", "A", 100, "bytes")
    tel.task_update("download:a", completed=10)
    clock.now += 1
    tel.task_update("download:a", completed=60)
    clock.now += 1
    tel.task_update("download:a", completed=5)  # discarded and restarted
    clock.now += 1
    tel.task_update("download:a", completed=25)
    assert tel.snapshot().rates.transferred == 70


def test_unknown_totals_are_never_guessed():
    tel, clock = make_tel()
    tel.task_queued("download:video-s2q1", "Source 2 video")
    tel.task_start("download:video-s1q1", "Source 1 video", None, "bytes")
    tel.task_update("download:video-s1q1", completed=0, total=1000)
    steady(tel, clock, "download:video-s1q1", seconds=4, rate=100)
    snap = tel.snapshot()
    assert snap.aggregate.total is None  # one transfer is still queued
    assert snap.aggregate.fraction is None and snap.remaining is None
    assert snap.aggregate.queued == 1
    lines = [line.plain for line in dashboard.render(snap, UNICODE, width=100, height=40,
                                                     command="download")]
    start = next(i for i, line in enumerate(lines) if "TRANSFER" in line)
    assert "total not known yet" in lines[start + 1] and "%" not in lines[start + 1]
    assert "remaining —" in lines[start + 1]


def test_failed_transfers_leave_the_total():
    tel, _ = make_tel()
    tel.task_start("download:a", "A", 100, "bytes")
    tel.task_start("download:b", "B", 50, "bytes")
    tel.task_done("download:b", ok=False, message="HTTP 403")
    agg = tel.snapshot().aggregate
    assert agg.total == 100 and agg.tasks == 1
    assert tel.snapshot().task("download:b").state == "failed"


def test_buffers_are_bounded_and_repeats_collapse():
    tel, clock = make_tel()
    for i in range(telemetry.MAX_EVENTS * 3):
        tel.event("info", f"message {i}")
    for _ in range(5):
        tel.event("warn", "same thing")
    for i in range(telemetry.MAX_DEBUG * 3):
        tel.debug(f"debug {i}")
    tel.task_start("download:a", "A", None, "bytes")
    tel.task_update("download:a", completed=0)
    done = 0
    for _ in range(telemetry.HISTORY_SECONDS * 2):
        clock.now += 1
        done += 10
        tel.task_update("download:a", completed=done)
    for i in range(telemetry.MAX_TASKS * 2):
        tel.task_start(f"mux:{i}", f"file {i}", None, "items")
        tel.task_done(f"mux:{i}", ok=True, message=None)
    snap = tel.snapshot()
    assert len(snap.events) <= telemetry.MAX_EVENTS
    assert snap.events[-1].text == "same thing" and snap.events[-1].count == 5
    assert len(snap.debug) == telemetry.MAX_DEBUG
    assert len(snap.rates.history) <= telemetry.HISTORY_SECONDS
    assert len(snap.tasks) <= telemetry.MAX_TASKS + 1


def test_telemetry_scrubs_secrets():
    tel, _ = make_tel()
    secret_url = f"{CONTENT}/s1q1.m3u8{QUERY}"
    tel.set_context(source=secret_url)
    tel.event("warn", f"retrying {secret_url}")
    tel.debug(f"GET {secret_url}")
    tel.task_start("download:a", f"fetch {secret_url}", None, "bytes")
    snap = tel.snapshot()
    text = repr(snap)
    assert SECRET_SIG not in text


# -- dashboard ----------------------------------------------------------------------------


def busy_snapshot(glyphs=UNICODE):
    tel, clock = make_tel()
    tel.set_context(source="https://echo360.ca/lesson/G_x/classroom", title="Lecture 7",
                    target=str(Path.home() / "Downloads"))
    tel.set_streams(stream_details(make_recording(), glyphs))
    for stage in (Stage.DISCOVER, Stage.SELECT, Stage.DOWNLOAD):
        tel.stage(stage)
    tel.task_start("download:video-s1q1", "Source 1 video (1280x720)", None, "bytes")
    tel.task_start("download:audio-s0q1", "Audio (shared)", None, "bytes")
    tel.task_update("download:video-s1q1", completed=0, total=27_000_000)
    tel.task_update("download:audio-s0q1", completed=0, total=3_000_000)
    steady(tel, clock, "download:video-s1q1", seconds=8, rate=2_000_000)
    tel.task_update("download:video-s1q1", rate=2_000_000.0, eta=5.5)
    tel.event("warn", "Skipping Source 2: forbidden")
    return tel.snapshot()


@pytest.mark.parametrize("width", [24, 36, 50, 64, 80, 100, 140])
@pytest.mark.parametrize("height", [6, 12, 40])
@pytest.mark.parametrize("glyphs", [UNICODE, ASCII])
def test_live_view_fits_any_terminal(width, height, glyphs):
    snap = busy_snapshot(glyphs)
    for expanded in (False, True):
        lines = dashboard.render(snap, glyphs, width=width, height=height, command="download",
                                 selected="download:video-s1q1", expanded=expanded, keys=True)
        assert len(lines) <= max(height, 3)
        for line in lines:
            assert line.cell_len <= max(width, 22), line.plain
            assert line.plain == line.plain.rstrip()
            if glyphs is ASCII:
                assert line.plain.isascii(), line.plain


def test_live_view_sections_and_labels():
    lines = [line.plain for line in dashboard.render(
        busy_snapshot(), UNICODE, width=100, height=40, command="download",
        selected="download:video-s1q1", keys=True)]
    text = "\n".join(lines)
    assert lines[0].strip().startswith("3CH0 / download") and "elapsed" in lines[0]
    for title in ("LESSON", "TRANSFER", "STREAMS", "PIPELINE", "EVENTS"):
        assert f"╭─ {title} ─" in text  # every section is a titled box
    assert text.count("╭") == text.count("╰") == 5
    for key in ("URL", "TITLE", "TARGET    ~/Downloads", "current", "average", "peak"):
        assert key in text
    assert "throughput · last" in text and " s" in text  # graph window is labelled
    assert "1280×720 · 30" in text  # FORMAT from the playlist attributes
    assert "▸ Source 1 video" in text  # selection marked by a glyph, not only colour
    assert "[WARN] Skipping Source 2" in text
    assert "ctrl-c stop" in lines[-1]
    assert "discover ✓ → select ✓ → [download] → subtitles" in text  # current stage marked
    for name in ("Source 1 video", "Audio (shared)"):  # a bar on every stream row
        row = next(line for line in lines if name in line and "%" in line)
        assert "━" in row or "─" in row


def test_live_view_without_bytes_has_no_network_section():
    tel, _ = make_tel()
    tel.stage(Stage.DISCOVER)
    tel.task_start("discover", "Watching player traffic", None, "items")
    tel.task_update("discover", completed=2)
    lines = [line.plain for line in dashboard.render(tel.snapshot(), UNICODE, width=80,
                                                     height=30, command="inspect",
                                                     pipeline=False)]
    text = "\n".join(lines)
    assert "current" not in text and "TRANSFER" not in text and "%" not in text
    assert "Watching player traffic" in text and "observing" in text
    assert "3CH0 / inspect · discover" in text and "PIPELINE" not in text
    assert "ctrl-c" not in text  # no key footer without keyboard input


def test_expanded_view_shows_details_and_debug():
    tel, _ = make_tel()
    tel.task_start("download:video-s1q1", "Source 1 video", 100, "bytes")
    tel.task_update("download:video-s1q1", completed=40, rate=10.0, eta=6.0)
    tel.debug("Source 1 video: [hlsnative] Total fragments: 15")
    text = "\n".join(line.plain for line in dashboard.render(
        tel.snapshot(), UNICODE, width=100, height=40, command="download",
        selected="download:video-s1q1", expanded=True, keys=True))
    assert "DETAIL" in text and "download:video-s1q1" in text and "(40.0%)" in text
    assert "REPORTED  10 B/s" in text and "from yt-dlp" in text and "Total fragments" in text
    assert "d hide details" in text


def test_stream_details_keys_match_pipeline_task_ids():
    from echo360_dl.media.download import Component, component_task_id

    details = stream_details(make_recording(), UNICODE)
    video = Component("video-s1q1", "video", "s1q1", "u", "Source 1 video")
    assert component_task_id(video) in details
    assert details["download:video-s1q1"] == ("1280×720 · 30", "H.264")
    assert details["download:audio-s0q1"][1] == "AAC"  # from the paired CODECS attribute


def test_formatting_units():
    assert fmt_clock(24) == "00:24" and fmt_clock(3725) == "1:02:05" and fmt_clock(None) == ""
    assert fmt_span(0.182) == "182 ms" and fmt_span(12.34) == "12.3 s"
    assert fmt_rate(6_800_000) == "6.8 MB/s"
    assert fmt_pair(162e6, 250e6) == "162.0 / 250.0 MB" and fmt_pair(3e6, None) == "3.0 MB"
    assert fmt_pair(5, 10, "items") == "5 / 10"


# -- keys and the live sink ---------------------------------------------------------------


def test_parse_keys():
    assert parse_keys(b"\x1b[A\x1b[Bjkd\r") == [UP, DOWN, DOWN, UP, DETAILS, ENTER]
    assert parse_keys(b"\x1b[C x") == []  # other keys are ignored


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="needs a pty")
def test_key_reader_restores_terminal_mode():
    import termios

    master, slave = os.openpty()
    try:
        stream = os.fdopen(slave, "r", closefd=False)
        before = termios.tcgetattr(slave)
        seen: list[str] = []
        reader = KeyReader(stream, seen.append)
        assert reader.start()
        assert not termios.tcgetattr(slave)[3] & termios.ECHO  # cbreak: no echo
        assert termios.tcgetattr(slave)[3] & termios.ISIG  # Ctrl-C still signals
        os.write(master, b"\x1b[Bd")
        for _ in range(50):
            if len(seen) >= 2:
                break
            time.sleep(0.02)
        reader.stop()
        assert seen == [DOWN, DETAILS]
        after = termios.tcgetattr(slave)
        pendin = getattr(termios, "PENDIN", 0)  # set by the kernel on mode changes
        assert after[3] & ~pendin == before[3] & ~pendin
        assert after[:3] == before[:3] and after[4:] == before[4:]
    finally:
        os.close(master)
        os.close(slave)


def test_live_sink_keys_move_selection_and_toggle_details():
    ui = UI("download", UIOptions(), stdout=io.StringIO(), stderr=io.StringIO(),
            stdin=io.StringIO(), width=80)
    sink = LiveProgress(ui)
    for task in ("download:a", "download:b"):
        sink.telemetry.task_start(task, task, 10, "bytes")
    snap = sink.telemetry.snapshot()
    assert sink.selected_id(snap) == "download:a"  # follows the first active task
    sink._on_key(DOWN)
    assert sink.selected_id(sink.telemetry.snapshot()) == "download:b"
    sink._on_key(DOWN)  # stays at the end
    assert sink.selected_id(sink.telemetry.snapshot()) == "download:b"
    sink._on_key(DETAILS)
    assert sink.expanded
    sink._on_key(ENTER)
    assert not sink.expanded


def test_line_progress_has_no_trailing_padding():
    err = io.StringIO()
    ui = UI("download", UIOptions(), stdout=io.StringIO(), stderr=err, stdin=io.StringIO(),
            width=80)
    sink = LineProgress(ui)
    sink.stage(Stage.DOWNLOAD)
    sink.task_start("download:a", "Source 1 video", total=10)
    sink.task_done("download:a")
    sink.warn("a warning")
    for line in err.getvalue().splitlines():
        assert line == line.rstrip()
    assert "[OK]      Source 1 video" in err.getvalue()


# -- summary ------------------------------------------------------------------------------


def test_summary_reports_measured_transfer_and_dedupes_warnings(tmp_path):
    tel, clock = make_tel()
    tel.task_start("download:a", "A", 10_000_000, "bytes")
    tel.task_update("download:a", completed=0)
    steady(tel, clock, "download:a", seconds=5, rate=2_000_000)
    result = DownloadResult(
        ResultStatus.PARTIAL, lesson_id="x",
        outputs=[OutputFile(tmp_path / "a.mp4", OutputKind.VIDEO, size=10_000_000)],
        failures=[Failure("download", "Source 2", "forbidden", "HTTP 403",
                          hint="Check it plays in the browser.")],
        warnings=["No native chapters."])
    ui = UI("download", UIOptions(), stdout=io.StringIO(), stderr=Utf8Buffer(),
            stdin=io.StringIO(), width=100)
    ui.err.print(result_view(result, ui, tel.snapshot(),
                             ["Skipping Source 3: forbidden", "Source 2: HTTP 403",
                              "No native chapters."]))
    text = ui.stderr.getvalue()
    flat = " ".join(text.replace("│", " ").split())
    assert "STATUS [PARTIAL] Download partly complete · 1 output" in flat
    for title in ("SUMMARY", "FAILURES", "WARNINGS"):
        assert f"╭─ {title} " in text  # each part in its own box
    assert "TARGET" in flat and str(tmp_path) in "".join(flat.split()) and "a.mp4" in flat
    assert "10.0 MB transferred in 00:05" in flat and "average 2.0 MB/s" in flat
    assert "next: Check it plays in the browser." in flat
    assert flat.count("No native chapters.") == 1 and flat.count("HTTP 403") == 1
    assert "Skipping Source 3: forbidden" in flat


# -- branding and compatibility -----------------------------------------------------------


def test_both_commands_are_installed_entry_points():
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["3ch0"] == scripts["echo360-dl"] == "echo360_dl.cli.app:main"


def test_help_uses_the_new_brand():
    result = invoke(["--help"])
    assert result.exit_code == 0
    assert "3CH0" in result.output and "Usage: 3ch0" in " ".join(result.output.split())
    assert "echo360-dl" in result.output  # the alias is documented


# -- flicker-free redraw, early totals, encode estimate, --plain --------------------------


class TTYUtf8(io.StringIO):
    encoding = "utf-8"

    def isatty(self) -> bool:
        return True


def test_live_region_rewrites_only_changed_lines_without_blanking():
    from rich.console import Console
    from rich.text import Text

    from echo360_dl.cli.live import ERASE_EOL, SHOW_CURSOR, SYNC_OFF, SYNC_ON, LiveRegion
    from echo360_dl.cli.theme import THEME

    stream = TTYUtf8()
    console = Console(file=stream, theme=THEME, width=60, height=20, force_terminal=True,
                      color_system=None)
    frame = [Text("header"), Text("row one"), Text("row two")]
    region = LiveRegion(console, lambda w, h: [t.copy() for t in frame],
                        refresh_per_second=0.01)  # no background repaint during the test
    region.start()
    first = stream.getvalue()
    assert first.count(ERASE_EOL) == 3 and "row two" in first
    stream.seek(0)
    stream.truncate()
    region.refresh()
    assert stream.getvalue() == ""  # unchanged frame: nothing is written at all
    frame[1] = Text("row ONE")
    region.refresh()
    update = stream.getvalue()
    assert update.startswith(SYNC_ON) and update.endswith(SYNC_OFF)  # applied atomically
    assert "row ONE" in update and "header" not in update and "row two" not in update
    assert "\x1b[2K" not in update  # lines are overwritten in place, never erased first
    region.stop()
    assert stream.getvalue().endswith(SHOW_CURSOR)


def test_queued_sizes_give_the_overall_total_from_the_start():
    tel, _ = make_tel()
    tel.task_queued("download:audio-s0q1", "Audio")
    tel.task_queued("download:video-s1q1", "Source 1 video")
    tel.task_queued("download:video-s2q1", "Source 2 video", "bytes", 40_000_000)
    tel.task_start("download:audio-s0q1", "Audio", None, "bytes")
    tel.task_update("download:audio-s0q1", completed=0, total=10_000_000)
    tel.task_start("download:video-s1q1", "Source 1 video", None, "bytes")
    tel.task_update("download:video-s1q1", completed=0, total=50_000_000)
    agg = tel.snapshot().aggregate
    assert agg.total == 100_000_000 and agg.queued == 1  # known before source 2 starts
    tel.task_start("download:video-s2q1", "Source 2 video", None, "bytes")
    assert tel.snapshot().task("download:video-s2q1").total == 40_000_000  # size kept
    text = "\n".join(line.plain for line in dashboard.render(
        tel.snapshot(), UNICODE, width=100, height=40, command="download"))
    assert "0.0%" in text and "total not known" not in text


def test_encoding_gets_a_bar_speed_and_estimate():
    tel, clock = make_tel()
    tel.stage(Stage.VERIFY)
    tel.task_start("encode:x.mp4", "Encoding Source 1 + Source 2 (side by side) (libx264)",
                   3240.0, "seconds")
    for _ in range(8):
        clock.now += 0.5
        tel.task_update("encode:x.mp4", completed=(clock.now - 1000) * 3.0, total=3240.0)
    task = tel.snapshot().task("encode:x.mp4")
    assert task.speed == pytest.approx(3.0) and task.left == pytest.approx((3240 - 12) / 3)
    lines = [line.plain for line in dashboard.render(tel.snapshot(), UNICODE, width=100,
                                                     height=40, command="download")]
    start = next(i for i, line in enumerate(lines) if "╭─ ENCODE ─" in line)
    assert "─" * 40 in lines[start + 2] and "0.4%" in lines[start + 2]  # the bar
    assert "remaining 17:56" in lines[start + 2]
    assert "00:12 / 54:00 of media · speed 3.0×" in lines[start + 3]
    row = next(line for line in lines if "Encoding" in line and "encoding" in line)
    assert "17:56" in row  # the table row has the estimate too


def test_plain_option_never_animates():
    tty_ui = UI("t", UIOptions(plain=True), stdout=io.StringIO(), stderr=TTYUtf8(),
                stdin=io.StringIO())
    assert isinstance(make_progress(tty_ui), LineProgress)
    result = invoke(["doctor", "--plain"])
    assert result.exit_code in (0, 6)
