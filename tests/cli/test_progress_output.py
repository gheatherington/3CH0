from __future__ import annotations

import io
import re
import signal
import threading

import pytest

from echo360_dl.cli import output as cli_output
from echo360_dl.cli.output import (
    UI,
    UIOptions,
    cancellation,
    fmt_bytes,
    make_sigint_handler,
    scrub_strings,
)
from echo360_dl.cli.progress import (
    LineProgress,
    LiveProgress,
    QuietProgress,
    make_progress,
)
from echo360_dl.cli.theme import status_label
from echo360_dl.models import ProgressSink, Stage

from .conftest import CONTENT, QUERY, SECRET_SIG, SECRET_UID, invoke


class TTYBuffer(io.StringIO):
    encoding = "utf-8"

    def isatty(self) -> bool:
        return True


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def make_ui(**options) -> tuple[UI, io.StringIO]:
    err = io.StringIO()
    return UI("test", UIOptions(**options), stdout=io.StringIO(), stderr=err,
              stdin=io.StringIO(), width=80), err


def test_sinks_satisfy_protocol():
    ui, _ = make_ui()
    for sink in (LineProgress(ui), LiveProgress(ui), QuietProgress()):
        assert isinstance(sink, ProgressSink)


def test_line_progress_unknown_total_never_shows_percent():
    ui, err = make_ui()
    clock = Clock()
    sink = LineProgress(ui, interval=5, clock=clock)
    sink.stage(Stage.DOWNLOAD, "Source 1")
    sink.task_start("a", "Source 1 video", total=None)
    for step in range(1, 30):
        clock.now = step
        sink.task_update("a", completed=step * 1_000_000)
    sink.task_done("a")
    text = err.getvalue()
    assert "%" not in text
    assert "so far" in text
    assert "\r" not in text
    updates = [line for line in text.splitlines() if "so far" in line]
    assert 3 <= len(updates) <= 6  # throttled, not one line per update
    assert "STAGE     3/6 download" in text


def test_line_progress_known_total_rate_eta():
    ui, err = make_ui()
    clock = Clock()
    sink = LineProgress(ui, interval=1, clock=clock)
    sink.task_start("a", "audio", total=10_000_000)
    clock.now = 2
    sink.task_update("a", completed=5_000_000, rate=1_000_000.0, eta=5.0)
    text = err.getvalue()
    assert "5.0 / 10.0 MB  50%" in text
    assert "1.0 MB/s" in text and "remaining 00:05" in text


def test_stage_headers_dedupe():
    ui, err = make_ui()
    sink = LineProgress(ui)
    sink.stage(Stage.DISCOVER)
    sink.stage(Stage.DISCOVER, "still discovering")
    sink.stage(Stage.SELECT)
    text = err.getvalue()
    assert text.count("1/6 discover") == 1  # one header; the repeat is a message line
    assert "still discovering" in text
    assert "2/6 select" in text


def test_progress_messages_are_scrubbed_and_debug_needs_verbose():
    secret_url = f"{CONTENT}/s1q1.m3u8{QUERY}"
    ui, err = make_ui()
    sink = LineProgress(ui)
    sink.info(f"fetching {secret_url}")
    sink.warn(f"retrying {secret_url}")
    sink.debug("hidden without verbose")
    sink.task_start("t", f"download {secret_url}")
    sink.task_done("t", ok=False, message=f"failed Signature={SECRET_SIG}")
    text = err.getvalue()
    assert SECRET_SIG not in text and SECRET_UID not in text
    assert "hidden without verbose" not in text
    assert "[WARN]" in text and "[FAIL]" in text
    vui, verr = make_ui(verbose=True)
    LineProgress(vui).debug(f"GET {secret_url}")
    assert "debug: GET" in verr.getvalue() and SECRET_SIG not in verr.getvalue()


def test_make_progress_selection():
    ui, _ = make_ui(quiet=True)
    assert isinstance(make_progress(ui), QuietProgress)
    ui, _ = make_ui()
    assert isinstance(make_progress(ui), LineProgress)  # not a terminal
    tty_ui = UI("t", UIOptions(), stdout=io.StringIO(), stderr=TTYBuffer(), stdin=io.StringIO())
    assert isinstance(make_progress(tty_ui), LiveProgress)
    json_ui = UI("t", UIOptions(json=True), stdout=io.StringIO(), stderr=TTYBuffer(),
                 stdin=io.StringIO())
    assert isinstance(make_progress(json_ui), LineProgress)


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)


def test_live_progress_lifecycle_on_terminal():
    err = TTYBuffer()
    ui = UI("download", UIOptions(ascii=True), stdout=io.StringIO(), stderr=err,
            stdin=io.StringIO(), width=60)
    with LiveProgress(ui) as sink:
        sink.stage(Stage.DOWNLOAD)
        sink.task_start("a", "Source 1 video", total=None)
        sink.task_update("a", completed=1000)
        with sink.paused():
            pass
        sink.task_done("a")
        sink.task_start("b", "audio", total=100)
        sink.task_done("b", ok=False, message="HTTP 403")
        sink.warn("something odd")
    text = _plain(err.getvalue())
    assert "[WARN]" in text and "something odd" in text  # warnings are durable
    events = [e.text for e in sink.telemetry.snapshot().events]
    assert "finished Source 1 video" in events
    assert "audio: HTTP 403" in events
    assert "\x1b[?25h" in err.getvalue()  # cursor shown again after the live display


def test_live_progress_restores_terminal_on_interrupt():
    err = TTYBuffer()
    ui = UI("download", UIOptions(), stdout=io.StringIO(), stderr=err, stdin=io.StringIO(),
            width=80)
    with pytest.raises(KeyboardInterrupt), LiveProgress(ui) as sink:
        sink.task_start("a", "Source 1 video", total=10)
        raise KeyboardInterrupt
    assert sink._region is None
    out = err.getvalue()
    assert out.rindex("\x1b[?25h") > out.rindex("\x1b[?25l")  # cursor hidden, then restored


def test_cancellation_notify_replaces_raw_write():
    event = threading.Event()
    stream = io.StringIO()
    seen: list[str] = []
    handler = make_sigint_handler(event, stream, seen.append)
    handler(signal.SIGINT, None)
    assert event.is_set() and stream.getvalue() == ""
    assert "Ctrl-C again" in seen[0]


def test_sigint_handler_first_sets_event_second_raises():
    event = threading.Event()
    stream = io.StringIO()
    handler = make_sigint_handler(event, stream)
    handler(signal.SIGINT, None)
    assert event.is_set()
    assert "Ctrl-C again" in stream.getvalue()
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGINT, None)


def test_cancellation_restores_previous_handler():
    before = signal.getsignal(signal.SIGINT)
    with cancellation(io.StringIO()) as event:
        assert signal.getsignal(signal.SIGINT) is not before
        signal.raise_signal(signal.SIGINT)
        assert event.is_set()
    assert signal.getsignal(signal.SIGINT) is before


def test_ui_interactive_rules(monkeypatch):
    monkeypatch.setattr(cli_output, "isatty", lambda stream: True)
    assert make_ui()[0].interactive is True
    assert make_ui(json=True)[0].interactive is False
    assert make_ui(non_interactive=True)[0].interactive is False
    monkeypatch.setattr(cli_output, "isatty", lambda stream: False)
    assert make_ui()[0].interactive is False


def test_no_color_on_terminal_emits_no_escape_codes(monkeypatch):
    err = TTYBuffer()
    ui = UI("t", UIOptions(no_color=True), stdout=io.StringIO(), stderr=err,
            stdin=io.StringIO())
    ui.err.print(status_label("OK"))
    assert err.getvalue() == "[OK]\n"
    monkeypatch.setenv("NO_COLOR", "1")
    err2 = TTYBuffer()
    UI("t", UIOptions(), stdout=io.StringIO(), stderr=err2, stdin=io.StringIO()).err.print(
        status_label("FAIL"))
    assert "\x1b" not in err2.getvalue()
    err3 = TTYBuffer()
    monkeypatch.delenv("NO_COLOR")
    UI("t", UIOptions(), stdout=io.StringIO(), stderr=err3, stdin=io.StringIO()).err.print(
        status_label("FAIL"))
    assert "\x1b[" in err3.getvalue() and "[FAIL]" in err3.getvalue()


def test_non_utf8_stream_forces_ascii():
    raw = io.TextIOWrapper(io.BytesIO(), encoding="latin-1")
    ui = UI("t", UIOptions(), stdout=io.StringIO(), stderr=raw, stdin=io.StringIO())
    assert ui.ascii is True and ui.glyphs.check == "OK"


def test_scrub_strings_and_bytes():
    data = scrub_strings({"a": [f"{CONTENT}/x{QUERY}"], "b": 3})
    assert SECRET_SIG not in str(data) and data["b"] == 3
    assert fmt_bytes(999) == "999 B" and fmt_bytes(27_000_000) == "27.0 MB"


@pytest.mark.parametrize("args", [
    [], ["doctor"], ["inspect"], ["download"], ["transcript"], ["auth"], ["auth", "login"],
    ["auth", "status"], ["auth", "logout"], ["chapters"], ["chapters", "validate"],
    ["chapters", "show"],
])
def test_help_renders(args):
    result = invoke([*args, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage" in result.output


def test_version():
    result = invoke(["--version"])
    assert result.exit_code == 0 and result.stdout.startswith("3ch0 ")


def test_force_color_on_a_pipe_does_not_animate(monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    ui, _ = make_ui()
    assert ui.live is False
    assert isinstance(make_progress(ui), LineProgress)


def test_main_json_mode_reports_option_errors_as_json(capsys):
    import json as _json

    from echo360_dl.cli.app import main

    with pytest.raises(SystemExit) as info:
        main(["download", "https://echo360.ca/lesson/x/classroom", "--json",
              "--container", "avi"])
    assert info.value.code == 2
    out, err = capsys.readouterr()
    doc = _json.loads(out)
    assert doc["ok"] is False and doc["error"]["kind"] == "usage"
    assert doc["command"] == "download" and "avi" in doc["error"]["message"]
    assert "\x1b[" not in out and "error:" in err


def test_main_json_mode_keeps_command_exit_codes(capsys, tmp_path):
    from echo360_dl.cli.app import main

    bad = tmp_path / "bad.json"
    bad.write_text('{"version": 1, "chapters": [{"start": "00:00:05", "title": ""}]}')
    with pytest.raises(SystemExit) as info:
        main(["chapters", "validate", str(bad), "--json"])
    assert info.value.code == 8


def test_main_cancel_message_respects_ascii(capsys, monkeypatch):
    from echo360_dl.cli import app as app_module

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(app_module, "app", interrupted)
    with pytest.raises(SystemExit) as info:
        app_module.main(["doctor", "--ascii"])
    assert info.value.code == 130
    assert capsys.readouterr().err.isascii()
