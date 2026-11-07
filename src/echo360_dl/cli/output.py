"""Output plumbing: stream routing, JSON documents, error reporting and serialisers.

Contract:
* ``--json`` writes exactly one JSON document to stdout
  (``{"schema_version": 1, "command": ..., "ok": ...}``) built via
  :func:`echo360_dl.models.to_jsonable`; every string in it is also passed through
  :func:`echo360_dl.redact.scrub` as a second line of defence.
* Human output, prompts, progress and errors go to stderr.
* A session is *interactive* only when stdin and stderr are TTYs, ``--non-interactive``
  was not given and ``--json`` is off.
"""

from __future__ import annotations

import contextlib
import json
import re
import signal
import sys
import threading
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from rich.console import Console, RenderableType
from rich.text import Text

from echo360_dl.cli.theme import (
    Glyphs,
    color_disabled,
    encoding_is_utf8,
    glyphs_for,
    make_console,
    status_block,
    status_label,
)
from echo360_dl.errors import Echo360DLError, ExitCode, UnavailableError
from echo360_dl.models import SCHEMA_VERSION, Transcript, to_jsonable
from echo360_dl.redact import scrub

CANCEL_MESSAGE = "cancelled — rerun to resume"


def isatty(stream: Any) -> bool:
    """TTY check used for interactivity (tests monkeypatch this)."""
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False


@dataclass(frozen=True)
class UIOptions:
    json: bool = False
    quiet: bool = False
    verbose: bool = False
    no_color: bool = False
    ascii: bool = False
    non_interactive: bool = False
    plain: bool = False  # never show the live display, even on a terminal


class UI:
    """Per-invocation output context. Streams are resolved at construction time so
    test runners that swap ``sys.std*`` are honoured."""

    def __init__(self, command: str, options: UIOptions, *, stdout: IO[str] | None = None,
                 stderr: IO[str] | None = None, stdin: IO[str] | None = None,
                 width: int | None = None) -> None:
        self.command = command
        self.options = options
        self.stdout = stdout or sys.stdout
        self.stderr = stderr or sys.stderr
        self.stdin = stdin or sys.stdin
        self.ascii = options.ascii or not encoding_is_utf8(self.stderr)
        self.glyphs: Glyphs = glyphs_for(ascii_only=self.ascii)
        self.no_color = color_disabled(options.no_color)
        self.err: Console = make_console(self.stderr, no_color=self.no_color,
                                         ascii_only=self.ascii, width=width)
        self._json_emitted = False
        #: Set by the live display while it runs: prints must go above it.
        self.live_printer: Callable[[Callable[[], None]], None] | None = None

    def _print(self, *renderables: RenderableType, **kwargs: Any) -> None:
        printer = self.live_printer
        if printer is not None:
            printer(lambda: self.err.print(*renderables, **kwargs))
        else:
            self.err.print(*renderables, **kwargs)

    # -- modes -------------------------------------------------------------------------
    @property
    def json(self) -> bool:
        return self.options.json

    @property
    def quiet(self) -> bool:
        return self.options.quiet

    @property
    def verbose(self) -> bool:
        return self.options.verbose and not self.options.quiet

    @property
    def interactive(self) -> bool:
        return (not self.options.non_interactive and not self.options.json
                and isatty(self.stdin) and isatty(self.stderr))

    @property
    def live(self) -> bool:
        """Animated output allowed: stderr is a real terminal (FORCE_COLOR alone is not
        enough - a pipe never gets carriage-return animation)."""
        try:
            raw_tty = bool(self.stderr.isatty())
        except (AttributeError, ValueError, OSError):
            raw_tty = False
        return raw_tty and self.err.is_terminal and not self.err.is_dumb_terminal

    @property
    def human(self) -> bool:
        """Regular human-readable output is wanted (not --json, not --quiet)."""
        return not self.options.json and not self.options.quiet

    # -- human output --------------------------------------------------------------------
    def show(self, *renderables: RenderableType) -> None:
        if self.human:
            self._print(*renderables)

    def always(self, *renderables: RenderableType, soft_wrap: bool = False) -> None:
        """Printed even with --quiet (errors, final outputs list)."""
        if not self.options.json:
            self._print(*renderables, soft_wrap=soft_wrap)

    def paths(self, paths: list[Path]) -> None:
        """Final outputs list: one unwrapped path per line (copy/paste friendly)."""
        for path in paths:
            self.always(Text(str(path)), soft_wrap=True)

    def warn(self, message: str) -> None:
        if not self.options.quiet:
            self._print(status_block("WARN", scrub(message)))

    def debug(self, message: str) -> None:
        if self.verbose:
            self._print(Text("debug: " + scrub(message), style="muted"))

    # -- JSON --------------------------------------------------------------------------
    def emit_json(self, ok: bool, **payload: Any) -> None:
        document = {"schema_version": SCHEMA_VERSION, "command": self.command, "ok": ok}
        document.update(payload)
        data = scrub_strings(to_jsonable(document))
        self.stdout.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        self.stdout.flush()
        self._json_emitted = True

    # -- failures ----------------------------------------------------------------------
    def fail(self, err: Echo360DLError) -> int:
        message, hint = scrub(err.message), scrub(err.hint) if err.hint else None
        if self.json:
            data = err.to_dict()
            data.update(message=message, hint=hint)
            self.emit_json(False, error=data)
            self.stderr.write(f"error: {message}\n")
            self.stderr.flush()
        else:
            self._print_error(message, hint)
        return int(err.exit_code)

    def cancelled(self) -> int:
        message = CANCEL_MESSAGE if not self.ascii else CANCEL_MESSAGE.replace("—", "-")
        if self.json:
            self.emit_json(False, error={"kind": "cancelled", "message": "Cancelled by user.",
                                         "hint": "Rerun the same command to resume.",
                                         "exit_code": int(ExitCode.CANCELLED)})
            self.stderr.write(message + "\n")
            self.stderr.flush()
        else:
            line = status_label("CANCELLED")
            line.append(" " + message)
            self.err.print(line)
        return int(ExitCode.CANCELLED)

    def unexpected(self, exc: BaseException) -> int:
        summary = scrub(f"{type(exc).__name__}: {exc}").splitlines()
        message = "Unexpected error: " + (summary[0] if summary else type(exc).__name__)
        hint = None if self.options.verbose else "Rerun with --verbose for details."
        if self.options.verbose:
            trace = scrub("".join(traceback.format_exception(exc)))
            self.stderr.write(trace)
            self.stderr.flush()
        if self.json:
            self.emit_json(False, error={"kind": "internal", "message": message, "hint": hint,
                                         "exit_code": int(ExitCode.ERROR)})
            self.stderr.write(f"error: {message}\n")
            self.stderr.flush()
        else:
            self._print_error(message, hint)
        return int(ExitCode.ERROR)

    def _print_error(self, message: str, hint: str | None) -> None:
        self.err.print(status_block("FAIL", message, hint))


def scrub_strings(data: Any) -> Any:
    """Apply :func:`scrub` to every string in a JSON-able structure."""
    if isinstance(data, str):
        return scrub(data)
    if isinstance(data, dict):
        return {k: scrub_strings(v) for k, v in data.items()}
    if isinstance(data, list):
        return [scrub_strings(v) for v in data]
    return data


# --------------------------------------------------------------------------------------
# Cancellation
# --------------------------------------------------------------------------------------


@contextlib.contextmanager
def cancellation(stderr: IO[str] | None = None,
                 notify: Callable[[str], None] | None = None) -> Iterator[threading.Event]:
    """Install a SIGINT handler for the duration of a cancellable operation.

    The first Ctrl-C sets the returned event (the pipeline stops at the next safe
    point and keeps resumable work); a second Ctrl-C raises KeyboardInterrupt.
    ``notify`` (e.g. the live console) shows the "stopping" message instead of a raw
    write to ``stderr``.
    """
    event = threading.Event()
    if threading.current_thread() is not threading.main_thread():
        yield event
        return
    handler = make_sigint_handler(event, stderr or sys.stderr, notify)
    previous = signal.signal(signal.SIGINT, handler)
    # Terminal closed / kill: unwind immediately so ``finally`` blocks remove temp files
    # and cookie jars (SIGKILL cannot be caught; stale jars are swept on the next run).
    others = {}
    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is not None:
            others[sig] = signal.signal(sig, _raise_interrupt)
    try:
        yield event
    finally:
        signal.signal(signal.SIGINT, previous)
        for sig, old in others.items():
            signal.signal(sig, old)


def _raise_interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


STOPPING_MESSAGE = "Stopping after the current step... press Ctrl-C again to abort."


def make_sigint_handler(event: threading.Event, stderr: IO[str],
                        notify: Callable[[str], None] | None = None
                        ) -> Callable[[int, object], None]:
    def handler(signum: int, frame: object) -> None:
        if event.is_set():
            raise KeyboardInterrupt
        event.set()
        with contextlib.suppress(Exception):
            if notify is not None:
                notify(STOPPING_MESSAGE)
                return
            stderr.write(f"\n{STOPPING_MESSAGE}\n")
            stderr.flush()

    return handler


# --------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------


def fmt_bytes(value: float | None) -> str:
    if value is None:
        return "?"
    size = float(value)
    for unit in ("B", "kB", "MB", "GB"):
        if abs(size) < 1000 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    return f"{size:.1f} TB"  # pragma: no cover - loop always returns


def fmt_amount(value: float | None, unit: str) -> str:
    if value is None:
        return "?"
    if unit == "bytes":
        return fmt_bytes(value)
    if unit == "seconds":
        return fmt_clock(max(value, 0.0))
    return f"{value:.0f}"


def fmt_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:  # NaN-safe
        return ""
    total = round(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_clock(seconds: float | None) -> str:
    """Elapsed/remaining time as ``MM:SS`` (``H:MM:SS`` from one hour); ``""`` if unknown."""
    if seconds is None or seconds < 0 or seconds != seconds:
        return ""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fmt_span(seconds: float | None) -> str:
    """A measured duration: ``182 ms``, ``12.3 s``, ``02:15``."""
    if seconds is None or seconds < 0 or seconds != seconds:
        return ""
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    return fmt_clock(seconds)


def fmt_rate(bytes_per_second: float | None) -> str:
    """``6.8 MB/s`` (decimal units, like :func:`fmt_bytes`)."""
    if bytes_per_second is None or bytes_per_second < 0 or bytes_per_second != bytes_per_second:
        return ""
    return fmt_bytes(bytes_per_second) + "/s"


def _scale(value: float) -> tuple[float, str]:
    for factor, unit in ((1e9, "GB"), (1e6, "MB"), (1e3, "kB")):
        if abs(value) >= factor:
            return factor, unit
    return 1.0, "B"


def fmt_pair(done: float | None, total: float | None, unit: str = "bytes") -> str:
    """``162.0 / 250.0 MB`` with one unit taken from the total; just the amount when the
    total is unknown."""
    if unit != "bytes":
        if total is None:
            return fmt_amount(done or 0, unit)
        return f"{fmt_amount(done or 0, unit)} / {fmt_amount(total, unit)}"
    if total is None:
        return fmt_bytes(done or 0)
    factor, name = _scale(float(total))
    if name == "B":
        return f"{done or 0:.0f} / {total:.0f} B"
    return f"{(done or 0) / factor:.1f} / {total / factor:.1f} {name}"


_SLUG = re.compile(r"[^A-Za-z0-9]+")


def slug(text: str | None, limit: int = 60) -> str:
    return _SLUG.sub("-", text or "").strip("-")[:limit].strip("-")


# --------------------------------------------------------------------------------------
# Transcript serialisers (``transcript`` command)
# --------------------------------------------------------------------------------------

TRANSCRIPT_FORMATS = ("vtt", "srt", "txt", "json")


def serialise_transcript(transcript: Transcript, fmt: str) -> str:
    """Render ``transcript`` as vtt/srt/txt/json via the media writers (one implementation
    shared with download sidecars).

    Timed formats require timed cues: timestamps are never invented for untimed text.
    Platform files already in the requested format are passed through unchanged.
    """
    from echo360_dl.media import subtitles

    if fmt not in TRANSCRIPT_FORMATS:
        raise ValueError(f"unknown transcript format {fmt!r}")
    if fmt in {"vtt", "srt"} and not transcript.timed:
        raise UnavailableError(
            "This transcript has no timestamps, so it cannot be written as "
            f"{fmt.upper()}.", hint="Use --format txt or --format json, or --transcribe local.")
    # Pass platform files through unchanged for caption formats only; JSON always uses
    # echo360-dl's own documented schema (the platform's JSON shape is unverified and may
    # embed signed URLs).
    if (transcript.raw is not None and transcript.raw_format == fmt and fmt != "json"
            and not transcript.generated):
        return transcript.raw.decode("utf-8", errors="replace")
    writer = {"vtt": subtitles.to_vtt, "srt": subtitles.to_srt, "txt": subtitles.to_txt,
              "json": subtitles.to_json}[fmt]
    return writer(transcript)
