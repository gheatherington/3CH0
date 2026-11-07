"""ffmpeg / ffprobe / yt-dlp tool access.

Everything that shells out to ffmpeg or ffprobe lives here so the rest of the media
package deals in Python values. Tool stderr is passed through
:func:`threech0.redact.scrub` before it reaches an exception message.
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from threech0.errors import (
    CancelledError,
    DependencyError,
    Echo360DLError,
    VerificationError,
)
from threech0.models import Chapter, StreamInfo, VerifyReport
from threech0.redact import scrub

FFMPEG_HINT = "Install ffmpeg (includes ffprobe) with: brew install ffmpeg"
YTDLP_HINT = "yt-dlp is a Python dependency of 3ch0; reinstall with: uv sync"

#: Handler names written by muxers by default; not useful as a track title.
_GENERIC_HANDLER = re.compile(r"(?i)^(?:\w*handler|core media \w+|iso media file.*|)$")


class FFmpegError(Echo360DLError):
    """ffmpeg exited with an error (message is scrubbed tool output)."""

    kind = "ffmpeg"


# --------------------------------------------------------------------------------------
# Versions / availability
# --------------------------------------------------------------------------------------


def _cli_version(tool: str) -> str | None:
    exe = shutil.which(tool)
    if not exe:
        return None
    flag = "--version" if tool == "yt-dlp" else "-version"
    try:
        out = subprocess.run([exe, flag], capture_output=True, text=True, timeout=20,
                             check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    first = (out.stdout or "").strip().splitlines()[:1]
    if not first:
        return None
    match = re.match(rf"{re.escape(tool)} version (\S+)", first[0])
    return match.group(1) if match else first[0].strip()


def _ytdlp_version() -> str | None:
    try:
        from yt_dlp.version import __version__
    except ImportError:
        return _cli_version("yt-dlp")
    return str(__version__)


def tool_versions() -> dict[str, str | None]:
    """Versions of ffmpeg, ffprobe and yt-dlp (``None`` when unavailable).

    yt-dlp is reported from the Python package the downloader actually uses.
    """
    return {"ffmpeg": _cli_version("ffmpeg"), "ffprobe": _cli_version("ffprobe"),
            "yt-dlp": _ytdlp_version()}


def require_tools(*names: str) -> None:
    """Raise :class:`DependencyError` naming every missing tool."""
    missing: list[str] = []
    for name in names:
        if name == "yt-dlp":
            try:
                import yt_dlp  # noqa: F401
            except ImportError:
                missing.append(name)
        elif not shutil.which(name):
            missing.append(name)
    if missing:
        hints = []
        if any(m in {"ffmpeg", "ffprobe"} for m in missing):
            hints.append(FFMPEG_HINT)
        if "yt-dlp" in missing:
            hints.append(YTDLP_HINT)
        raise DependencyError(f"Required tool not found: {', '.join(missing)}.",
                              hint="; ".join(hints))


# --------------------------------------------------------------------------------------
# Running ffmpeg
# --------------------------------------------------------------------------------------


def run_ffmpeg(args: list[str], *, cancel: threading.Event | None = None,
               poll_interval: float = 0.1,
               on_progress: Callable[[float], None] | None = None) -> None:
    """Run ``ffmpeg`` with ``args``; poll ``cancel`` while it runs.

    ``on_progress`` receives the output position in seconds, parsed from ffmpeg's
    machine-readable ``-progress`` stream (real progress, not an estimate).

    On cancellation (event or KeyboardInterrupt) the process is terminated and
    :class:`CancelledError` (or KeyboardInterrupt) propagates; the caller removes
    any partial output file.
    """
    require_tools("ffmpeg")
    progress_args = ["-progress", "pipe:1", "-nostats"] if on_progress else []
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", *progress_args, *args]
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE if on_progress else subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    errors: list[bytes] = []
    readers = [threading.Thread(target=lambda: errors.append(proc.stderr.read()),  # type: ignore[union-attr]
                                daemon=True)]
    if on_progress is not None:
        readers.append(threading.Thread(target=_read_progress,
                                        args=(proc.stdout, on_progress), daemon=True))
    for reader in readers:
        reader.start()
    try:
        while True:
            if cancel is not None and cancel.is_set():
                _stop(proc)
                raise CancelledError("Cancelled while running ffmpeg.")
            try:
                proc.wait(timeout=poll_interval)
                break
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        _stop(proc)
        raise
    finally:
        for reader in readers:
            reader.join(timeout=5)
    stderr = b"".join(errors)
    if proc.returncode != 0:
        text = scrub(stderr.decode("utf-8", "replace")).strip()
        lines = text.splitlines()[-6:]
        raise FFmpegError("ffmpeg failed: " + (" | ".join(lines) or f"exit {proc.returncode}"))


def _read_progress(stream: Any, on_progress: Callable[[float], None]) -> None:
    for raw in iter(stream.readline, b""):
        line = raw.decode("ascii", "replace").strip()
        if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
            # Both keys are microseconds in current ffmpeg ("out_time_ms" is misnamed).
            value = line.split("=", 1)[1]
            if value.isdigit():
                with contextlib.suppress(Exception):
                    on_progress(int(value) / 1_000_000)


def _stop(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


# --------------------------------------------------------------------------------------
# ffprobe
# --------------------------------------------------------------------------------------


def ffprobe_json(path: Path, *extra: str) -> dict[str, Any]:
    """``ffprobe -show_format -show_streams -show_chapters`` as a dict.

    Raises :class:`VerificationError` when the file cannot be read as media.
    """
    require_tools("ffprobe")
    cmd = ["ffprobe", "-v", "error", "-of", "json", "-show_format", "-show_streams",
           "-show_chapters", *extra, str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=600, check=False)
    except subprocess.TimeoutExpired:
        raise VerificationError(f"ffprobe timed out reading {path.name}.") from None
    if out.returncode != 0:
        msg = scrub(out.stderr.decode("utf-8", "replace")).strip().splitlines()[-3:]
        raise VerificationError(f"ffprobe could not read {path.name}: {' | '.join(msg)}")
    try:
        data = json.loads(out.stdout.decode("utf-8", "replace") or "{}")
    except json.JSONDecodeError as exc:
        raise VerificationError(f"ffprobe returned invalid JSON for {path.name}: {exc}") from None
    return data if isinstance(data, dict) else {}


def count_packets(path: Path, selector: str = "v:0") -> int | None:
    """Number of packets in one stream (demux only, no decoding)."""
    require_tools("ffprobe")
    cmd = ["ffprobe", "-v", "error", "-count_packets", "-select_streams", selector,
           "-show_entries", "stream=nb_read_packets", "-of", "json", str(path)]
    out = subprocess.run(cmd, capture_output=True, timeout=1800, check=False)
    if out.returncode != 0:
        return None
    try:
        streams = json.loads(out.stdout or b"{}").get("streams") or []
        return int(streams[0]["nb_read_packets"]) if streams else None
    except (ValueError, KeyError, TypeError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def stream_title(tags: dict[str, str]) -> str | None:
    """Track title from container tags (MKV ``title``; MP4 ``name``/``handler_name``)."""
    lowered = {k.lower(): v for k, v in tags.items()}
    for key in ("title", "name", "handler_name"):
        value = (lowered.get(key) or "").strip()
        if value and not (key == "handler_name" and _GENERIC_HANDLER.match(value)):
            return value
    return None


def stream_info(raw: dict[str, Any]) -> StreamInfo:
    tags = {str(k): str(v) for k, v in (raw.get("tags") or {}).items()}
    lowered = {k.lower(): v for k, v in tags.items()}
    disposition = raw.get("disposition") or {}
    return StreamInfo(
        index=_int(raw.get("index")) or 0,
        codec_type=str(raw.get("codec_type") or "unknown"),
        codec_name=raw.get("codec_name"),
        language=lowered.get("language"),
        title=stream_title(tags),
        default=bool(disposition.get("default")),
        width=_int(raw.get("width")),
        height=_int(raw.get("height")),
        duration=_float(raw.get("duration")),
        start_time=_float(raw.get("start_time")),
    )


def report_from_probe(data: dict[str, Any]) -> VerifyReport:
    streams = [stream_info(s) for s in data.get("streams") or []]
    chapters = []
    for ch in data.get("chapters") or []:
        start = _float(ch.get("start_time"))
        if start is None:
            continue
        title = str((ch.get("tags") or {}).get("title") or "")
        chapters.append(Chapter(start=start, end=_float(ch.get("end_time")), title=title))
    duration = _float((data.get("format") or {}).get("duration"))
    problems = [] if streams else ["no media streams found"]
    return VerifyReport(ok=not problems, duration=duration, streams=streams,
                        chapters=chapters, problems=problems)


def probe_report(path: Path) -> VerifyReport:
    """Streams (codec, language, title, default disposition, start time), chapters and
    duration of a media file. Unreadable files give ``ok=False`` with a problem;
    a missing ffprobe raises :class:`DependencyError`."""
    path = Path(path)
    if not path.is_file():
        return VerifyReport(ok=False, problems=[f"{path.name}: file not found"])
    try:
        data = ffprobe_json(path)
    except VerificationError as exc:
        return VerifyReport(ok=False, problems=[exc.message])
    return report_from_probe(data)
