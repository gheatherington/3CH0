"""Parsing and formatting of time values used in chapters, subtitles and CLI options."""

from __future__ import annotations

import math
import re

_CLOCK = re.compile(r"^(?:(?:(\d+):)?(\d{1,2}):)?(\d{1,2}(?:[.,]\d+)?)$")
_UNITS = re.compile(r"^(?:(\d+(?:\.\d+)?)h)?\s*(?:(\d+(?:\.\d+)?)m)?\s*(?:(\d+(?:\.\d+)?)s)?$")


def parse_time(value: str | int | float) -> float:
    """Parse ``"HH:MM:SS(.mmm)"``, ``"MM:SS"``, ``"SS(.mmm)"``, ``"1h2m3s"``/``"10m"`` or a
    number of seconds. Raises ValueError for negative, empty or malformed input."""
    if isinstance(value, bool):
        raise ValueError("boolean is not a time value")
    if isinstance(value, int | float):
        seconds = float(value)
    else:
        text = value.strip()
        if not text:
            raise ValueError("empty time value")
        if text.startswith("-"):
            raise ValueError(f"negative time value: {value!r}")
        match = _CLOCK.match(text)
        if match:
            hours, minutes, secs = match.groups()
            if hours is not None and minutes is None:
                raise ValueError(f"malformed time value: {value!r}")
            if (hours is not None or minutes is not None) and float(
                    secs.replace(",", ".")) >= 60:
                raise ValueError(f"seconds field must be < 60 in {value!r}")
            if hours is not None and int(minutes) >= 60:
                raise ValueError(f"minutes field must be < 60 in {value!r}")
            seconds = (int(hours or 0) * 3600 + int(minutes or 0) * 60
                       + float(secs.replace(",", ".")))
        else:
            match = _UNITS.match(text.lower())
            if not match or not any(match.groups()):
                raise ValueError(f"malformed time value: {value!r}")
            h, m, s = (float(g) if g else 0.0 for g in match.groups())
            seconds = h * 3600 + m * 60 + s
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"time value must be a finite non-negative number: {value!r}")
    return seconds


def format_clock(seconds: float, *, millis: bool = False, sep: str = ".") -> str:
    """``3725.5 -> "01:02:05"`` (or ``"01:02:05.500"`` with millis)."""
    if seconds < 0 or not math.isfinite(seconds):
        raise ValueError("seconds must be finite and non-negative")
    total_ms = round(seconds * 1000)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    base = f"{h:02d}:{m:02d}:{s:02d}"
    return f"{base}{sep}{ms:03d}" if millis else base


def format_duration(seconds: float | None) -> str:
    """Compact human duration: ``"1h 02m"``, ``"4m 05s"``, ``"12s"``, ``"—"``."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "—"
    total = round(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"
