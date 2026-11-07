"""Chapters: user JSON files, native/interval sets, validation, ffmetadata export.

Chapter JSON schema (version 1)::

    {"version": 1, "chapters": [{"start": "00:00:00", "title": "Introduction"},
                                {"start": "00:08:30", "end": "00:20:00", "title": "..."}]}

``start``/``end`` accept ``HH:MM:SS(.mmm)``, ``MM:SS`` or seconds (number). Exported
JSON uses the same schema plus ``provenance`` (and ``source_detail``), so it can be
read back with :func:`load_chapter_file`.

Times are seconds on the recording timeline: 0 is the start of the HLS media
timeline, which is where the audio starts (see ``media.mux`` for the timing decision).
"""

from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path
from typing import Any

from echo360_dl.errors import UnavailableError, ValidationError
from echo360_dl.models import Chapter, ChapterMode, ChapterModeKind, ChapterProvenance, ChapterSet
from echo360_dl.timecode import format_clock, format_duration, parse_time

#: Tolerance (s) when comparing chapter bounds with the media duration.
DURATION_TOLERANCE = 0.5
_MAX_FILE_BYTES = 2_000_000


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _time(value: Any, where: str, problems: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        problems.append(f"{where}: must be a time string or a number of seconds")
        return None
    try:
        return parse_time(value)
    except ValueError as exc:
        problems.append(f"{where}: {exc}")
        return None


def parse_chapter_data(obj: Any, source_detail: str | None) -> ChapterSet:
    """Validate decoded chapter JSON and build a USER_FILE :class:`ChapterSet`.

    Raises :class:`ValidationError` listing every problem found.
    """
    problems: list[str] = []
    if not isinstance(obj, dict):
        raise ValidationError("Chapter file must be a JSON object with 'version' and "
                              "'chapters'.")
    if obj.get("version") != 1 or isinstance(obj.get("version"), bool):
        problems.append(f"version must be 1 (got {obj.get('version')!r})")
    items = obj.get("chapters")
    if not isinstance(items, list):
        problems.append("'chapters' must be a list")
        items = []
    elif not items:
        problems.append("'chapters' is empty")
    chapters: list[Chapter] = []
    for n, item in enumerate(items, start=1):
        where = f"chapter {n}"
        if not isinstance(item, dict):
            problems.append(f"{where}: must be an object")
            continue
        start = _time(item.get("start"), f"{where} start", problems) \
            if "start" in item else None
        if "start" not in item:
            problems.append(f"{where}: missing 'start'")
        end = _time(item["end"], f"{where} end", problems) \
            if item.get("end") is not None else None
        title = item.get("title")
        if not isinstance(title, str):
            problems.append(f"{where}: 'title' must be a string")
            title = ""
        if start is not None:
            chapters.append(Chapter(start=start, title=title.strip(), end=end))
    if not problems:
        problems = validate_chapters(chapters, None)
    if problems:
        name = f" ({source_detail})" if source_detail else ""
        raise ValidationError(f"Invalid chapter file{name}: " + "; ".join(problems),
                              hint="See the chapter JSON schema in the README.")
    return ChapterSet(chapters=chapters, provenance=ChapterProvenance.USER_FILE,
                      source_detail=source_detail)


def load_chapter_file(path: Path) -> ChapterSet:
    path = Path(path).expanduser()
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            raise ValidationError(f"Chapter file {path.name} is too large.")
        text = path.read_bytes().decode("utf-8-sig")
    except FileNotFoundError:
        raise ValidationError(f"Chapter file not found: {path}") from None
    except UnicodeDecodeError:
        raise ValidationError(f"Chapter file {path.name} is not UTF-8 text.") from None
    except OSError as exc:
        raise ValidationError(f"Cannot read chapter file {path.name}: {exc.strerror}") from None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"Chapter file {path.name} is not valid JSON (line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}).") from None
    return parse_chapter_data(obj, path.name)


# --------------------------------------------------------------------------------------
# Validation / resolution
# --------------------------------------------------------------------------------------


def validate_chapters(chapters: list[Chapter], duration: float | None) -> list[str]:
    """Rules: non-negative finite times, strictly increasing starts, end > start,
    end <= next start, within ``duration`` (when known), non-empty trimmed titles
    without control characters (newlines are allowed and escaped on export)."""
    problems: list[str] = []
    for n, ch in enumerate(chapters, start=1):
        where = f"chapter {n}"
        if not math.isfinite(ch.start) or ch.start < 0:
            problems.append(f"{where}: start must be a non-negative time")
        if not ch.title or not ch.title.strip():
            problems.append(f"{where}: title is empty")
        elif any(unicodedata.category(c) == "Cc" and c != "\n" for c in ch.title):
            problems.append(f"{where}: title contains control characters")
        if ch.end is not None and (not math.isfinite(ch.end) or ch.end <= ch.start):
            problems.append(f"{where}: end must be after start")
        if n < len(chapters):
            nxt = chapters[n]
            if nxt.start <= ch.start:
                problems.append(f"chapter {n + 1}: start must be after chapter {n}'s start "
                                "(starts must strictly increase)")
            elif ch.end is not None and ch.end > nxt.start + 1e-9:
                problems.append(f"{where}: end overlaps the start of chapter {n + 1}")
        if duration is not None:
            if ch.start >= duration:
                problems.append(f"{where}: starts at {format_clock(ch.start, millis=True)}, "
                                f"at or after the end of the media "
                                f"({format_clock(duration, millis=True)})")
            elif ch.end is not None and ch.end > duration + DURATION_TOLERANCE:
                problems.append(f"{where}: ends after the end of the media "
                                f"({format_clock(duration, millis=True)})")
    return problems


def resolve_chapters(chapters: list[Chapter], duration: float | None) -> list[Chapter]:
    """Fill missing ends from the next chapter's start and the media duration.

    An explicit end is clamped to ``duration``. The last chapter keeps ``end=None``
    only when the duration is unknown.
    """
    out: list[Chapter] = []
    for i, ch in enumerate(chapters):
        end = ch.end
        if end is None:
            end = chapters[i + 1].start if i + 1 < len(chapters) else duration
        if end is not None and duration is not None:
            end = min(end, duration)
        out.append(Chapter(start=ch.start, title=ch.title, end=end))
    return out


def interval_chapters(duration: float, interval: float) -> ChapterSet:
    """Neutral "Part N" chapters every ``interval`` seconds (GENERATED_INTERVAL).

    A trailing remainder shorter than 1 s does not get its own chapter.
    """
    if not (math.isfinite(duration) and duration > 0):
        raise UnavailableError("Interval chapters need a known media duration.")
    if not (math.isfinite(interval) and interval > 0):
        raise ValidationError("Chapter interval must be a positive number of seconds.")
    chapters: list[Chapter] = []
    start = 0.0
    n = 1
    while start < duration - 1.0 or not chapters:
        end = min(start + interval, duration)
        chapters.append(Chapter(start=start, title=f"Part {n}", end=end))
        start += interval
        n += 1
    if chapters[-1].end is not None and chapters[-1].end < duration:
        last = chapters[-1]
        chapters[-1] = Chapter(start=last.start, title=last.title, end=duration)
    return ChapterSet(chapters=chapters, provenance=ChapterProvenance.GENERATED_INTERVAL,
                      source_detail=f"every {format_duration(interval)}")


# --------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------


def ffmetadata_escape(value: str) -> str:
    """Escape ``= ; # \\`` and newlines for ffmetadata (backslash first)."""
    out = value.replace("\\", "\\\\")
    for ch in ("=", ";", "#"):
        out = out.replace(ch, "\\" + ch)
    return out.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\\n")


def to_ffmetadata(chapters: list[Chapter], *, title: str | None = None) -> str:
    """ffmetadata text with TIMEBASE=1/1000. Every chapter needs a resolved end."""
    lines = [";FFMETADATA1"]
    if title:
        lines.append(f"title={ffmetadata_escape(title)}")
    for ch in chapters:
        if ch.end is None:
            raise ValueError("chapters must be resolved (end set) before export")
        start_ms = round(ch.start * 1000)
        end_ms = max(round(ch.end * 1000), start_ms + 1)
        lines += ["", "[CHAPTER]", "TIMEBASE=1/1000", f"START={start_ms}", f"END={end_ms}",
                  f"title={ffmetadata_escape(ch.title)}"]
    return "\n".join(lines) + "\n"


def chapters_to_json(chapter_set: ChapterSet) -> dict[str, Any]:
    return {
        "version": 1,
        "provenance": chapter_set.provenance.value,
        "source_detail": chapter_set.source_detail,
        "chapters": [
            {"start": format_clock(ch.start, millis=True), "title": ch.title,
             **({"end": format_clock(ch.end, millis=True)} if ch.end is not None else {})}
            for ch in chapter_set.chapters
        ],
    }


# --------------------------------------------------------------------------------------
# Mode handling
# --------------------------------------------------------------------------------------


def choose_chapters(mode: ChapterMode, native: ChapterSet | None,
                    duration: float | None) -> tuple[ChapterSet | None, list[str]]:
    """Apply the chapter precedence rules.

    * FILE: the user file wins; a warning notes overridden native chapters.
    * AUTO: native chapters only (never fabricated); absence is reported.
    * INTERVAL: generated "Part N" chapters (needs ``duration``).
    * NONE: nothing.
    """
    warnings: list[str] = []
    has_native = bool(native and native.chapters)
    if mode.kind is ChapterModeKind.NONE:
        return None, warnings
    if mode.kind is ChapterModeKind.FILE:
        if mode.file is None:
            raise ValidationError("Chapter mode 'file' needs a chapter file.")
        chosen = load_chapter_file(mode.file)
        if has_native:
            warnings.append(f"Native Echo360 chapters ({len(native.chapters)}) were "  # type: ignore[union-attr]
                            f"overridden by {chosen.source_detail}.")
        return chosen, warnings
    if mode.kind is ChapterModeKind.INTERVAL:
        if duration is None:
            raise UnavailableError("Interval chapters need a known media duration.")
        interval = mode.interval or 600.0
        if has_native:
            warnings.append("Native Echo360 chapters exist but --chapters interval was "
                            "requested; using generated interval chapters.")
        return interval_chapters(duration, interval), warnings
    if has_native:
        return native, warnings
    warnings.append("No native chapters were found for this lesson, so none were embedded "
                    "(use --chapters-file or --chapters interval to add some).")
    return None, warnings
