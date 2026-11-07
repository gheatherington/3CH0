"""Stable, filesystem-safe output names that carry the lesson identity.

Layout (``[short_id]`` always present so names stay unique and greppable)::

    <course> - <YYYY-MM-DD> - <title> [<short_id>] - <Source 1>.mp4
    <course> - <YYYY-MM-DD> - <title> [<short_id>] - audio.m4a
    <course> - <YYYY-MM-DD> - <title> [<short_id>].en.vtt        (sidecars)

With a user-chosen name (``--name``) the lesson part is replaced by that name, and a
download producing exactly one media file drops the ``- <label>`` part::

    <name>.mp4                     (one source, audio only, or one combined picture)
    <name> - <Source 1>.mp4        (several media files)
    <name>.en.vtt                  (sidecars)

Unknown parts are omitted; without a title the head is "Echo360 lesson". Names are
NFC-normalised Unicode with path separators, control/bidi characters, Windows
reserved characters/names, leading dots and trailing spaces/dots removed, and are
capped at :data:`MAX_NAME_BYTES` UTF-8 bytes without splitting a code point.
"""

from __future__ import annotations

import re
import unicodedata

from echo360_dl.models import DownloadRequest, Layout, Recording

MAX_NAME_BYTES = 180
#: Budget for the base name; extensions such as ``.en-CA.transcript.platform.json``
#: are appended after it and stay within :data:`MAX_NAME_BYTES`.
BASE_BUDGET = 130  # + " - " + label (<= 40) + extension stays within MAX_NAME_BYTES
_LABEL_BUDGET = 40

_REPLACEMENTS = {
    ":": " - ", "/": "-", "\\": "-", "|": "-", '"': "'", "<": "(", ">": ")",
    "?": "", "*": "",
}
_BIDI = set("‎‏‪‫‬‭‮⁦⁧⁨⁩")
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                     *(f"LPT{i}" for i in range(1, 10))}
_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})")
DEFAULT_HEAD = "Echo360 lesson"


def truncate_utf8(text: str, max_bytes: int) -> str:
    """Cut ``text`` to at most ``max_bytes`` UTF-8 bytes on a code-point boundary."""
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    return data[:max(max_bytes, 0)].decode("utf-8", "ignore")


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    out = []
    for ch in text:
        if ch in _REPLACEMENTS:
            out.append(_REPLACEMENTS[ch])
        elif ch in _BIDI or unicodedata.category(ch) in {"Cc", "Cs", "Co", "Zl", "Zp"}:
            out.append(" ")
        elif ch == "﻿":
            continue
        else:
            out.append(ch)
    text = re.sub(r"\s+", " ", "".join(out))
    text = re.sub(r"(?: - ){2,}", " - ", text)
    return text


def _trim(text: str) -> str:
    text = text.strip().lstrip(". ")
    return text.rstrip(" .")


def safe_component(text: str | None, *, fallback: str = "untitled",
                   max_bytes: int = MAX_NAME_BYTES) -> str:
    """One safe path component (no separators) of at most ``max_bytes`` bytes."""
    cleaned = _trim(_clean(text or ""))
    cleaned = _trim(truncate_utf8(cleaned, max_bytes))
    if not cleaned:
        cleaned = fallback
    stem = cleaned.split(".")[0].strip().upper()
    if stem in _WINDOWS_RESERVED:
        cleaned = "_" + cleaned
    return _trim(truncate_utf8(cleaned, max_bytes)) or fallback


def _date(recording: Recording) -> str | None:
    match = _DATE.match(recording.recorded_at or "")
    return match.group(1) if match else None


def _head(recording: Recording) -> str:
    parts = [recording.course, _date(recording), recording.title or DEFAULT_HEAD]
    return " - ".join(_trim(_clean(p)) for p in parts if p and _trim(_clean(p)))


def _short_id(recording: Recording) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "", recording.short_id) or "lesson"


def _fit(head: str, tail: str, budget: int) -> str:
    room = budget - len(tail.encode("utf-8"))
    head = _trim(truncate_utf8(head, room)) or DEFAULT_HEAD
    return safe_component(head + tail, fallback=DEFAULT_HEAD, max_bytes=budget)


def lesson_tag(recording: Recording) -> str:
    """The id that appears as ``[<tag>]`` in every output name of this lesson."""
    return _short_id(recording)


def lesson_basename(recording: Recording) -> str:
    """``<course> - <date> - <title> [<short_id>]`` (used for sidecars)."""
    return _fit(_head(recording), f" [{_short_id(recording)}]", BASE_BUDGET)


def media_basename(recording: Recording, label: str | None) -> str:
    """``<lesson basename> - <label>``: the lesson part is exactly the sidecar base name,
    so a long label (e.g. "Source 1 + Source 2 (side by side)") never shortens the title
    and every file of a lesson shares one prefix."""
    label_part = safe_component(label or "audio", fallback="audio", max_bytes=_LABEL_BUDGET)
    return f"{lesson_basename(recording)} - {label_part}"


def custom_basename(name: str) -> str:
    """A user-chosen base name made filesystem-safe and capped like the automatic one."""
    return safe_component(name, fallback=DEFAULT_HEAD, max_bytes=BASE_BUDGET)


def single_media_output(request: DownloadRequest) -> bool:
    """True when the request asks for exactly one media file (decided from the request,
    not from what succeeds, so names never depend on another source failing)."""
    if request.audio_only:
        return True
    if request.layout is not Layout.SEPARATE:
        return not request.keep_separate
    return len(request.source_indexes or request.recording.sources) == 1


def request_basename(request: DownloadRequest) -> str:
    """Base name for this request's sidecars: ``--name`` if given, else the lesson name."""
    if request.name:
        return custom_basename(request.name)
    return lesson_basename(request.recording)


def request_media_basename(request: DownloadRequest, label: str | None) -> str:
    """Media base name for this request. Without ``--name`` this is
    :func:`media_basename`; with it, ``<name>`` for a single output, else
    ``<name> - <label>``."""
    if not request.name:
        return media_basename(request.recording, label)
    base = custom_basename(request.name)
    if single_media_output(request):
        return base
    label_part = safe_component(label or "audio", fallback="audio", max_bytes=_LABEL_BUDGET)
    return f"{base} - {label_part}"


def language_tag(code: str | None) -> str:
    """Filename-safe language tag (``en``, ``en-CA``; ``und`` when unknown)."""
    if not code:
        return "und"
    tag = re.sub(r"[^A-Za-z0-9-]", "-", code.strip().replace("_", "-")).strip("-")
    return tag[:16] or "und"


def with_suffix(base: str, *parts: str) -> str:
    """``base`` plus dot-separated ``parts`` (``base.en.vtt``), capped in bytes."""
    suffix = "".join("." + safe_component(p, fallback="x", max_bytes=24) for p in parts if p)
    room = MAX_NAME_BYTES - len(suffix.encode("utf-8"))
    return _trim(truncate_utf8(base, room)) + suffix
