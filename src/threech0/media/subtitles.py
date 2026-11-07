"""Subtitle/transcript parsing, validation and export.

* Input: WebVTT, SRT, 3ch0 JSON (the :func:`to_json` schema) and plain text.
  Plain text becomes **untimed** cues - timestamps are never invented.
* Encodings: UTF-8 (with or without BOM), UTF-16 (BOM or BOM-less detection), with
  CRLF/CR line endings normalised. Undecodable/binary input raises
  :class:`ValidationError`.
* Speakers: WebVTT ``<v Name>`` is parsed into :attr:`Cue.speaker` and written back
  as ``<v Name>``; SRT/TXT render speakers as ``Name: text``.
* Overlap policy: WebVTT and SRT allow overlapping cues (two speakers at once), so
  :func:`validate_cues` accepts them unless ``allow_overlap=False``. MP4 ``mov_text``
  cannot show overlapping samples; :func:`normalize_cues` trims an overlapping cue's
  end to the next cue's start for that case (end times only ever move earlier).
"""

from __future__ import annotations

import html
import json
import math
import re
import unicodedata
from collections.abc import Iterable

from threech0.errors import TranscriptUnavailableError, ValidationError
from threech0.models import Cue, Transcript, TranscriptProvenance
from threech0.timecode import format_clock

_TS = r"(?:(\d+):)?(\d{1,2}):(\d{1,2})[.,](\d{1,3})"
_TIMING = re.compile(rf"^\s*{_TS}\s*-->\s*{_TS}(?:\s+(.*))?$")
_VOICE = re.compile(r"<v(?:\.[^\s>]*)?(?:\s+([^>]*))?>", re.IGNORECASE)
_TAG = re.compile(r"</?[^>\n]*>")
_CONTROL_OK = {"\n", "\t"}

LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German", "it": "Italian",
    "pt": "Portuguese", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
    "ru": "Russian", "hi": "Hindi", "nl": "Dutch", "sv": "Swedish", "pl": "Polish",
    "tr": "Turkish", "vi": "Vietnamese", "fa": "Persian", "uk": "Ukrainian", "he": "Hebrew",
    "el": "Greek", "cs": "Czech", "da": "Danish", "fi": "Finnish", "no": "Norwegian",
    "ro": "Romanian", "hu": "Hungarian", "id": "Indonesian", "th": "Thai", "pa": "Punjabi",
    "ta": "Tamil", "ur": "Urdu", "bn": "Bengali", "tl": "Tagalog", "ms": "Malay",
}


# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------


def language_name(code: str | None) -> str:
    if not code:
        return "Unknown language"
    primary = re.split(r"[-_]", code.strip())[0].lower()
    return LANGUAGE_NAMES.get(primary, code)


def track_title(t: Transcript) -> str:
    """Container track title, e.g. "English — Echo360 captions"."""
    name = language_name(t.language)
    kind = {
        TranscriptProvenance.PLATFORM_CAPTIONS: "Echo360 captions",
        TranscriptProvenance.PLATFORM_TRANSCRIPT: "Echo360 transcript",
        TranscriptProvenance.GENERATED_LOCAL: "generated (local)",
        TranscriptProvenance.USER_FILE: "user file",
    }[t.provenance]
    if t.provenance is not TranscriptProvenance.GENERATED_LOCAL and t.label and \
            "automatic" in t.label.lower():
        kind = "Echo360 automatic captions"  # Echo360's own speech recognition (isAutomated)
    if not t.language:
        return kind[0].upper() + kind[1:]  # no "Unknown language — " prefix
    return f"{name} — {kind}"


# --------------------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------------------


def _decode(data: bytes) -> str:
    if not data or not data.strip(b"\x00 \t\r\n\xef\xbb\xbf\xff\xfe"):
        raise ValidationError("Subtitle data is empty.")
    for bom, codec in ((b"\xef\xbb\xbf", "utf-8"), (b"\xff\xfe", "utf-16-le"),
                       (b"\xfe\xff", "utf-16-be")):
        if data.startswith(bom):
            try:
                return data[len(bom):].decode(codec)
            except UnicodeDecodeError as exc:
                raise ValidationError(f"Subtitle data is not valid {codec}: {exc}") from None
    sample = data[:4096]
    if len(data) % 2 == 0 and sample.count(b"\x00") > len(sample) // 4:
        codec = "utf-16-le" if sample[1::2].count(b"\x00") > sample[0::2].count(b"\x00") \
            else "utf-16-be"
        try:
            return data.decode(codec)
        except UnicodeDecodeError:
            raise ValidationError("Subtitle data looks like UTF-16 but does not decode.") \
                from None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(
            f"Subtitle data is not UTF-8 or UTF-16 text (byte {exc.start}).",
            hint="Convert the file to UTF-8 first.") from None


def _bad_controls(text: str) -> list[str]:
    return sorted({f"U+{ord(c):04X}" for c in text
                   if unicodedata.category(c) == "Cc" and c not in _CONTROL_OK})


def _normalise_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    return unicodedata.normalize("NFC", text)


def detect_format(text: str) -> str:
    stripped = text.lstrip()
    if stripped.startswith("WEBVTT"):
        return "vtt"
    if stripped.startswith(("{", "[")):
        return "json"
    if re.search(rf"(?m)^{_TS}\s*-->\s*{_TS}", text):
        return "srt"
    return "txt"


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


def _seconds(h: str | None, m: str, s: str, ms: str) -> float:
    if int(m) >= 60 or int(s) >= 60:
        raise ValueError("minutes/seconds out of range")
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def _parse_timing(line: str, lineno: int) -> tuple[float, float, str]:
    match = _TIMING.match(line)
    if not match:
        raise ValidationError(f"Malformed cue timing on line {lineno}: {line.strip()[:60]!r}")
    g = match.groups()
    try:
        return _seconds(*g[0:4]), _seconds(*g[4:8]), (g[8] or "").strip()
    except ValueError:
        raise ValidationError(f"Invalid timestamp on line {lineno}.") from None


def _payload_to_cue(lines: list[str], start: float, end: float) -> Cue | None:
    raw = "\n".join(lines).strip()
    speaker = None
    voice = _VOICE.search(raw)
    if voice:
        speaker = html.unescape((voice.group(1) or "").strip()) or None
    text = html.unescape(_TAG.sub("", raw)).replace(" ", " ")
    text = "\n".join(line.strip() for line in text.split("\n")).strip()
    if not text:
        return None
    return Cue(text=text, start=start, end=end, speaker=speaker)


def _blocks(text: str) -> list[tuple[int, list[str]]]:
    blocks: list[tuple[int, list[str]]] = []
    current: list[str] = []
    first = 0
    for no, line in enumerate(text.split("\n"), start=1):
        if line.strip():
            if not current:
                first = no
            current.append(line)
        elif current:
            blocks.append((first, current))
            current = []
    if current:
        blocks.append((first, current))
    return blocks


def _parse_vtt(text: str) -> list[Cue]:
    blocks = _blocks(text)
    if not blocks or not blocks[0][1][0].lstrip().startswith("WEBVTT"):
        raise ValidationError("WebVTT data must start with 'WEBVTT'.")
    header = blocks[0][1]
    cues: list[Cue] = []
    # The header block may (invalidly but commonly) run straight into a cue.
    rest = [(blocks[0][0] + i, header[i:]) for i in range(1, len(header))
            if "-->" in header[i]][:1]
    for lineno, lines in rest + blocks[1:]:
        head = lines[0].strip()
        if head.startswith(("NOTE", "STYLE", "REGION")) and "-->" not in head:
            continue
        idx = 0 if "-->" in lines[0] else 1
        if idx >= len(lines) or "-->" not in lines[idx]:
            raise ValidationError(f"Cue without timing near line {lineno}.")
        start, end, _settings = _parse_timing(lines[idx], lineno + idx)
        cue = _payload_to_cue(lines[idx + 1:], start, end)
        if cue:
            cues.append(cue)
    return cues


def _parse_srt(text: str) -> list[Cue]:
    cues: list[Cue] = []
    for lineno, lines in _blocks(text):
        idx = next((i for i, line in enumerate(lines[:2]) if "-->" in line), None)
        if idx is None:
            raise ValidationError(f"SRT block without timing near line {lineno}.")
        start, end, _ = _parse_timing(lines[idx], lineno + idx)
        payload = [re.sub(r"\{\\[^}]*\}", "", line) for line in lines[idx + 1:]]
        cue = _payload_to_cue(payload, start, end)
        if cue:
            # "Name: text" is left as text: SRT has no speaker syntax and guessing
            # would misread ordinary text such as "Question: ...".
            cues.append(cue)
    return cues


def _parse_json(text: str) -> list[Cue]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON transcript: {exc}") from None
    items = obj.get("cues") if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        raise ValidationError("JSON transcript must be a list of cues or {\"cues\": [...]}.")
    cues: list[Cue] = []
    for n, item in enumerate(items, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValidationError(f"JSON cue {n} must be an object with a 'text' string.")
        start, end = item.get("start"), item.get("end")
        for value in (start, end):
            if value is not None and (isinstance(value, bool)
                                      or not isinstance(value, int | float)
                                      or not math.isfinite(value)):
                raise ValidationError(f"JSON cue {n}: start/end must be numbers or null.")
        speaker = item.get("speaker")
        cues.append(Cue(text=item["text"], start=None if start is None else float(start),
                        end=None if end is None else float(end),
                        speaker=speaker if isinstance(speaker, str) and speaker else None))
    return cues


def _parse_txt(text: str) -> list[Cue]:
    return [Cue(text=line.strip()) for line in text.split("\n") if line.strip()]


def parse_subtitle_bytes(data: bytes, fmt: str | None) -> list[Cue]:
    """Parse subtitle/transcript bytes into cues (``fmt``: vtt/srt/json/txt or None).

    Raises :class:`ValidationError` for undecodable, binary or malformed data.
    """
    text = _normalise_text(_decode(data))
    bad = _bad_controls(text)
    if bad:
        raise ValidationError(
            f"Subtitle data contains control characters ({', '.join(bad[:5])}); "
            "it does not look like text.")
    kind = (fmt or "").lower().lstrip(".") or detect_format(text)
    kind = {"webvtt": "vtt", "text": "txt", "plain": "txt"}.get(kind, kind)
    parsers = {"vtt": _parse_vtt, "srt": _parse_srt, "json": _parse_json, "txt": _parse_txt}
    if kind not in parsers:
        raise ValidationError(f"Unsupported subtitle format {fmt!r}.",
                              hint="Supported: vtt, srt, json, txt.")
    return parsers[kind](text)


def loses_information(raw: bytes | None, fmt: str | None) -> bool:
    """True when converting ``raw`` to our VTT/SRT/TXT would drop information
    (styling, positioning, regions, notes, inline tags other than voices, or a
    format we cannot represent losslessly such as a JSON transcript)."""
    if not raw:
        return False
    kind = (fmt or "").lower()
    if kind == "txt":
        return False
    try:
        text = _normalise_text(_decode(raw))
    except ValidationError:
        return True
    if kind == "vtt":
        if re.search(r"(?m)^(STYLE|REGION|NOTE)\b", text):
            return True
        if re.search(rf"(?m)^\s*{_TS}\s*-->\s*{_TS}[ \t]+\S", text):
            return True  # cue settings (position, align, line, ...)
        tags = re.findall(r"<(/?)([A-Za-z0-9.:]+)", text)
        return any(not name.lower().startswith("v") for _, name in tags)
    if kind == "srt":
        return bool(re.search(r"<[A-Za-z/]|\{\\", text))
    return True


# --------------------------------------------------------------------------------------
# Validation / normalisation
# --------------------------------------------------------------------------------------


def validate_cues(cues: list[Cue], duration: float | None, *,
                  allow_overlap: bool = True, tolerance: float = 0.5) -> list[str]:
    """Problems with ``cues``: mixed timing, end <= start, negative/out-of-bounds times,
    decreasing starts, overlaps (when not allowed), empty text and control characters.
    ``duration`` (seconds) bounds the cues with ``tolerance``."""
    problems: list[str] = []
    timed = [c.start is not None and c.end is not None for c in cues]
    if any(timed) and not all(timed):
        problems.append("some cues have timestamps and others do not")
    prev: Cue | None = None
    for n, cue in enumerate(cues, start=1):
        if not cue.text.strip():
            problems.append(f"cue {n}: empty text")
        bad = _bad_controls(cue.text)
        if bad:
            problems.append(f"cue {n}: control characters {', '.join(bad)}")
        if cue.start is None or cue.end is None:
            continue
        if cue.start < 0:
            problems.append(f"cue {n}: negative start")
        if cue.end <= cue.start:
            problems.append(f"cue {n}: end ({cue.end:.3f}) must be after start "
                            f"({cue.start:.3f})")
        if duration is not None and cue.end > duration + tolerance:
            problems.append(f"cue {n}: ends at {cue.end:.3f}s, after the media end "
                            f"({duration:.3f}s)")
        if prev is not None and prev.start is not None and prev.end is not None:
            if cue.start < prev.start:
                problems.append(f"cue {n}: starts before the previous cue (not ordered)")
            elif not allow_overlap and cue.start < prev.end - 1e-6:
                problems.append(f"cue {n}: overlaps the previous cue")
        prev = cue
    return problems


def normalize_cues(cues: Iterable[Cue], duration: float | None = None, *,
                   trim_overlaps: bool = False) -> list[Cue]:
    """Sort timed cues, drop empty ones, clamp to ``[0, duration]`` and optionally trim
    overlaps. Never creates timestamps; untimed cues are returned in order."""
    items = [Cue(text=c.text.strip(), start=c.start, end=c.end, speaker=c.speaker)
             for c in cues if c.text.strip()]
    if not items or any(c.start is None or c.end is None for c in items):
        return items
    items.sort(key=lambda c: (c.start, c.end))
    out: list[Cue] = []
    for i, cue in enumerate(items):
        start = max(0.0, cue.start or 0.0)
        end = cue.end or 0.0
        if duration is not None:
            end = min(end, duration)
        if trim_overlaps and i + 1 < len(items):
            end = min(end, items[i + 1].start or end)
        if end > start:
            out.append(Cue(text=cue.text, start=start, end=end, speaker=cue.speaker))
    return out


# --------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------


def _require_timed(t: Transcript, fmt: str) -> None:
    if not t.cues:
        raise TranscriptUnavailableError(f"Cannot write {fmt}: the transcript has no cues.")
    if not t.timed:
        raise TranscriptUnavailableError(
            f"Cannot write {fmt}: this transcript has no timestamps (untimed text).",
            hint="Untimed transcripts are saved as .txt only; timestamps are never invented.")


def _one_paragraph(text: str) -> str:
    return "\n".join(line for line in text.split("\n") if line.strip())


def _vtt_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_vtt(t: Transcript, *, voice_tags: bool = True) -> str:
    """WebVTT. Speakers as ``<v Name>`` (or ``Name: text`` with ``voice_tags=False``,
    used for MP4 ``mov_text`` which cannot carry voice spans)."""
    _require_timed(t, "WebVTT")
    out = ["WEBVTT", ""]
    if t.generated:
        note = "NOTE generated locally by speech recognition"
        if t.generator:
            note += f" ({t.generator.replace('-->', '->')})"
        out += [note + "; not an official transcript.", ""]
    for cue in t.cues:
        body = _vtt_escape(_one_paragraph(cue.text))
        if cue.speaker:
            speaker = _vtt_escape(cue.speaker.replace("\n", " "))
            body = f"<v {speaker}>{body}" if voice_tags else f"{speaker}: {body}"
        out += [f"{format_clock(cue.start or 0.0, millis=True)} --> "
                f"{format_clock(cue.end or 0.0, millis=True)}", body, ""]
    return "\n".join(out)


def to_srt(t: Transcript) -> str:
    _require_timed(t, "SRT")
    out: list[str] = []
    for n, cue in enumerate(t.cues, start=1):
        body = _one_paragraph(cue.text)
        if cue.speaker:
            body = f"{cue.speaker}: {body}"
        out += [str(n), f"{format_clock(cue.start or 0.0, millis=True, sep=',')} --> "
                f"{format_clock(cue.end or 0.0, millis=True, sep=',')}", body, ""]
    return "\n".join(out)


def to_txt(t: Transcript) -> str:
    """Plain text, one cue per line, ``Speaker: text`` when the speaker is known."""
    lines = []
    for cue in t.cues:
        text = " ".join(cue.text.split("\n")).strip()
        lines.append(f"{cue.speaker}: {text}" if cue.speaker else text)
    return "\n".join(lines) + ("\n" if lines else "")


def to_json(t: Transcript) -> str:
    """3ch0 transcript JSON (version 1) with provenance; parseable again."""
    return json.dumps({
        "version": 1,
        "language": t.language,
        "provenance": t.provenance.value,
        "label": t.label,
        "generator": t.generator,
        "source_detail": t.source_detail,
        "timed": t.timed,
        "cues": [{"start": c.start, "end": c.end, "speaker": c.speaker, "text": c.text}
                 for c in t.cues],
    }, ensure_ascii=False, indent=2) + "\n"
