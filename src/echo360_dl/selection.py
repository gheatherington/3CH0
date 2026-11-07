"""Parsing of user selections and deterministic rendition choice.

Quality semantics (documented in README):
* ``best``  - highest resolution (then bandwidth) available for the source.
* ``worst`` - lowest resolution (then bandwidth).
* ``720p`` / ``720`` - best rendition whose height is <= 720. If every rendition
  is taller, :class:`QualityUnavailableError` is raised listing what exists.
  Renditions with unknown height are only chosen when no rendition has a height.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from echo360_dl.errors import (
    ForbiddenError,
    QualityUnavailableError,
    UnavailableError,
    UsageError,
)
from echo360_dl.models import (
    AccessState,
    AudioRendition,
    AudioTrack,
    ChapterMode,
    ChapterModeKind,
    QualitySpec,
    Recording,
    SubtitleMode,
    SubtitleModeKind,
    VideoRendition,
    VideoSource,
)
from echo360_dl.timecode import parse_time

_LANG = re.compile(r"^[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})*$")


@dataclass(frozen=True)
class StreamSelection:
    source_indexes: tuple[int, ...]
    audio_only: bool = False
    all_sources: bool = False


def parse_streams(text: str) -> StreamSelection:
    """``all`` | ``audio`` | comma list of source numbers (``1``, ``1,2``)."""
    value = text.strip().lower()
    if value == "all":
        return StreamSelection((), all_sources=True)
    if value in {"audio", "audio-only"}:
        return StreamSelection((), audio_only=True)
    indexes: list[int] = []
    for part in value.split(","):
        part = part.strip().removeprefix("source").removeprefix("s")
        if not part.isdigit() or int(part) < 1:
            raise UsageError(f"Invalid --streams value {text!r}.",
                             hint="Use 'all', 'audio', or source numbers such as '1' or '1,2'.")
        if int(part) not in indexes:
            indexes.append(int(part))
    return StreamSelection(tuple(indexes))


#: Access states a source may have and still be selected by ``--streams all``.
USABLE_ACCESS = frozenset({AccessState.AVAILABLE, AccessState.UNKNOWN})


def skipped_by_all(recording: Recording) -> list[VideoSource]:
    """Sources that ``--streams all`` leaves out (e.g. forbidden); callers should warn."""
    return [s for s in recording.sources if s.access not in USABLE_ACCESS]


def resolve_streams(selection: StreamSelection, recording: Recording) -> tuple[int, ...]:
    """Map a selection onto the recording's sources; unknown numbers are an error.

    ``all`` means every source whose access is available/unknown. Sources that returned
    403 etc. are skipped (see :func:`skipped_by_all`) but can still be requested by number.
    """
    if selection.audio_only:
        return ()
    known = [s.index for s in recording.sources]
    if selection.all_sources:
        if not known:
            raise UnavailableError("No video sources were discovered for this lesson.")
        usable = tuple(s.index for s in recording.sources if s.access in USABLE_ACCESS)
        if not usable:
            states = ", ".join(f"{s.display_name}: {s.access.value}" for s in recording.sources)
            raise UnavailableError(f"No accessible video source ({states}).",
                                   hint="A 403 does not prove a source is absent; try "
                                        "'echo360-dl auth login' and re-run, or pick a "
                                        "source explicitly with --streams N.")
        return usable
    missing = [i for i in selection.source_indexes if i not in known]
    refused = [i for i in missing if i in recording.unreachable_sources]
    if refused:
        states = ", ".join(f"Source {i}: {recording.unreachable_sources[i].value}"
                           for i in refused)
        raise ForbiddenError(
            f"Could not access {states} during discovery. Echo360 answers 403 both for "
            "sources this login cannot access and for sources that do not exist.",
            hint="If the source plays in the browser, run 'echo360-dl auth login' and retry. "
                 f"Accessible sources: {', '.join(map(str, known)) or 'none'}.")
    if missing:
        raise UnavailableError(
            f"Source {', '.join(map(str, missing))} not found in this recording.",
            hint=f"Available sources: {', '.join(map(str, known)) or 'none'}.")
    return selection.source_indexes


def parse_quality(text: str) -> QualitySpec:
    value = text.strip().lower()
    if value in {"best", "worst"}:
        return QualitySpec(kind=value)  # type: ignore[arg-type]
    match = re.fullmatch(r"(\d{3,4})p?", value)
    if not match or int(match[1]) <= 0:
        raise UsageError(f"Invalid --quality value {text!r}.",
                         hint="Use 'best', 'worst', or a height such as '720p'.")
    return QualitySpec(kind="max_height", max_height=int(match[1]))


def _rank(r: VideoRendition) -> tuple[int, int, int]:
    return (r.height or 0, r.width or 0, r.bandwidth or r.average_bandwidth or 0)


def select_rendition(source: VideoSource, quality: QualitySpec) -> VideoRendition:
    if not source.renditions:
        raise UnavailableError(f"{source.display_name} has no downloadable renditions.")
    ordered = sorted(source.renditions, key=_rank)
    if quality.kind == "best":
        return ordered[-1]
    if quality.kind == "worst":
        return ordered[0]
    cap = quality.max_height or 0
    with_height = [r for r in ordered if r.height]
    if not with_height:
        return ordered[-1]
    fitting = [r for r in with_height if r.height <= cap]  # type: ignore[operator]
    if not fitting:
        available = ", ".join(f"{h}p" for h in sorted({r.height for r in with_height}))
        raise QualityUnavailableError(
            f"{source.display_name}: no rendition at or below {cap}p (available: {available}).",
            hint="Choose a higher --quality or use 'worst'.")
    return fitting[-1]


def select_audio(track: AudioTrack, paired: VideoRendition | None = None,
                 quality: QualitySpec | None = None) -> AudioRendition:
    """Audio rendition paired with ``paired`` (same HLS group) or the best-guess best.

    Echo360 publishes no bandwidth for audio renditions; the group paired with the
    highest video variant is treated as the higher-quality audio (observed: q1).
    """
    if not track.renditions:
        raise UnavailableError("The recording has no downloadable audio rendition.")
    if paired and paired.audio_group:
        for r in track.renditions:
            if r.group_id == paired.audio_group:
                return r
    ordered = sorted(track.renditions, key=lambda r: (r.bandwidth or 0, r.id))
    if quality and quality.kind == "worst":
        return ordered[0]
    return ordered[-1]


def parse_subtitles(text: str) -> SubtitleMode:
    value = text.strip()
    if value.lower() in {"auto", ""}:
        return SubtitleMode(SubtitleModeKind.AUTO)
    if value.lower() in {"none", "off", "no"}:
        return SubtitleMode(SubtitleModeKind.NONE)
    langs = tuple(dict.fromkeys(p.strip() for p in value.split(",") if p.strip()))
    bad = [lang for lang in langs if not _LANG.match(lang)]
    if bad or not langs:
        raise UsageError(f"Invalid --subtitles value {text!r}.",
                         hint="Use 'auto', 'none', or language codes such as 'en' or 'en,fr'.")
    return SubtitleMode(SubtitleModeKind.LANGUAGES, langs)


def parse_chapters(mode: str, *, file: str | None = None,
                   interval: str | None = None) -> ChapterMode:
    """``--chapters auto|none|interval`` plus ``--chapters-file`` / ``--chapter-interval``.

    A chapters file always selects FILE mode (it takes precedence over native
    chapters). ``interval`` requires an explicit ``--chapters interval``.
    """
    from pathlib import Path

    value = mode.strip().lower()
    if file:
        if value not in {"auto", "file"}:
            raise UsageError("--chapters-file cannot be combined with --chapters "
                             f"{value!r}.")
        return ChapterMode(ChapterModeKind.FILE, file=Path(file).expanduser())
    if value == "auto":
        return ChapterMode(ChapterModeKind.AUTO)
    if value in {"none", "off"}:
        return ChapterMode(ChapterModeKind.NONE)
    if value == "interval":
        try:
            seconds = parse_time(interval or "10m")
        except ValueError as exc:
            raise UsageError(f"Invalid --chapter-interval: {exc}") from None
        if seconds < 30:
            raise UsageError("--chapter-interval must be at least 30 seconds.")
        return ChapterMode(ChapterModeKind.INTERVAL, interval=seconds)
    if value == "file":
        raise UsageError("--chapters file requires --chapters-file PATH.")
    raise UsageError(f"Invalid --chapters value {mode!r}.",
                     hint="Use 'auto', 'none', or 'interval' (or pass --chapters-file).")


def normalize_language(code: str | None) -> str | None:
    """Lower-case primary subtag comparison key: ``en-CA`` -> ``en``."""
    if not code:
        return None
    return re.split(r"[-_]", code.strip())[0].lower() or None


def language_matches(wanted: str, available: str | None) -> bool:
    if not available:
        return False
    w, a = wanted.lower().replace("_", "-"), available.lower().replace("_", "-")
    return w == a or (normalize_language(w) == normalize_language(a) and "-" not in w)


_ISO639_2 = {
    "en": "eng", "fr": "fre", "es": "spa", "de": "ger", "it": "ita", "pt": "por",
    "zh": "chi", "ja": "jpn", "ko": "kor", "ar": "ara", "ru": "rus", "hi": "hin",
    "nl": "dut", "sv": "swe", "pl": "pol", "tr": "tur", "vi": "vie", "fa": "per",
    "uk": "ukr", "he": "heb", "el": "gre", "cs": "cze", "da": "dan", "fi": "fin",
    "no": "nor", "nb": "nob", "ro": "rum", "hu": "hun", "id": "ind", "th": "tha",
    "pa": "pan", "ta": "tam", "ur": "urd", "bn": "ben", "tl": "tgl", "ms": "may",
}


def iso639_2(code: str | None) -> str:
    """Container language tag (ISO 639-2/B). Unknown -> ``"und"``."""
    primary = normalize_language(code)
    if not primary:
        return "und"
    if len(primary) == 3:
        return primary
    return _ISO639_2.get(primary, "und")


def is_known_language(code: str | None) -> bool:
    """True for language codes in the built-in ISO 639-1/639-2 table."""
    primary = normalize_language(code)
    return bool(primary) and (primary in _ISO639_2 or primary in _ISO639_2.values())


def parse_layout_order(text: str | None) -> tuple[int, ...]:
    """``--layout-order 2,1`` -> ``(2, 1)`` (first = left/top)."""
    if not text:
        return ()
    selection = parse_streams(text)
    if selection.all_sources or selection.audio_only or len(selection.source_indexes) != 2:
        raise UsageError(f"Invalid --layout-order {text!r}.",
                         hint="Give exactly two source numbers, e.g. --layout-order 2,1 "
                              "(the first is placed left/top).")
    return selection.source_indexes


def composite_pair(selected: tuple[int, ...], order: tuple[int, ...] = ()) -> tuple[int, int]:
    """The two sources (in display order) that make up a side-by-side/top-bottom video."""
    if order:
        missing = [i for i in order if i not in selected]
        if missing:
            raise UsageError(
                f"--layout-order names source {', '.join(map(str, missing))}, which is not "
                "selected.", hint="Include it in --streams, or change --layout-order.")
        return order[0], order[1]
    if len(selected) == 2:
        return selected[0], selected[1]
    raise UsageError(
        f"A combined layout needs exactly two video sources; {len(selected)} selected.",
        hint="Pick two with --streams 1,2 (or --layout-order 1,2), or use --layout separate.")
