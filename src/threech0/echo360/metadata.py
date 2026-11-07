"""Tolerant extractors for Echo360 lesson metadata, captions, transcripts and chapters.

EVIDENCE STATUS - read before changing anything here.

OBSERVED on echo360.ca (live sanitised trace, 2026-11-07, one lesson):

* O-TRANSCRIPT-API ``GET /api/ui/echoplayer/lessons/{lesson}/medias/{media}/transcript`` ->
  ``{"status", "message", "data": {"id", "created", "status", "isAutomated", "institutionId",
  "mediaId", "contentJSON": {"cues": [{"startMs", "endMs", "speaker", "content", "notes",
  "confidence"}], "notes"}}}``. ``isAutomated`` marks Echo360's own speech recognition.
* O-CAPTION ``https://captions.echo360.ca/<institution>/captions-<media>-<YYYYMMDDhhmmss>.vtt``
  fetched by the player as a ``texttrack`` with ``content-type: application/octet-stream``.
* O-DOWNLOAD-INFO ``GET .../medias/{media}/download-info`` ->
  ``data.audioFiles[]`` and ``data.transcriptFiles[{fileName, label}]`` (not used yet).
* O-PLAYER-APP: no JSON API carries title, course, duration or the source list. They are
  in the classroom HTML: ``Echo["echoPlayerV2FullApp"]("<JSON string>")`` (see
  :func:`apply_player_app` for the observed keys). This object takes precedence over every
  hypothesis below. It has no camera/screen labels and no caption language.

Everything else is still a **hypothesis**, chosen from public third-party sources and common
naming, and is never presented as verified:

* H-TITLE   ``lesson.lesson.name`` / ``lesson.lesson.id`` and ``lesson.startTimeUTC``:
  github.com/soraxas/echo360 (``echo360/videos.py``, ``EchoCloudVideo``), observed by that
  project on other Echo360 regions.
* H-MEDIA   ``"mediaId": "<uuid>"`` in player data: same project (``_MEDIA_ID_RE``).
* H-CAPTION ``https://captions.<host>/....vtt`` URLs in page data: same project
  (``_CAPTIONS_VTT_RE``).
* H-TRANSCRIPT-API ``/api/ui/echoplayer/lessons/{lesson}/medias/{media}/transcript``: same
  project. H-TRANSCRIPT-JSON ``data.contentJSON.cues[]`` with ``startMs``/``endMs``/
  ``content``: dot15.pages.dev/TIL/dl-echo360 (a personal write-up).
* H-COURSE, H-DURATION, H-LABELS, H-CHAPTERS: generic candidate names; no public source.

Whenever an extractor relying on a hypothesis contributes a value, a
``DiscoveryNote(observed=False)`` names the field so the user can check it with
``--save-trace``.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

from threech0.echo360.urls import is_echo360_url
from threech0.errors import ValidationError
from threech0.models import (
    CaptionTrack,
    Chapter,
    ChapterProvenance,
    ChapterSet,
    Cue,
    DiscoveryMethod,
    DiscoveryNote,
    TranscriptProvenance,
)
from threech0.redact import public_url
from threech0.timecode import parse_time

_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_SOURCE_REF = re.compile(r"(?:^|/)s(\d{1,2})(?:_av|q\d+)\.m3u8(?:$|\?)", re.IGNORECASE)
_SUB_EXT = re.compile(r"\.(vtt|srt)$", re.IGNORECASE)

# H-TITLE
_ID_KEYS = ("id", "lessonId", "lesson_id", "lessonID")
_TITLE_KEYS = ("name", "title", "displayName", "lessonName", "lessonTitle")
# H-COURSE
_COURSE_KEYS = ("courseName", "courseTitle", "course_name", "sectionName")
_COURSE_OBJ_KEYS = ("course",)
# H-TITLE (start time from soraxas: startTimeUTC); others generic
_START_KEYS = ("startTimeUTC", "startTime", "startDate", "recordedAt", "recordingStart",
               "captureStartTime", "start_time")
# H-DURATION
_DURATION_MS_KEYS = ("durationMs", "durationMillis", "durationInMs", "duration_ms")
_DURATION_S_KEYS = ("durationSeconds", "durationInSeconds", "duration_seconds", "duration")
# H-MEDIA
_MEDIA_KEYS = ("mediaId", "media_id", "mediaID")
# H-LABELS
_LABEL_KEYS = ("label", "sourceLabel", "sourceName", "displayName", "title", "name")
_INDEX_KEYS = ("sourceIndex", "sourceNumber", "source_index")
_GENERIC_LABEL = re.compile(
    r"(?i)^(?:default|video|audio|stream|source|main|primary|secondary|media|track|"
    r"(?:source|stream|video|track|feed|s)\s*[-_#]?\s*\d+|.*\.(?:m3u8|mp4|m4s|ts))$")
# H-CAPTION
_URL_KEYS = ("url", "uri", "href", "src", "s3Url", "downloadUrl", "link")
_LANG_KEYS = ("language", "lang", "languageCode", "locale", "srclang")
_CAPTION_URL_RE = re.compile(r"https://captions\.[^\"'\s\\<>]+?\.vtt", re.IGNORECASE)
_MEDIA_ID_RE = re.compile(r'\\?"mediaId\\?"\s*:\s*\\?"([0-9a-fA-F-]{36})')
# H-CHAPTERS
_CHAPTER_LIST_KEY = re.compile(
    r"(?i)^(?:chapters?|chapterMarkers|chapterList|navigation|tableOfContents|toc|outline)$")
_CH_TITLE_KEYS = ("title", "name", "label", "text")
_CH_MS_KEYS = ("startMs", "startTimeMs", "offsetMs", "timeMs", "positionMs")
_CH_S_KEYS = ("start", "startTime", "time", "offset", "position", "seconds", "timestamp")
# H-TRANSCRIPT-JSON
_CUE_TEXT_KEYS = ("content", "text", "caption", "body", "transcript")
_CUE_START_MS = ("startMs", "startTimeMs", "beginMs", "start_ms")
_CUE_END_MS = ("endMs", "endTimeMs", "stopMs", "end_ms")
_CUE_START_S = ("start", "startTime", "begin", "start_time")
_CUE_END_S = ("end", "endTime", "stop", "end_time")
_SPEAKER_KEYS = ("speaker", "speakerName", "speaker_name")


def _hyp(message: str) -> DiscoveryNote:
    return DiscoveryNote("info", message + " (field name is a hypothesis, not verified for "
                                           "echo360.ca)", observed=False)


# -- generic walking -----------------------------------------------------------------------


def iter_nodes(obj: Any, path: tuple[Any, ...] = (), depth: int = 0
               ) -> Iterator[tuple[tuple[Any, ...], Any]]:
    """Yield ``(path, node)`` for every dict/list node (depth bounded)."""
    if depth > 40:
        return
    if isinstance(obj, dict):
        yield path, obj
        for k, v in obj.items():
            yield from iter_nodes(v, (*path, k), depth + 1)
    elif isinstance(obj, list):
        yield path, obj
        for i, v in enumerate(obj):
            yield from iter_nodes(v, (*path, i), depth + 1)


def iter_strings(obj: Any, depth: int = 0) -> Iterator[tuple[str | None, str]]:
    """Yield ``(key, value)`` for every string value (key None inside lists)."""
    if depth > 40:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                yield str(k), v
            else:
                yield from iter_strings(v, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, str):
                yield None, v
            else:
                yield from iter_strings(v, depth + 1)


def _get(obj: Any, path: tuple[Any, ...]) -> Any:
    for p in path:
        obj = obj[p]
    return obj


def mentions(obj: Any, needle: str) -> bool:
    return bool(needle) and any(needle in v for _k, v in iter_strings(obj))


def lesson_scopes(data: Any, url: str, lesson_id: str,
                  media_ids: tuple[str, ...] = ()) -> list[Any]:
    """Sub-trees of ``data`` that describe this lesson.

    If the response URL names the lesson, the whole document is in scope. Otherwise each
    place where the lesson id (or a known media id) appears as a value is widened to its
    enclosing list element (e.g. one entry of a course syllabus); with no enclosing list the
    whole document is used. This keeps other lessons' entries (prefetch, syllabus lists)
    out of scope.
    """
    if lesson_id and lesson_id in urlsplit(url).path:
        return [data]
    needles = [n for n in (lesson_id, *media_ids) if n]
    scopes: list[Any] = []
    seen: set[int] = set()
    for path, node in iter_nodes(data):
        if not isinstance(node, dict):
            continue
        if not any(isinstance(v, str) and v in needles for v in node.values()):
            continue
        cut = max((i for i, p in enumerate(path) if isinstance(p, int)), default=None)
        scope = data if cut is None else _get(data, path[:cut + 1])
        if id(scope) not in seen:
            seen.add(id(scope))
            scopes.append(scope)
    return scopes


# -- results -------------------------------------------------------------------------------


@dataclass
class LessonMetadata:
    title: str | None = None
    course: str | None = None
    recorded_at: str | None = None
    duration: float | None = None
    media_ids: list[str] = field(default_factory=list)
    playlist_urls: list[str] = field(default_factory=list)
    source_labels: dict[int, str] = field(default_factory=dict)
    captions: list[CaptionTrack] = field(default_factory=list)
    chapters: ChapterSet | None = None
    notes: list[DiscoveryNote] = field(default_factory=list)

    @property
    def listed_source_indexes(self) -> set[int]:
        """Source numbers named by playlist URLs in metadata (``s<N>_av``)."""
        out: set[int] = set()
        for url in self.playlist_urls:
            m = _SOURCE_REF.search(urlsplit(url).path)
            if m:
                out.add(int(m[1]))
        return out


def find_media_ids(scopes: list[Any], html: str | None = None) -> tuple[list[str], str | None]:
    """H-MEDIA: UUID values under ``mediaId``-like keys (and the page-source regex)."""
    found: list[str] = []
    key_used = None
    for scope in scopes:
        for path, node in iter_nodes(scope):
            if not isinstance(node, dict):
                continue
            for key in _MEDIA_KEYS:
                v = node.get(key)
                if isinstance(v, str) and _UUID.match(v) and v.lower() not in found:
                    found.append(v.lower())
                    key_used = key_used or key
            if path and path[-1] == "media" and isinstance(node.get("id"), str) and _UUID.match(
                    node["id"]) and node["id"].lower() not in found:
                found.append(node["id"].lower())
                key_used = key_used or "media.id"
    if html:
        for m in _MEDIA_ID_RE.finditer(html):
            if _UUID.match(m[1]) and m[1].lower() not in found:
                found.append(m[1].lower())
                key_used = key_used or "mediaId (page source)"
    return found, key_used


def _first_str(node: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, str] | None:
    for key in keys:
        v = node.get(key)
        if isinstance(v, str) and v.strip():
            return key, v.strip()
    return None


def _iso(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        seconds = value / 1000 if value > 10_000_000_000 else value
        if not 946_684_800 <= seconds <= 4_102_444_800:  # 2000..2100
            return None
        return dt.datetime.fromtimestamp(seconds, dt.UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.isoformat()
        return parsed.astimezone(dt.UTC).isoformat().replace("+00:00", "Z")
    return None


_ISO_DURATION = re.compile(
    r"^P(?:(\d+(?:\.\d+)?)D)?T?(?:(\d+(?:\.\d+)?)H)?(?:(\d+(?:\.\d+)?)M)?(?:(\d+(?:\.\d+)?)S)?$")


def _duration(key: str, value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if key in _DURATION_MS_KEYS and isinstance(value, int | float):
        seconds = value / 1000
    elif isinstance(value, int | float):
        seconds = float(value)
    elif isinstance(value, str):
        m = _ISO_DURATION.match(value.strip())
        if m and any(m.groups()):
            d, h, mi, s = (float(g) if g else 0.0 for g in m.groups())
            seconds = d * 86400 + h * 3600 + mi * 60 + s
        else:
            try:
                seconds = parse_time(value)
            except ValueError:
                return None
    else:
        return None
    return seconds if 0 < seconds <= 86_400 else None


def extract_metadata(docs: list[tuple[str, Any]], lesson_id: str, *,
                     html: str | None = None,
                     known_media_ids: tuple[str, ...] = ()) -> LessonMetadata:
    """Walk observed JSON documents ``(url, parsed)`` for lesson facts (all hypotheses)."""
    meta = LessonMetadata()
    scopes: list[tuple[str, Any]] = []
    if html:
        from threech0.echo360.embedded import extract_embedded_json

        # Lesson data embedded in the classroom page (see embedded.py); filtered to the
        # lesson by lesson_scopes() like any other document.
        docs = [*docs, *((f"page:{e.target or e.kind}", e.data)
                         for e in extract_embedded_json(html))]
    for url, data in docs:
        for scope in lesson_scopes(data, url, lesson_id, known_media_ids):
            scopes.append((url, scope))
    scope_objs = [s for _u, s in scopes]

    media_ids, media_key = find_media_ids(scope_objs, html)
    meta.media_ids = media_ids
    if media_ids and media_key:
        meta.notes.append(_hyp(f"Media id read from '{media_key}'"))

    # H-TITLE: the dict whose id-like key equals the lesson id
    title_found = start_found = None
    for scope in scope_objs:
        for _path, node in iter_nodes(scope):
            if not isinstance(node, dict):
                continue
            if not any(node.get(k) == lesson_id for k in _ID_KEYS):
                continue
            if title_found is None:
                title_found = _first_str(node, _TITLE_KEYS)
            if start_found is None:
                for key in _START_KEYS:
                    if key in node and _iso(node[key]):
                        start_found = (key, _iso(node[key]))
                        break
    if title_found is None:
        for scope in scope_objs:
            for _path, node in iter_nodes(scope):
                if isinstance(node, dict):
                    hit = _first_str(node, ("lessonName", "lessonTitle"))
                    if hit:
                        title_found = hit
                        break
            if title_found:
                break
    if title_found:
        meta.title = title_found[1][:300]
        meta.notes.append(_hyp(f"Title read from '{title_found[0]}' of the lesson object"))

    # Start time: lesson object first, otherwise the first start-like key in scope.
    if start_found is None:
        for scope in scope_objs:
            for _path, node in iter_nodes(scope):
                if not isinstance(node, dict):
                    continue
                for key in _START_KEYS:
                    if key in node and _iso(node[key]):
                        start_found = (key, _iso(node[key]))
                        break
                if start_found:
                    break
            if start_found:
                break
    if start_found:
        meta.recorded_at = start_found[1]
        meta.notes.append(_hyp(f"Recording start read from '{start_found[0]}'"))

    # H-COURSE
    course_found: tuple[str, str] | None = None
    for scope in scope_objs:
        for path, node in iter_nodes(scope):
            if not isinstance(node, dict):
                continue
            hit = _first_str(node, _COURSE_KEYS)
            if hit is None and path and path[-1] in _COURSE_OBJ_KEYS:
                inner = _first_str(node, ("name", "title", "code"))
                hit = (f"{path[-1]}.{inner[0]}", inner[1]) if inner else None
            if hit:
                course_found = hit
                break
        if course_found:
            break
    if course_found:
        meta.course = course_found[1][:300]
        meta.notes.append(_hyp(f"Course read from '{course_found[0]}'"))

    # H-DURATION
    for scope in scope_objs:
        for _path, node in iter_nodes(scope):
            if not isinstance(node, dict):
                continue
            for key in (*_DURATION_MS_KEYS, *_DURATION_S_KEYS):
                if key in node:
                    value = _duration(key, node[key])
                    if value is not None:
                        meta.duration = value
                        meta.notes.append(_hyp(f"Duration read from '{key}'"))
                        break
            if meta.duration is not None:
                break
        if meta.duration is not None:
            break

    # Playlist URLs listed in metadata (the URL text itself is the evidence of the source).
    for scope in scope_objs:
        for _key, value in iter_strings(scope):
            v = value.strip()
            if v.lower().split("?", 1)[0].endswith(".m3u8") and is_echo360_url(v) and \
                    v not in meta.playlist_urls:
                meta.playlist_urls.append(v)

    labels, label_notes = extract_source_labels(scope_objs)
    meta.source_labels = labels
    meta.notes.extend(label_notes)
    meta.captions, cap_notes = extract_captions(scope_objs, html)
    meta.notes.extend(cap_notes)
    meta.chapters, ch_notes = extract_chapters(scope_objs)
    meta.notes.extend(ch_notes)
    for _url, data in docs:
        app = find_player_app(data)
        if app is None:
            continue
        ids = {(app.get("lesson") or {}).get("id"), (app.get("context") or {}).get("lessonId")}
        ids.discard(None)
        if ids and lesson_id not in ids:
            continue  # player data for a different lesson (never mix lessons)
        apply_player_app(meta, app)
        break
    return meta


# --------------------------------------------------------------------------------------
# OBSERVED: the classroom page's player bootstrap object (2026-11-07, echo360.ca)
# --------------------------------------------------------------------------------------
#
# The classroom HTML calls ``Echo["echoPlayerV2FullApp"]("<JSON string>")``. Observed keys:
#   lesson.{name, displayName, lessonHasName, timing.{start,end} (local, no offset),
#           timeZone.{id, name, standardOffset}}
#   video.{mediaId, duration ("PT3240.045S"),
#          playableMedias[{sourceIndex, trackType ["Audio"|"Video"...], quality, uri,
#                          isHls, isLive}]}   (sourceIndex 0 = shared audio "s0_a")
#   sectionInfo.{course.{courseName, courseIdentifier}, section.sectionNumber, term.name}
#   title, captions (URL string), chapters (null on the observed lesson), locale (UI only)
# ``startTimeMillis`` is the *playback resume position*, never a trim instruction.
# No camera/screen labels and no caption language were present.

_ISO_DURATION = re.compile(
    r"^P(?:(\d+(?:\.\d+)?)D)?(?:T(?:(\d+(?:\.\d+)?)H)?(?:(\d+(?:\.\d+)?)M)?"
    r"(?:(\d+(?:\.\d+)?)S)?)?$")


def parse_iso_duration(text: str) -> float | None:
    m = _ISO_DURATION.match(text.strip()) if isinstance(text, str) else None
    if not m or not any(m.groups()):
        return None
    d, h, mi, sec = (float(g) if g else 0.0 for g in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + sec


def find_player_app(data: Any) -> dict[str, Any] | None:
    """The observed player bootstrap object, recognised by its structure."""
    if isinstance(data, dict):
        lesson, video = data.get("lesson"), data.get("video")
        if isinstance(lesson, dict) and isinstance(video, dict) and \
                isinstance(video.get("playableMedias"), list):
            return data
    return None


def _local_to_iso(local: Any, zone: Any) -> str | None:
    if not isinstance(local, str):
        return None
    try:
        when = dt.datetime.fromisoformat(local)
    except ValueError:
        return None
    if when.tzinfo is None and isinstance(zone, str) and zone:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        # An unknown zone keeps the local time without an offset.
        with contextlib.suppress(ZoneInfoNotFoundError, ValueError):
            when = when.replace(tzinfo=ZoneInfo(zone))
    return when.isoformat()


def apply_player_app(meta: LessonMetadata, app: dict[str, Any]) -> None:
    """Overwrite guessed values with the observed player object's values."""
    lesson = app.get("lesson") or {}
    video = app.get("video") or {}
    section = app.get("sectionInfo") or {}
    replaced: set[str] = set()

    name = lesson.get("displayName") or lesson.get("name")
    if lesson.get("lessonHasName") is False:
        name = None
    title = name or app.get("title")
    if isinstance(title, str) and title.strip():
        meta.title = title.strip()[:300]
        replaced.add("Title")
    timing = lesson.get("timing") if isinstance(lesson.get("timing"), dict) else {}
    zone = (lesson.get("timeZone") or {}).get("id") if isinstance(
        lesson.get("timeZone"), dict) else None
    start = _local_to_iso(timing.get("start"), zone)
    if start:
        meta.recorded_at = start
        replaced.add("Recording start")
    course = (section.get("course") or {}).get("courseName") if isinstance(
        section.get("course"), dict) else None
    if isinstance(course, str) and course.strip():
        meta.course = course.strip()[:300]
        replaced.add("Course")
    duration = parse_iso_duration(video.get("duration"))
    if duration is not None:
        meta.duration = duration
        replaced.add("Duration")
    media_id = video.get("mediaId")
    if isinstance(media_id, str) and media_id:
        meta.media_ids = [media_id, *[m for m in meta.media_ids if m != media_id]]
        replaced.add("Media id")
    urls: list[str] = []
    medias = [m for m in video.get("playableMedias") or [] if isinstance(m, dict)]
    # Audio+video masters first (they pair the shared audio), then video-only fallbacks.
    for want_audio in (True, False):
        for media in medias:
            types = {str(t).lower() for t in media.get("trackType") or []}
            uri = media.get("uri")
            if not isinstance(uri, str) or "video" not in types:
                continue
            if ("audio" in types) is want_audio and uri not in urls:
                urls.append(uri)
    if urls:
        meta.playlist_urls = [*urls, *[u for u in meta.playlist_urls if u not in urls]]
    meta.notes = [n for n in meta.notes
                  if not any(n.message.startswith(f"{field_name} read from")
                             for field_name in replaced)]
    meta.notes.append(DiscoveryNote(
        "info", "Lesson details read from the classroom page's player data "
                "(Echo[\"echoPlayerV2FullApp\"]); structure observed on echo360.ca.",
        observed=True))


def extract_source_labels(scopes: list[Any]) -> tuple[dict[int, str], list[DiscoveryNote]]:
    """H-LABELS: a dict naming a source clearly. The source number comes either from a
    playlist reference in the same dict (``s<N>_av.m3u8`` / ``s<N>q<M>.m3u8``) or an
    explicit ``sourceIndex``-like integer; the label must not be generic. Conflicting
    labels for one source are discarded."""
    candidates: dict[int, set[str]] = {}
    keys_used: set[str] = set()
    for scope in scopes:
        for _path, node in iter_nodes(scope):
            if not isinstance(node, dict):
                continue
            index = None
            for key in _INDEX_KEYS:
                v = node.get(key)
                if isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 8:
                    index = v
                    break
            if index is None:
                refs = {int(m[1]) for v in node.values() if isinstance(v, str)
                        for m in [_SOURCE_REF.search(v)] if m}
                if len(refs) == 1:
                    index = refs.pop()
            if index is None:
                continue
            for key in _LABEL_KEYS:
                v = node.get(key)
                if not isinstance(v, str):
                    continue
                label = " ".join(v.split())
                if (0 < len(label) <= 60 and not _GENERIC_LABEL.match(label)
                        and "://" not in label and "/" not in label):
                    candidates.setdefault(index, set()).add(label)
                    keys_used.add(key)
                    break
    labels = {i: next(iter(v)) for i, v in candidates.items() if len(v) == 1}
    notes = []
    if labels:
        notes.append(_hyp("Source labels read from '" + "', '".join(sorted(keys_used)) + "'"))
    return labels, notes


def _language(node: dict[str, Any]) -> str | None:
    hit = _first_str(node, _LANG_KEYS)
    if hit and re.fullmatch(r"[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})*", hit[1]):
        return hit[1].replace("_", "-")
    return None


def _format_of(url: str) -> str | None:
    path = urlsplit(url).path
    m = _SUB_EXT.search(path)
    if m:
        return m[1].lower()
    low = url.lower()
    if any(w in low for w in ("caption", "transcript", "subtitle")):
        if path.lower().endswith(".json"):
            return "json"
        if path.lower().endswith(".txt"):
            return "txt"
    return None


def extract_captions(scopes: list[Any], html: str | None = None
                     ) -> tuple[list[CaptionTrack], list[DiscoveryNote]]:
    """H-CAPTION: caption/transcript resources referenced by lesson metadata."""
    found: dict[tuple[str | None, str], dict[str, Any]] = {}
    keys_used: set[str] = set()
    for scope in scopes:
        for _path, node in iter_nodes(scope):
            if not isinstance(node, dict):
                continue
            for key in _URL_KEYS:
                v = node.get(key)
                if not isinstance(v, str) or not is_echo360_url(v):
                    continue
                fmt = _format_of(v)
                if not fmt:
                    continue
                kind = ("captions" if fmt in ("vtt", "srt") else "transcript")
                lang = _language(node)
                entry = found.setdefault((lang, kind), {"formats": {}, "label": None})
                entry["formats"].setdefault(fmt, v)
                label = _first_str(node, ("label", "name"))
                if label and entry["label"] is None and len(label[1]) <= 60:
                    entry["label"] = label[1]
                keys_used.add(key)
    texts = [v for scope in scopes for _k, v in iter_strings(scope)]
    if html:
        texts.append(html.replace("\\/", "/"))
    for text in texts:
        for m in _CAPTION_URL_RE.finditer(text):
            url = m[0]
            if not is_echo360_url(url):
                continue
            if any(url in e["formats"].values() for e in found.values()):
                continue
            entry = found.setdefault((None, "captions"), {"formats": {}, "label": None})
            entry["formats"].setdefault("vtt", url)
            keys_used.add("captions.* URL pattern")
    tracks: list[CaptionTrack] = []
    for n, ((lang, kind), entry) in enumerate(sorted(found.items(), key=lambda kv: (
            kv[0][0] or "", kv[0][1])), start=1):
        timed = True if kind == "captions" else None
        tracks.append(CaptionTrack(
            id=f"{kind}-{lang or 'und'}-{n}", language=lang,
            kind=(TranscriptProvenance.PLATFORM_CAPTIONS if kind == "captions"
                  else TranscriptProvenance.PLATFORM_TRANSCRIPT),
            formats=dict(entry["formats"]), label=entry["label"], timed=timed,
            method=DiscoveryMethod.METADATA))
    notes = []
    guessed = sorted(k for k in keys_used if k != "captions.* URL pattern")
    if guessed:  # the captions.echo360.ca URL pattern itself is observed (O-CAPTION)
        notes.append(_hyp("Caption/transcript links read from '" + "', '".join(guessed) + "'"))
    return tracks, notes


def _chapter_time(node: dict[str, Any]) -> float | None:
    for key in _CH_MS_KEYS:
        v = node.get(key)
        if isinstance(v, int | float) and not isinstance(v, bool) and v >= 0:
            return v / 1000
    for key in _CH_S_KEYS:
        v = node.get(key)
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, int | float | str):
            try:
                return parse_time(v)
            except ValueError:
                continue
    return None


def extract_chapters(scopes: list[Any]) -> tuple[ChapterSet | None, list[DiscoveryNote]]:
    """H-CHAPTERS: a list under a chapter-like key whose items each have a title and a start
    time. Returns None unless every item parses (no partial guessing)."""
    for scope in scopes:
        for path, node in iter_nodes(scope):
            if not (isinstance(node, list) and node and path
                    and isinstance(path[-1], str) and _CHAPTER_LIST_KEY.match(path[-1])):
                continue
            chapters: list[Chapter] = []
            ok = True
            for item in node:
                if not isinstance(item, dict):
                    ok = False
                    break
                title = _first_str(item, _CH_TITLE_KEYS)
                start = _chapter_time(item)
                if title is None or start is None:
                    ok = False
                    break
                chapters.append(Chapter(start=start, title=title[1][:200]))
            if not ok or not chapters:
                continue
            chapters.sort(key=lambda c: c.start)
            dedup: list[Chapter] = []
            for ch in chapters:
                if dedup and abs(dedup[-1].start - ch.start) < 1e-6:
                    continue
                dedup.append(ch)
            return (ChapterSet(chapters=dedup, provenance=ChapterProvenance.NATIVE,
                               source_detail=f"Echo360 metadata '{path[-1]}' (hypothesis)"),
                    [_hyp(f"Native chapters read from '{path[-1]}'")])
    return None, []


# -- transcripts ---------------------------------------------------------------------------


def transcript_api_url(host: str, lesson_id: str, media_id: str) -> str:
    """H-TRANSCRIPT-API: candidate player transcript endpoint (from soraxas/echo360).
    Not observed on echo360.ca; callers must label its use as a hypothesis."""
    return (f"https://{host}/api/ui/echoplayer/lessons/{quote(lesson_id, safe='')}"
            f"/medias/{quote(media_id, safe='')}/transcript")


def _num(node: dict[str, Any], keys: tuple[str, ...], scale: float) -> float | None:
    for key in keys:
        v = node.get(key)
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, int | float):
            return v * scale if v >= 0 else None
        if isinstance(v, str):
            try:
                return parse_time(v) if scale == 1 else float(v) * scale
            except ValueError:
                continue
    return None


def _cue_list(obj: Any) -> list[dict[str, Any]] | None:
    """Prefer ``data.contentJSON.cues`` (H-TRANSCRIPT-JSON); else the first list of dicts
    that carry a text field."""
    try:
        preferred = obj["data"]["contentJSON"]["cues"]
        if isinstance(preferred, list):
            return preferred
    except (KeyError, TypeError, IndexError):
        pass
    for _path, node in iter_nodes(obj):
        if isinstance(node, list) and node and all(isinstance(i, dict) for i in node) and any(
                any(isinstance(i.get(k), str) for k in _CUE_TEXT_KEYS) for i in node):
            return node
    return None


def cues_from_json(obj: Any) -> list[Cue] | None:
    """Cues from an Echo360-style JSON transcript, or None if no cue list is present.
    Missing times stay None: untimed text is never given invented timestamps."""
    items = _cue_list(obj)
    if items is None:
        return None
    cues: list[Cue] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = _first_str(item, _CUE_TEXT_KEYS)
        if text is None:
            continue
        start = _num(item, _CUE_START_MS, 0.001)
        if start is None:
            start = _num(item, _CUE_START_S, 1.0)
        end = _num(item, _CUE_END_MS, 0.001)
        if end is None:
            end = _num(item, _CUE_END_S, 1.0)
        if start is None or (end is not None and end < start):
            start = end = None  # inconsistent or missing start: keep the text untimed
        speaker = _first_str(item, _SPEAKER_KEYS)
        cues.append(Cue(text=text[1], start=start, end=end,
                        speaker=speaker[1] if speaker else None))
    return cues


def transcript_label(obj: Any) -> str | None:
    """``data.isAutomated`` (O-TRANSCRIPT-API): Echo360's own speech recognition made it."""
    data = obj.get("data") if isinstance(obj, dict) else None
    automated = data.get("isAutomated") if isinstance(data, dict) else None
    if automated is True:
        return "Echo360 automatic transcript (speech recognition)"
    if automated is False:
        return "Echo360 transcript"
    return None


def parse_transcript_json(data: bytes) -> list[Cue]:
    try:
        obj = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        raise ValidationError("The transcript response is not valid JSON.") from None
    cues = cues_from_json(obj)
    if cues is None:
        raise ValidationError("The transcript JSON contains no recognisable cue list.")
    return cues


def looks_like_transcript(obj: Any) -> bool:
    """A JSON body that contains a cue list with at least one timed, texted entry."""
    items = _cue_list(obj)
    if not items:
        return False
    return any(isinstance(i, dict) and _first_str(i, _CUE_TEXT_KEYS) and (
        _num(i, _CUE_START_MS, 0.001) is not None or _num(i, _CUE_START_S, 1.0) is not None)
        for i in items[:20])


def parse_plain_text(data: bytes) -> list[Cue]:
    """Untimed transcript text: one cue per paragraph, start=end=None."""
    text = data.decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    return [Cue(text=p) for p in paragraphs if p]


def _short_hash(url: str) -> str:
    return hashlib.sha256(public_url(url).encode()).hexdigest()[:8]


_URL_LANG_PARAM = ("lang", "language", "locale", "languagecode", "srclang")
_URL_LANG_PATH = re.compile(r"(?:^|[/_.-])([a-z]{2,3}(?:-[a-z]{2})?)\.(?:vtt|srt)$")
_VTT_LANG_HEADER = re.compile(r"^Language:\s*([A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})?)\s*$",
                              re.MULTILINE)
#: Short path tokens that look like language codes but are not.
_NOT_LANG = {"vtt", "srt", "av", "cc", "sub", "subs", "web", "hls", "dash", "src", "s1", "s2"}


def caption_language(url: str, body: bytes | None) -> str | None:
    """Best-effort language of an observed caption file: the WebVTT ``Language:`` header
    (used by many platforms), then a URL parameter, then a ``…/en.vtt``-style file name.
    Returns None when nothing reliable is found (UIs then show "unknown", not "missing").
    """
    if body:
        head = body[:2048].decode("utf-8-sig", errors="replace")
        if head.startswith("WEBVTT"):
            header = head.split("\n\n", 1)[0]
            match = _VTT_LANG_HEADER.search(header)
            if match:
                return match[1].replace("_", "-")
    query = {k.lower(): v for k, v in parse_qs(urlsplit(url).query).items()}
    for key in _URL_LANG_PARAM:
        value = (query.get(key) or [""])[0].strip()
        if re.fullmatch(r"[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})?", value):
            return value.replace("_", "-")
    match = _URL_LANG_PATH.search(urlsplit(url).path.lower())
    if match and match[1] not in _NOT_LANG:
        from threech0.selection import is_known_language

        # File names are weak evidence: accept only recognised language codes.
        if is_known_language(match[1]):
            return match[1]
    return None


def caption_track_from_response(url: str, content_type: str | None, body: bytes | None
                                ) -> CaptionTrack | None:
    """An observed caption/transcript-looking response as a :class:`CaptionTrack`."""
    ctype = (content_type or "").split(";", 1)[0].strip().lower()
    path = urlsplit(url).path.lower()
    if ctype == "text/vtt" or path.endswith(".vtt"):
        fmt = "vtt"
    elif ctype in ("application/x-subrip", "text/srt") or path.endswith(".srt"):
        fmt = "srt"
    elif "json" in ctype and body:
        try:
            obj = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return None
        if not looks_like_transcript(obj):
            return None
        cues = cues_from_json(obj) or []
        return CaptionTrack(
            id=f"transcript-observed-{_short_hash(url)}", language=None,
            kind=TranscriptProvenance.PLATFORM_TRANSCRIPT, formats={"json": url},
            label=transcript_label(obj),
            timed=bool(cues) and all(c.start is not None and c.end is not None for c in cues),
            method=DiscoveryMethod.OBSERVED_REQUEST)
    else:
        return None
    return CaptionTrack(
        id=f"captions-observed-{_short_hash(url)}", language=caption_language(url, body),
        kind=TranscriptProvenance.PLATFORM_CAPTIONS, formats={fmt: url}, timed=True,
        method=DiscoveryMethod.OBSERVED_REQUEST)
