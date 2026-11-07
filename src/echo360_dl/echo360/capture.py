"""Browser-independent bookkeeping for observing the classroom player's network traffic.

:class:`CaptureState` receives :class:`ObservedResponse` events (from Playwright in
production, from fake event streams in tests), correlates them with the lesson and decides
when discovery is complete.

Completion condition (implemented by :meth:`CaptureState.completion`):

1. **complete** - at least one *successful* correlated playlist (master or media) has been
   seen, AND no *relevant* response (correlated playlist, Echo360 JSON, caption-like
   response) arrived for :data:`QUIET_PERIOD` seconds, AND every source number listed by
   lesson metadata (``s<N>_av`` playlist URLs found in correlated JSON) has been seen.
   Media segment traffic is deliberately ignored for quietness: it continues for as long
   as the video plays.
2. **settled** - as (1) but metadata lists sources that never appeared, and
   :data:`SETTLE_PERIOD` seconds have passed since the first playlist. Discovery continues
   (missing sources are fetched directly / probed) and a warning note is added.
3. **timeout** - the overall deadline passed. The caller returns what was found plus a
   warning, or raises if no playlist was seen.

Correlation: a content playlist belongs to the lesson when its ``x-lid`` equals the lesson
id (observed parameter). Without ``x-lid`` it belongs when its ``x-mid`` (or the media UUID
in its path) is a media id known from correlated metadata. Playlists naming another lesson
are rejected as prefetch.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit

from echo360_dl.echo360 import metadata
from echo360_dl.echo360.urls import (
    content_path_ids,
    is_echo360_url,
    lesson_param,
    media_param,
)
from echo360_dl.models import REPLAYABLE_HEADERS, DiscoveryNote
from echo360_dl.redact import public_url

QUIET_PERIOD = 3.0
SETTLE_PERIOD = 20.0
PLAY_HEURISTIC_AFTER = 8.0
MAX_JSON_DOCS = 300
MAX_BODY = 4 * 1024 * 1024

# Observed on echo360.ca (2026-11-07 trace): "s<N>_av.m3u8" (video + audio group) and
# "s<N>_v.m3u8" (video only) are both masters of source N. "s0q*" are the shared audio
# renditions, so index 0 is audio, never a video source.
_MASTER_NAME = re.compile(r"/s(\d{1,2})_(av|v)\.m3u8$", re.IGNORECASE)
AUDIO_SOURCE_INDEX = 0
_MEDIA_NAME = re.compile(r"/s(\d{1,2})q\d+\.m3u8$", re.IGNORECASE)


@dataclass
class ObservedResponse:
    url: str
    status: int
    content_type: str | None = None
    body: bytes | None = field(default=None, repr=False)
    method: str = "GET"
    resource_type: str | None = None
    request_headers: dict[str, str] = field(default_factory=dict, repr=False)


@dataclass
class ObservedPlaylist:
    url: str
    status: int
    kind: Literal["master", "media", "unknown"]
    source_index: int | None
    body: bytes | None = field(default=None, repr=False)


@dataclass(frozen=True)
class Completion:
    reason: Literal["complete", "settled", "timeout"]
    missing_sources: tuple[int, ...] = ()


def is_video_only_master(url: str) -> bool:
    m = _MASTER_NAME.search(urlsplit(url).path)
    return bool(m) and m[2].lower() == "v"


def playlist_kind(url: str, body: bytes | None) -> tuple[Literal["master", "media", "unknown"],
                                                         int | None]:
    path = urlsplit(url).path
    m = _MASTER_NAME.search(path)
    if m:
        return "master", int(m[1])
    m = _MEDIA_NAME.search(path)
    if m:
        index = int(m[1])
        # s0q* are audio renditions: still a playlist, but not a video source.
        return "media", (None if index == AUDIO_SOURCE_INDEX else index)
    if body and b"#EXT-X-STREAM-INF" in body:
        return "master", None
    if body and b"#EXTINF" in body:
        return "media", None
    return "unknown", None


def is_playlist_response(url: str, content_type: str | None) -> bool:
    ctype = (content_type or "").lower()
    return urlsplit(url).path.lower().endswith(".m3u8") or "mpegurl" in ctype


def _is_caption_like(url: str, content_type: str | None) -> bool:
    ctype = (content_type or "").split(";", 1)[0].strip().lower()
    path = urlsplit(url).path.lower()
    return ctype in ("text/vtt", "application/x-subrip", "text/srt") or path.endswith(
        (".vtt", ".srt"))


class CaptureState:
    """Accumulates correlated observations for one lesson."""

    def __init__(self, lesson_id: str, *, clock: Callable[[], float] | None = None,
                 quiet_period: float = QUIET_PERIOD, settle_period: float = SETTLE_PERIOD
                 ) -> None:
        self.lesson_id = lesson_id
        self._clock = clock or time.monotonic
        self.quiet_period = quiet_period
        self.settle_period = settle_period
        self.started = self._clock()
        self.playlists: dict[str, ObservedPlaylist] = {}  # key: public URL
        self.uncorrelated_playlists: list[ObservedPlaylist] = []
        self.rejected_playlists = 0
        self.other_lessons: set[str] = set()
        self.json_docs: list[tuple[str, Any]] = []
        self._json_seen: set[str] = set()  # public URLs of JSON documents already kept
        self.caption_responses: list[tuple[str, str | None, bytes | None, bool]] = []
        self.observed_media_ids: set[str] = set()
        self.request_headers: dict[str, str] = {}
        self.first_playlist_at: float | None = None
        self.last_relevant: float = self.started
        self.responses_seen = 0
        self.notes: list[DiscoveryNote] = []
        self._meta_cache: metadata.LessonMetadata | None = None
        self._meta_dirty = True

    # -- intake ---------------------------------------------------------------------------
    def restart_clock(self) -> None:
        """Reset timers (e.g. after an interactive login) without discarding observations."""
        self.started = self.last_relevant = self._clock()
        self._json_seen: set[str] = set()

    def add(self, resp: ObservedResponse) -> None:
        self.responses_seen += 1
        if not is_echo360_url(resp.url):
            return
        now = self._clock()
        if is_playlist_response(resp.url, resp.content_type):
            self._add_playlist(resp, now)
            return
        ctype = (resp.content_type or "").lower()
        body = resp.body if resp.body is not None and len(resp.body) <= MAX_BODY else None
        if _is_caption_like(resp.url, resp.content_type) and 200 <= resp.status < 300:
            self.caption_responses.append((resp.url, resp.content_type, body,
                                           self._mentions_lesson(resp.url, body)))
            self.last_relevant = now
            return
        if "json" in ctype and body and 200 <= resp.status < 300:
            key = public_url(resp.url)
            if key in self._json_seen:
                # Polling/heartbeat endpoints repeat: never extend the quiet window, and do
                # not let repeats crowd out new documents.
                return
            if len(self.json_docs) >= MAX_JSON_DOCS:
                return
            try:
                data = json.loads(body)
            except (ValueError, UnicodeDecodeError, RecursionError):
                return
            self._json_seen.add(key)
            self.json_docs.append((resp.url, data))
            self._meta_dirty = True
            if self._mentions_lesson(resp.url, body):
                # Only lesson-related metadata counts as discovery activity; unrelated
                # JSON (analytics, settings) is kept for parsing but does not delay
                # completion.
                self.last_relevant = now
            if metadata.looks_like_transcript(data):
                self.caption_responses.append((resp.url, resp.content_type, body,
                                               self._mentions_lesson(resp.url, body)))

    def _mentions_lesson(self, url: str, body: bytes | None) -> bool:
        needles = [self.lesson_id, *self.known_media_ids()]
        text = url + (body[:MAX_BODY].decode("utf-8", "replace") if body else "")
        return lesson_param(url) == self.lesson_id or any(n and n in text for n in needles)

    def _add_playlist(self, resp: ObservedResponse, now: float) -> None:
        kind, index = playlist_kind(resp.url, resp.body)
        entry = ObservedPlaylist(url=resp.url, status=resp.status, kind=kind,
                                 source_index=index, body=resp.body)
        lid = lesson_param(resp.url)
        if lid is not None and lid != self.lesson_id:
            self.rejected_playlists += 1
            self.other_lessons.add(lid)
            return
        if lid is None:
            mid = (media_param(resp.url) or content_path_ids(resp.url)[1] or "").lower()
            if not mid or mid not in self.known_media_ids():
                self.uncorrelated_playlists.append(entry)
                return
        self._accept(entry, resp.request_headers, now)

    def _accept(self, entry: ObservedPlaylist, request_headers: dict[str, str],
                now: float) -> None:
        key = public_url(entry.url)
        previous = self.playlists.get(key)
        if previous is None or (not 200 <= previous.status < 300 and 200 <= entry.status < 300):
            self.playlists[key] = entry
        mid = media_param(entry.url) or content_path_ids(entry.url)[1]
        if mid:
            self.observed_media_ids.add(mid.lower())
        if 200 <= entry.status < 300:
            if self.first_playlist_at is None:
                self.first_playlist_at = now
            if not self.request_headers:
                self.request_headers = {k.lower(): v for k, v in request_headers.items()
                                        if k.lower() in REPLAYABLE_HEADERS}
        self.last_relevant = now

    def finalize_uncorrelated(self) -> None:
        """Accept held playlists whose media id became known from metadata."""
        known = self.known_media_ids()
        keep: list[ObservedPlaylist] = []
        for entry in self.uncorrelated_playlists:
            mid = (media_param(entry.url) or content_path_ids(entry.url)[1] or "").lower()
            if mid and mid in known:
                self._accept(entry, {}, self._clock())
            else:
                keep.append(entry)
        self.uncorrelated_playlists = keep

    # -- derived views --------------------------------------------------------------------
    def correlated_json(self) -> list[tuple[str, Any]]:
        """JSON docs whose URL or body mentions the lesson id or a known media id."""
        needles = [n for n in (self.lesson_id, *sorted(self.observed_media_ids)) if n]
        out = []
        for url, data in self.json_docs:
            if any(n in url for n in needles) or any(metadata.mentions(data, n) for n in needles):
                out.append((url, data))
        return out

    def metadata(self, html: str | None = None) -> metadata.LessonMetadata:
        if self._meta_dirty or self._meta_cache is None or html is not None:
            self._meta_cache = metadata.extract_metadata(
                self.correlated_json(), self.lesson_id, html=html,
                known_media_ids=tuple(sorted(self.observed_media_ids)))
            self._meta_dirty = html is not None
        return self._meta_cache

    def known_media_ids(self) -> set[str]:
        ids = set(self.observed_media_ids)
        if self.json_docs:
            ids.update(self.metadata().media_ids)
        return ids

    def accounted_sources(self) -> set[int]:
        return {p.source_index for p in self.playlists.values()
                if p.source_index is not None and 200 <= p.status < 300}

    def successful_playlists(self) -> list[ObservedPlaylist]:
        return [p for p in self.playlists.values() if 200 <= p.status < 300]

    # -- completion -----------------------------------------------------------------------
    def completion(self, now: float, deadline: float) -> Completion | None:
        if self.first_playlist_at is not None and now - self.last_relevant >= self.quiet_period:
            expected = self.metadata().listed_source_indexes if self.json_docs else set()
            missing = tuple(sorted(expected - self.accounted_sources()))
            if not missing:
                return Completion("complete")
            if now - self.first_playlist_at >= self.settle_period:
                return Completion("settled", missing)
        if now >= deadline:
            expected = self.metadata().listed_source_indexes if self.json_docs else set()
            return Completion("timeout", tuple(sorted(expected - self.accounted_sources())))
        return None


def run_capture_loop(state: CaptureState, *, pump: Callable[[float], None],
                     clock: Callable[[], float], deadline: float,
                     play: Callable[[], str | None] | None = None,
                     play_after: float = PLAY_HEURISTIC_AFTER,
                     tick: float = 0.25,
                     on_tick: Callable[[], None] | None = None) -> Completion:
    """Pump events until :meth:`CaptureState.completion` says stop.

    ``pump(seconds)`` lets the browser deliver events (it may raise to abort, e.g. when the
    window was closed). If no playlist has appeared ``play_after`` seconds after the start,
    ``play()`` is called once (a labelled heuristic: click a visible play control); it
    returns a description of what it clicked, or None.
    """
    played = False
    while True:
        pump(tick)
        if on_tick is not None:
            on_tick()
        now = clock()
        done = state.completion(now, deadline)
        if done is not None:
            return done
        if (play is not None and not played and state.first_playlist_at is None
                and now - state.started >= play_after):
            played = True
            what = play()
            state.notes.append(DiscoveryNote(
                "info",
                (f"Heuristic: no media request after {play_after:.0f} s, so a visible play "
                 f"control was clicked ({what})." if what else
                 f"Heuristic: no media request after {play_after:.0f} s and no visible play "
                 "control was found to click."),
                observed=False))
