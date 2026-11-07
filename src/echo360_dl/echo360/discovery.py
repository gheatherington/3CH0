"""Authenticated discovery of an echo360.ca lesson: sources, audio, captions, chapters.

Browser mode (default): :func:`browser.capture_lesson` observes the classroom player's
traffic (completion condition documented in :mod:`echo360_dl.echo360.capture`), then
:func:`build_recording` fetches each correlated master playlist with httpx using the
browser's cookies (scoped exactly as Playwright reports them) and parses it.

Request-file mode: a saved "Copy as cURL" request supplies cookies for one stream URL;
sibling sources are found by naming-convention probing (labelled ``NAMING_PROBE``). Lesson
metadata, transcripts and chapters are unavailable in this mode.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from urllib.parse import urlsplit, urlunsplit

import httpx

from echo360_dl import paths
from echo360_dl.echo360 import metadata
from echo360_dl.echo360.capture import (
    AUDIO_SOURCE_INDEX,
    CaptureState,
    is_video_only_master,
    playlist_kind,
)
from echo360_dl.echo360.curl_import import load_request_file
from echo360_dl.echo360.hls import MasterPlaylist, PlaylistError, build_audio_tracks, parse_master
from echo360_dl.echo360.http import FetchResult
from echo360_dl.echo360.session import LOGIN_HINT, REQUEST_FILE_REASON, Echo360Session
from echo360_dl.echo360.trace import TraceRecorder
from echo360_dl.echo360.urls import (
    content_path_ids,
    is_echo360_url,
    lesson_param,
    media_param,
)
from echo360_dl.errors import (
    AuthExpiredError,
    Echo360DLError,
    UnauthorizedError,
    UnavailableError,
    UsageError,
)
from echo360_dl.models import (
    AccessState,
    AuthContext,
    CaptionTrack,
    ClassroomRef,
    ContentSession,
    DiscoveryMethod,
    DiscoveryNote,
    DiscoveryOptions,
    ProgressSink,
    Recording,
    Stage,
    TranscriptProvenance,
    VideoSource,
)
from echo360_dl.redact import public_url

MAX_SOURCE_INDEX = 8
PROBE_MISSES_ABOVE_HIGHEST = 2
_SOURCE_PATH = re.compile(r"/s(\d{1,2})(?:_av|_v|q\d+)\.m3u8$", re.IGNORECASE)


@dataclass
class _Candidate:
    index: int
    url: str
    method: DiscoveryMethod
    fallback_url: str | None = None  # e.g. the observed video-only master s<N>_v


def sibling_master_url(url: str, index: int) -> str | None:
    """``.../s2_av.m3u8?q`` or ``.../s2q1.m3u8?q`` -> ``.../s<index>_av.m3u8?q`` (naming
    convention observed on echo360.ca; None if the URL does not follow it)."""
    parts = urlsplit(url)
    if not _SOURCE_PATH.search(parts.path):
        return None
    path = _SOURCE_PATH.sub(f"/s{index}_av.m3u8", parts.path)
    return urlunsplit(parts._replace(path=path))


# -- public entry point -------------------------------------------------------------------


def discover(ref: ClassroomRef, options: DiscoveryOptions,
             progress: ProgressSink) -> tuple[Recording, ContentSession]:
    progress.stage(Stage.DISCOVER, "Discovering recording")
    trace = TraceRecorder() if options.save_trace else None
    try:
        if options.auth_mode == "request-file":
            return discover_from_request_file(ref, options, progress, trace=trace)
        return _discover_with_browser(ref, options, progress, trace=trace)
    finally:
        if trace is not None and options.save_trace is not None:
            path = trace.write(options.save_trace, lesson_id=ref.lesson_id)
            progress.info(f"Sanitised discovery trace written to {path}")


def _trace_observer(trace: TraceRecorder | None):
    if trace is None:
        return None

    def observe(method: str, url: str, status: int | None, ctype: str | None,
                body: bytes | None) -> None:
        trace.record(method, url, status, ctype, body, origin="httpx")
    return observe


def _discover_with_browser(ref: ClassroomRef, options: DiscoveryOptions,
                           progress: ProgressSink, *, trace: TraceRecorder | None
                           ) -> tuple[Recording, ContentSession]:
    from echo360_dl.echo360 import browser

    profile = options.profile_dir or paths.browser_profile_dir()
    outcome = browser.capture_lesson(
        ref, profile_dir=profile, interactive=options.interactive, headless=options.headless,
        timeout=options.timeout, login_timeout=options.login_timeout, progress=progress,
        trace=trace)

    def refresher() -> AuthContext:
        fresh = browser.capture_lesson(
            ref, profile_dir=profile, interactive=False, headless=True,
            timeout=min(options.timeout, 60.0), login_timeout=options.login_timeout,
            progress=progress)
        if not fresh.state.successful_playlists():
            raise AuthExpiredError("A fresh headless capture saw no media requests.",
                                   hint=LOGIN_HINT)
        return fresh.auth

    if trace is not None and outcome.html:
        trace.record_html(ref.url, outcome.html)
    session = Echo360Session(outcome.auth, refresher=refresher,
                             observer=_trace_observer(trace))
    try:
        recording = build_recording(ref, outcome.state, session, progress,
                                    completion_reason=outcome.completion.reason,
                                    missing_sources=outcome.completion.missing_sources,
                                    html=outcome.html, timeout=options.timeout)
    except BaseException:
        session.close()
        raise
    return recording, session


# -- recording assembly (browser-free, unit tested) -----------------------------------------


def build_recording(ref: ClassroomRef, state: CaptureState, session: Echo360Session,
                    progress: ProgressSink, *, completion_reason: str = "complete",
                    missing_sources: tuple[int, ...] = (), html: str | None = None,
                    timeout: float = 0.0) -> Recording:
    """Turn observations into a :class:`Recording` (fetches playlists via ``session``)."""
    notes: list[DiscoveryNote] = list(state.notes)
    meta = state.metadata(html)
    state.finalize_uncorrelated()
    if state.rejected_playlists:
        notes.append(DiscoveryNote(
            "info", f"Ignored {state.rejected_playlists} playlist request(s) for "
                    f"{len(state.other_lessons)} other lesson(s) (player prefetch)."))
    if state.uncorrelated_playlists:
        notes.append(DiscoveryNote(
            "info", f"Ignored {len(state.uncorrelated_playlists)} playlist request(s) that "
                    "could not be tied to this lesson."))
    if completion_reason == "settled" and missing_sources:
        notes.append(DiscoveryNote(
            "warning", "Metadata lists source(s) "
                       f"{', '.join(map(str, missing_sources))} that the player never "
                       "requested; they were fetched directly.", observed=False))
    if not state.playlists and not meta.playlist_urls:
        if completion_reason == "timeout":
            raise UnavailableError(
                f"No video playlist was requested by the player within {timeout:.0f} s.",
                hint="If the saved sign-in may have expired, run '3ch0 auth login'. "
                     "Otherwise check that the lesson plays in a browser, try a longer "
                     "--timeout, or use --save-trace to inspect what was observed.")
        raise UnavailableError("No video playlist was found for this lesson.",
                               hint="Use --save-trace to inspect what was observed.")
    if completion_reason == "timeout":
        notes.append(DiscoveryNote(
            "warning", f"Discovery stopped at the {timeout:.0f} s time limit; results may be "
                       "incomplete."))

    candidates = _candidates(state, meta)
    sources, masters = _fetch_sources(candidates, session, notes)
    listed = meta.listed_source_indexes
    unreachable: dict[int, AccessState] = {}
    if not listed:
        _probe_siblings(candidates, sources, masters, session, notes, unreachable)
    _finish_sources(sources, masters, meta, notes)

    tracks, mapping = build_audio_tracks(sorted(masters.items()))
    for src in sources:
        src.audio_track_id = mapping.get(src.index)

    captions = _captions(ref, state, meta, session, notes)
    observed_mid = sorted(state.observed_media_ids)
    media_id = (meta.media_ids[0] if meta.media_ids else observed_mid[0] if observed_mid
                else None)
    institution = next((content_path_ids(c.url)[0] for c in candidates
                        if content_path_ids(c.url)[0]), None)
    notes.extend(meta.notes)
    recording = Recording(
        lesson_id=ref.lesson_id, classroom_url=ref.url, host=ref.host, media_id=media_id,
        institution_id=institution, title=meta.title, course=meta.course,
        recorded_at=meta.recorded_at, duration=meta.duration,
        sources=sorted(sources, key=lambda s: s.index), audio_tracks=tracks,
        captions=captions, chapters=meta.chapters, notes=_dedupe(notes),
        unreachable_sources=unreachable)
    _check_usable(recording)
    progress.info(_summary(recording))
    return recording


def _candidates(state: CaptureState, meta: metadata.LessonMetadata) -> list[_Candidate]:
    by_index: dict[int, _Candidate] = {}
    unnamed: list[str] = []
    video_only: dict[int, str] = {}
    # Lesson metadata (the observed player object lists every source's playlists) is the
    # strongest evidence: its audio-paired masters come first, video-only ones are kept as
    # fallbacks. Playlists seen in traffic only fill sources the metadata did not list.
    meta_video_only: dict[int, str] = {}
    for url in meta.playlist_urls:
        kind, idx = playlist_kind(url, None)
        if kind != "master" or idx is None or idx == AUDIO_SOURCE_INDEX:
            continue
        if is_video_only_master(url):
            meta_video_only.setdefault(idx, url)
        elif idx not in by_index:
            by_index[idx] = _Candidate(idx, url, DiscoveryMethod.METADATA)
    for idx, url in meta_video_only.items():
        if idx in by_index:
            if by_index[idx].fallback_url is None:
                by_index[idx] = _Candidate(idx, by_index[idx].url, DiscoveryMethod.METADATA,
                                           fallback_url=url)
        else:
            by_index[idx] = _Candidate(idx, url, DiscoveryMethod.METADATA)
    for entry in state.playlists.values():
        idx = entry.source_index
        if entry.kind == "master" and idx is not None:
            if idx == AUDIO_SOURCE_INDEX or idx in by_index:
                continue
            if is_video_only_master(entry.url):
                video_only.setdefault(idx, entry.url)  # prefer s<N>_av (has audio pairing)
                continue
            by_index[idx] = _Candidate(idx, entry.url, DiscoveryMethod.OBSERVED_REQUEST)
        elif entry.kind == "master":
            unnamed.append(entry.url)
    for idx, url in video_only.items():
        if idx in by_index:
            continue
        sibling = sibling_master_url(url, idx)
        if sibling:  # try the audio-paired master first, fall back to the observed one
            by_index[idx] = _Candidate(idx, sibling, DiscoveryMethod.NAMING_PROBE,
                                       fallback_url=url)
        else:
            by_index[idx] = _Candidate(idx, url, DiscoveryMethod.OBSERVED_REQUEST)
    for entry in state.playlists.values():
        idx = entry.source_index
        if entry.kind == "media" and idx is not None and idx not in by_index:
            sibling = sibling_master_url(entry.url, idx)
            if sibling:
                by_index[idx] = _Candidate(idx, sibling, DiscoveryMethod.NAMING_PROBE)
    for url in meta.playlist_urls:
        kind, idx = playlist_kind(url, None)
        if kind == "master" and idx is not None and idx not in by_index \
                and idx != AUDIO_SOURCE_INDEX:
            by_index[idx] = _Candidate(idx, url, DiscoveryMethod.METADATA)
    next_index = max(by_index, default=0) + 1
    for url in unnamed:
        by_index[next_index] = _Candidate(next_index, url, DiscoveryMethod.OBSERVED_REQUEST)
        next_index += 1
    return [by_index[i] for i in sorted(by_index)]


def _fetch_master(url: str, session: Echo360Session) -> tuple[FetchResult, MasterPlaylist | None,
                                                               str | None]:
    result = session.fetch(url)
    if not result.ok:
        return result, None, result.detail
    try:
        master = parse_master(result.text, result.url)
    except PlaylistError as exc:
        return result, None, str(exc)
    if not master.is_master:
        return result, None, f"{public_url(url)} is not a master playlist"
    return result, master, None


def _fetch_sources(candidates: list[_Candidate], session: Echo360Session,
                   notes: list[DiscoveryNote]
                   ) -> tuple[list[VideoSource], dict[int, MasterPlaylist]]:
    sources: list[VideoSource] = []
    masters: dict[int, MasterPlaylist] = {}
    for cand in candidates:
        result, master, problem = _fetch_master(cand.url, session)
        if master is None and cand.fallback_url:
            cand = _Candidate(cand.index, cand.fallback_url, DiscoveryMethod.OBSERVED_REQUEST)
            result, master, problem = _fetch_master(cand.url, session)
        source = VideoSource(index=cand.index, playlist_url=cand.url, method=cand.method,
                             access=result.state, access_detail=problem)
        if master is not None:
            if master.drm:
                source.access = AccessState.UNSUPPORTED
                source.access_detail = "the playlist uses DRM, which is not supported"
            else:
                source.renditions = master.variants
                masters[cand.index] = master
        elif result.ok:
            source.access = AccessState.UNSUPPORTED
        if cand.method is DiscoveryMethod.NAMING_PROBE:
            if master is not None:
                notes.append(DiscoveryNote(
                    "info", f"Source {cand.index}: using s{cand.index}_av.m3u8, derived from "
                            "the observed playlist name; the server returned it.",
                    observed=True))
            else:
                notes.append(DiscoveryNote(
                    "info", f"Source {cand.index}: master playlist name derived from the "
                            "observed naming convention (s<N>_av.m3u8).", observed=False))
        if source.access is not AccessState.AVAILABLE:
            notes.append(DiscoveryNote(
                "warning", f"Source {cand.index}: {source.access_detail or source.access.value}"))
        sources.append(source)
    return sources, masters


def _probe_siblings(candidates: list[_Candidate], sources: list[VideoSource],
                    masters: dict[int, MasterPlaylist], session: Echo360Session,
                    notes: list[DiscoveryNote],
                    unreachable: dict[int, AccessState] | None = None) -> None:
    """Fallback: probe ``s<N>_av.m3u8`` siblings (N <= 8). Below the highest available
    source every gap is probed; above it probing stops after two consecutive misses.
    Probe failures never become sources: a 403 cannot prove absence and is reported."""
    template = next((c.url for c in candidates if sibling_master_url(c.url, 1)), None)
    if template is None:
        return
    known = {s.index for s in sources}
    highest = max((i for i in masters), default=0)
    misses = 0
    probed: list[int] = []
    for n in range(1, MAX_SOURCE_INDEX + 1):
        if n in known:
            continue
        if n > highest and misses >= PROBE_MISSES_ABOVE_HIGHEST:
            break
        url = sibling_master_url(template, n)
        if url is None:
            break
        probed.append(n)
        result, master, problem = _fetch_master(url, session)
        if master is not None and not master.drm:
            sources.append(VideoSource(index=n, playlist_url=url, renditions=master.variants,
                                       method=DiscoveryMethod.NAMING_PROBE,
                                       access=AccessState.AVAILABLE))
            masters[n] = master
            highest = max(highest, n)
            misses = 0
            continue
        if n > highest:
            misses += 1
        if unreachable is not None and result.state is not AccessState.NOT_FOUND:
            unreachable[n] = result.state
        if result.state is AccessState.FORBIDDEN:
            notes.append(DiscoveryNote(
                "info", f"Source {n}: could not access (403 does not prove absence).",
                observed=True))
        elif result.state is AccessState.NOT_FOUND:
            notes.append(DiscoveryNote("info", f"Source {n}: not found (HTTP 404/410)."))
        elif result.state in (AccessState.EXPIRED, AccessState.UNAUTHORIZED,
                              AccessState.TRANSIENT, AccessState.UNKNOWN):
            notes.append(DiscoveryNote("warning", f"Source {n}: {problem}"))
    if probed:
        notes.append(DiscoveryNote(
            "info", "Fallback: probed sibling playlist name(s) "
                    f"{', '.join(f's{n}_av' for n in probed)} by naming convention.",
            observed=False))


def _finish_sources(sources: list[VideoSource], masters: dict[int, MasterPlaylist],
                    meta: metadata.LessonMetadata, notes: list[DiscoveryNote]) -> None:
    seen_ids: set[str] = set()
    for src in sorted(sources, key=lambda s: s.index):
        renamed = []
        for r in src.renditions:
            rid = r.id if r.id not in seen_ids else f"s{src.index}-{r.id}"
            seen_ids.add(rid)
            renamed.append(r if rid == r.id else replace(r, id=rid))
        src.renditions = renamed
        label = meta.source_labels.get(src.index)
        if label:
            src.label = label
            src.label_method = DiscoveryMethod.METADATA
        if src.index in masters and masters[src.index].subtitle_groups:
            notes.append(DiscoveryNote(
                "info", f"Source {src.index}: the playlist lists HLS subtitle renditions, "
                        "which are not used."))


def _captions(ref: ClassroomRef, state: CaptureState, meta: metadata.LessonMetadata,
              session: Echo360Session, notes: list[DiscoveryNote]) -> list[CaptionTrack]:
    tracks: list[CaptionTrack] = list(meta.captions)
    seen = {public_url(u) for t in tracks for u in t.formats.values()}
    prefetch_seen = bool(state.other_lessons)
    assumed = 0
    for url, ctype, body, correlated in state.caption_responses:
        if public_url(url) in seen:
            continue
        if not correlated and prefetch_seen:
            continue  # other lessons were being prefetched; cannot attribute this one
        track = metadata.caption_track_from_response(url, ctype, body)
        if track is None:
            continue
        if not correlated:
            assumed += 1
        seen.add(public_url(url))
        tracks.append(track)
    if assumed:
        notes.append(DiscoveryNote(
            "info", f"{assumed} caption/transcript response(s) did not name the lesson; they "
                    "were assumed to belong to it (no other lesson was seen).", observed=False))
    # The transcript JSON shape (data.contentJSON.cues[] startMs/endMs/content) was observed
    # on echo360.ca on 2026-11-07, so no hypothesis note is needed for it any more.
    if tracks:
        return tracks
    media_ids = meta.media_ids or sorted(state.observed_media_ids)
    if not media_ids:
        return tracks
    url = metadata.transcript_api_url(ref.host, ref.lesson_id, media_ids[0])
    result = session.fetch(url)
    cues = None
    if result.ok and "json" in (result.content_type or "").lower():
        try:
            cues = metadata.parse_transcript_json(result.content)
        except Echo360DLError:
            cues = None
    if cues:
        tracks.append(CaptionTrack(
            id="transcript-und-api", language=None,
            kind=TranscriptProvenance.PLATFORM_TRANSCRIPT, formats={"json": url},
            timed=all(c.start is not None and c.end is not None for c in cues),
            method=DiscoveryMethod.NAMING_PROBE, access=AccessState.AVAILABLE))
        notes.append(DiscoveryNote(
            "info", "Transcript found at a candidate player endpoint "
                    "(/api/ui/echoplayer/lessons/…/medias/…/transcript, from a third-party "
                    "project; not verified for echo360.ca).", observed=False))
    else:
        notes.append(DiscoveryNote(
            "info", "No platform captions or transcript were found "
                    f"(candidate transcript endpoint: {result.state.value})."))
    return tracks


def _check_usable(recording: Recording) -> None:
    if any(s.access is AccessState.AVAILABLE for s in recording.sources):
        return
    states = {s.access for s in recording.sources}
    if AccessState.EXPIRED in states:
        raise AuthExpiredError("Authorization for this recording has expired.",
                               hint=LOGIN_HINT)
    if AccessState.UNAUTHORIZED in states:
        raise UnauthorizedError("Echo360 refused access to this recording's playlists.",
                                hint=LOGIN_HINT)
    if not recording.sources:
        raise UnavailableError("No accessible video source was found for this lesson.")
    recording.notes.append(DiscoveryNote(
        "warning", "No source could be fetched outside the browser; downloads will fail. "
                   "A 403 does not prove absence."))


def _summary(recording: Recording) -> str:
    ok = [s for s in recording.sources if s.access is AccessState.AVAILABLE]
    shared = " (shared audio)" if any(t.shared for t in recording.audio_tracks) else ""
    return (f"Found {len(ok)} accessible video source(s) of {len(recording.sources)}, "
            f"{len(recording.audio_tracks)} audio track(s){shared}, "
            f"{len(recording.captions)} caption/transcript resource(s).")


def _dedupe(notes: list[DiscoveryNote]) -> list[DiscoveryNote]:
    return list(dict.fromkeys(notes))


# -- request-file mode -----------------------------------------------------------------------


def discover_from_request_file(ref: ClassroomRef, options: DiscoveryOptions,
                               progress: ProgressSink, *, trace: TraceRecorder | None = None,
                               transport: httpx.BaseTransport | None = None,
                               sleep: Callable[[float], None] = time.sleep
                               ) -> tuple[Recording, ContentSession]:
    if options.request_file is None:
        raise UsageError("--request-file is required with this authentication mode.")
    imported = load_request_file(options.request_file)
    url = imported.url
    for warning in imported.warnings:
        progress.warn(warning)
    if not is_echo360_url(url):
        raise UsageError(f"The saved request is not for an Echo360 host ({public_url(url)}).",
                         hint="Save the request for the lesson's .m3u8 playlist.")
    lid = lesson_param(url)
    if lid is None:
        raise UsageError("The saved request has no x-lid parameter, so it cannot be matched "
                         "to this lesson.", hint="Save the request for the lesson's .m3u8 "
                                                 "playlist (it carries x-lid).")
    if lid != ref.lesson_id:
        raise UsageError("The saved request belongs to a different lesson than the URL given.",
                         hint="Save a request from this lesson's classroom page.")
    kind, index = playlist_kind(url, None)
    if index is None:
        raise UsageError(f"{public_url(url)} is not a recognised Echo360 stream playlist "
                         "(expected s<N>_av.m3u8 or s<N>q<M>.m3u8).")
    if imported.auth.is_expired():
        raise AuthExpiredError("The saved request's authorization has expired.",
                               hint="Save a fresh request, or use browser sign-in.")
    session = Echo360Session(imported.auth, refresher=None,
                             no_refresh_reason=REQUEST_FILE_REASON, transport=transport,
                             sleep=sleep, observer=_trace_observer(trace))
    try:
        notes = [DiscoveryNote("info", "Request-file mode: lesson metadata, transcripts and "
                                       "chapters are unavailable.")]
        master_url = url if kind == "master" else sibling_master_url(url, index)
        if master_url is None:
            raise UsageError(f"{public_url(url)} does not follow the s<N>_av naming.")
        method = (DiscoveryMethod.USER_INPUT if kind == "master"
                  else DiscoveryMethod.NAMING_PROBE)
        candidates = [_Candidate(index, master_url, method)]
        sources, masters = _fetch_sources(candidates, session, notes)
        first = sources[0]
        if first.access is not AccessState.AVAILABLE:
            if first.access in (AccessState.EXPIRED, AccessState.UNAUTHORIZED,
                                AccessState.FORBIDDEN):
                raise UnauthorizedError(
                    f"The saved request was refused: {first.access_detail}",
                    hint="The request may have expired; save a fresh one or use browser "
                         "sign-in.")
            raise UnavailableError(f"The saved request could not be used: "
                                   f"{first.access_detail}")
        unreachable: dict[int, AccessState] = {}
        _probe_siblings(candidates, sources, masters, session, notes, unreachable)
        _finish_sources(sources, masters, metadata.LessonMetadata(), notes)
        tracks, mapping = build_audio_tracks(sorted(masters.items()))
        for src in sources:
            src.audio_track_id = mapping.get(src.index)
        institution, path_media = content_path_ids(url)
        recording = Recording(
            lesson_id=ref.lesson_id, classroom_url=ref.url, host=ref.host,
            media_id=(media_param(url) or path_media), institution_id=institution,
            sources=sorted(sources, key=lambda s: s.index), audio_tracks=tracks,
            notes=_dedupe(notes), unreachable_sources=unreachable)
        progress.info(_summary(recording))
        return recording, session
    except BaseException:
        session.close()
        raise
