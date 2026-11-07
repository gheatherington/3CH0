"""Authenticated content session handed to the media pipeline (``models.ContentSession``)."""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable

import httpx

from threech0.echo360 import metadata
from threech0.echo360.http import FetchResult, HttpClient, Observer
from threech0.errors import (
    AuthError,
    AuthExpiredError,
    DependencyError,
    Echo360DLError,
    ForbiddenError,
    NetworkError,
    TranscriptUnavailableError,
    UnauthorizedError,
    UnsupportedError,
)
from threech0.models import AccessState, AuthContext, CaptionTrack, Transcript
from threech0.redact import scrub

MAX_REFRESHES = 2
LOGIN_HINT = "Run: 3ch0 auth login"
REQUEST_FILE_REASON = ("Authorization came from a saved request file, which cannot be "
                       "refreshed. Save a new request, or use browser sign-in instead.")
_FORMAT_PREFERENCE = ("vtt", "srt", "json", "txt")

Refresher = Callable[[], AuthContext]


class Echo360Session:
    """Implements :class:`threech0.models.ContentSession`.

    ``refresher`` re-captures authorization (a headless browser run with the same profile).
    Each :meth:`refresh` call invokes it at most once, and at most :data:`MAX_REFRESHES`
    times over the session's lifetime. Without a refresher (request-file mode) refresh
    raises :class:`AuthExpiredError` explaining why.
    """

    def __init__(self, auth: AuthContext, *, refresher: Refresher | None = None,
                 no_refresh_reason: str | None = None,
                 transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 observer: Observer | None = None,
                 max_refreshes: int = MAX_REFRESHES) -> None:
        self._auth = auth
        self._refresher = refresher
        self._no_refresh_reason = no_refresh_reason
        self._max_refreshes = max_refreshes
        self.refresh_count = 0
        self._http = HttpClient(auth, transport=transport, sleep=sleep, observer=observer)

    # -- ContentSession -------------------------------------------------------------------
    @property
    def auth(self) -> AuthContext:
        return self._auth

    def refresh(self) -> AuthContext:
        if self._refresher is None:
            raise AuthExpiredError(
                "Authorization cannot be refreshed. "
                + (self._no_refresh_reason or "No refresh method is available."),
                hint=LOGIN_HINT)
        if self.refresh_count >= self._max_refreshes:
            raise AuthExpiredError(
                f"Authorization expired again after {self.refresh_count} refreshes; giving up.",
                hint=LOGIN_HINT)
        self.refresh_count += 1
        try:
            fresh = self._refresher()
        except AuthError as exc:
            raise AuthExpiredError(f"Could not refresh authorization: {exc.message}",
                                   hint=exc.hint or LOGIN_HINT) from None
        except Echo360DLError as exc:
            raise AuthExpiredError(f"Could not refresh authorization: {exc.message}",
                                   hint=LOGIN_HINT) from None
        except Exception as exc:  # browser failures are third-party errors
            raise AuthExpiredError(
                f"Could not refresh authorization: {scrub(str(exc))[:300]}",
                hint=LOGIN_HINT) from None
        if not fresh.cookies:
            raise AuthExpiredError("Refreshing returned no credentials.", hint=LOGIN_HINT)
        self._auth = fresh
        self._http.auth = fresh
        return fresh

    def probe(self, url: str) -> AccessState:
        return self._http.probe(url).state

    def fetch_transcript(self, track: CaptionTrack, fmt: str | None = None) -> Transcript:
        if not track.formats:
            raise TranscriptUnavailableError(f"Transcript {track.id!r} has no downloadable "
                                             "format.")
        if fmt is not None:
            fmt = fmt.lower()
            if fmt not in track.formats:
                raise TranscriptUnavailableError(
                    f"Format {fmt!r} is not available for this transcript "
                    f"(available: {', '.join(sorted(track.formats))}).")
            chosen = fmt
        else:
            chosen = next((f for f in _FORMAT_PREFERENCE if f in track.formats),
                          sorted(track.formats)[0])
        result = self.fetch(track.formats[chosen])
        _raise_for(result, what="transcript")
        label: str | None = None
        data = result.content
        if chosen in ("vtt", "srt"):
            try:
                from threech0.media.subtitles import parse_subtitle_bytes
            except ImportError as exc:
                raise DependencyError("Subtitle parsing is not available "
                                      f"({type(exc).__name__}).") from None
            cues = parse_subtitle_bytes(data, chosen)
        elif chosen == "json":
            cues = metadata.parse_transcript_json(data)
            if track.label is None:
                with contextlib.suppress(ValueError, UnicodeDecodeError):
                    label = metadata.transcript_label(json.loads(data))
        elif chosen == "txt":
            cues = metadata.parse_plain_text(data)
        else:
            raise UnsupportedError(f"Transcript format {chosen!r} is not supported.")
        return Transcript(language=track.language, provenance=track.kind, cues=list(cues),
                          label=track.label or label, source_detail=f"Echo360 {chosen}",
                          raw=data, raw_format=chosen)

    # -- helpers --------------------------------------------------------------------------
    def fetch(self, url: str) -> FetchResult:
        return self._http.get(url)

    @property
    def http(self) -> HttpClient:
        return self._http

    def close(self) -> None:
        self._http.close()


def _raise_for(result: FetchResult, *, what: str) -> None:
    state = result.state
    if state is AccessState.AVAILABLE:
        return
    if state is AccessState.EXPIRED:
        raise AuthExpiredError(f"Could not fetch the {what}: {result.detail}", hint=LOGIN_HINT)
    if state is AccessState.UNAUTHORIZED:
        raise UnauthorizedError(f"Could not fetch the {what}: {result.detail}", hint=LOGIN_HINT)
    if state is AccessState.TRANSIENT:
        raise NetworkError(f"Could not fetch the {what}: {result.detail}")
    if state is AccessState.FORBIDDEN:
        raise ForbiddenError(f"Could not access the {what}: {result.detail}",
                             hint=LOGIN_HINT)
    raise TranscriptUnavailableError(f"The {what} is not available: {result.detail}")
