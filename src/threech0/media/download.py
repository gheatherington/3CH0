"""Component downloads with the yt-dlp Python API.

A *component* is one HLS media playlist: the video-only playlist of a selected
source (``s1q1.m3u8``) or a distinct audio rendition (``s0q1.m3u8``, shared by all
sources and downloaded once).

Observed layout (live, 2026-11-06): VOD fMP4, a single media file addressed with
``#EXT-X-MAP:URI=...,BYTERANGE=...`` plus ``#EXT-X-BYTERANGE`` segments. Because every
segment has a byte range, the exact component size is known from the playlist and
is used for real progress totals and for an exact completeness check.

Credentials: cookies go to yt-dlp only through a Netscape cookie file written by
:meth:`AuthContext.write_netscape_cookiejar` inside the private work dir (0700 dir,
0600 file), one per component attempt, deleted in ``finally``. Only replayable
headers are sent via ``http_headers``; never a ``Cookie`` header.

yt-dlp resume caveat (verified against yt-dlp 2026.10.26): when resuming an fMP4
download its HLS downloader skips completed media fragments but re-appends the
``#EXT-X-MAP`` init fragment, corrupting the output. On resume we therefore hand
yt-dlp a playlist in which the MAP line is rewritten as an ordinary byte-range
segment with the same fragment index, so it is skipped like any completed fragment.
The final size is then checked against the playlist; a mismatch discards the
component and downloads it once more from scratch.
"""

from __future__ import annotations

import atexit
import contextlib
import dataclasses
import http.cookiejar
import json
import os
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from threech0.errors import (
    AuthError,
    CancelledError,
    Echo360DLError,
    ForbiddenError,
    NetworkError,
    UnauthorizedError,
    UnavailableError,
    UnsupportedError,
)
from threech0.models import AuthContext, ContentSession, ProgressSink
from threech0.paths import ensure_private_dir
from threech0.redact import public_url, scrub

#: Seconds between playlist retries (multiplied by the attempt number). Tests lower it.
RETRY_BACKOFF = 0.5
MANIFEST_NAME = "manifest.json"
_HEADER_CASE = {"user-agent": "User-Agent", "referer": "Referer", "origin": "Origin",
                "accept-language": "Accept-Language"}


# --------------------------------------------------------------------------------------
# Components and the work directory
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Component:
    key: str  # file-safe, e.g. "video-s1q1" / "audio-s0q1"
    kind: Literal["video", "audio"]
    rendition_id: str
    url: str  # media playlist URL (may carry signed query; never displayed raw)
    label: str  # human description, e.g. "Source 1 video (720p)"
    target: str = ""  # Failure.target, e.g. "Source 1" or "audio"

    @property
    def filename(self) -> str:
        return f"{self.key}.mp4"


def component_key(kind: str, rendition_id: str) -> str:
    return f"{kind}-" + (re.sub(r"[^A-Za-z0-9_-]", "_", rendition_id) or "media")


class WorkDir:
    """``<output_dir>/.3ch0-<short_id>/``: components plus a small manifest of
    completed components (rendition id and size). Contains no secrets except
    short-lived cookie files that are deleted after each attempt."""

    def __init__(self, path: Path, lesson_id: str | None = None) -> None:
        self.path = ensure_private_dir(Path(path))
        self.lesson_id = lesson_id
        self._lock = threading.Lock()
        self._manifest = self._load()

    @property
    def manifest_path(self) -> Path:
        return self.path / MANIFEST_NAME

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.manifest_path.read_text("utf-8"))
        except (OSError, ValueError):
            data = {}
        if not isinstance(data, dict) or data.get("version") != 1:
            data = {}
        if self.lesson_id and data.get("lesson_id") not in (None, self.lesson_id):
            data = {}
        data.setdefault("version", 1)
        data.setdefault("lesson_id", self.lesson_id)
        if not isinstance(data.get("components"), dict):
            data["components"] = {}
        return data

    def _save(self) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._manifest, indent=2, sort_keys=True), "utf-8")
        os.replace(tmp, self.manifest_path)

    def file_for(self, component: Component) -> Path:
        return self.path / component.filename

    def completed(self, component: Component) -> Path | None:
        """Path of a previously completed component, if it is intact."""
        with self._lock:
            entry = self._manifest["components"].get(component.key)
        path = self.file_for(component)
        if not isinstance(entry, dict) or entry.get("rendition_id") != component.rendition_id:
            return None
        try:
            size = path.stat().st_size
        except OSError:
            return None
        return path if size == entry.get("size") and size > 0 else None

    def mark_complete(self, component: Component, size: int) -> None:
        with self._lock:
            self._manifest["components"][component.key] = {
                "kind": component.kind, "rendition_id": component.rendition_id,
                "file": component.filename, "size": size, "completed_at": time.time()}
            self._save()

    def forget(self, component: Component) -> None:
        with self._lock:
            self._manifest["components"].pop(component.key, None)
            self._save()

    def discard_partial(self, component: Component) -> None:
        """Remove the component file and yt-dlp resume state."""
        target = self.file_for(component)
        for path in self.path.glob(target.name + "*"):
            with contextlib.suppress(OSError):
                path.unlink()

    def has_partial(self, component: Component) -> bool:
        return (self.path / (component.filename + ".part")).exists()

    @contextlib.contextmanager
    def cookie_jar(self, component: Component, auth: AuthContext):
        """Write a 0600 cookie file for one attempt; always delete it afterwards.

        The jar lives in a private per-process temp dir (``$TMPDIR``, 0700), never in
        the output tree (which may be a synced folder), and contains only the cookies
        that match the component's URL.
        """
        directory = private_jar_dir()
        path = directory / f"{component.key}-{threading.get_ident()}.txt"
        try:
            auth.write_netscape_cookiejar(path, for_url=component.url)
            yield path
        finally:
            with contextlib.suppress(OSError):
                path.unlink()

    def remove_cookie_files(self) -> None:
        # Legacy location (inside the work dir) plus this process's private jar dir.
        for path in self.path.glob(".cookies-*"):
            with contextlib.suppress(OSError):
                path.unlink()
        remove_private_jar_dir()


_JAR_PREFIX = "3ch0-jar-"
atexit.register(lambda: remove_private_jar_dir())
_jar_dir: Path | None = None
_jar_lock = threading.Lock()


def private_jar_dir() -> Path:
    """Per-process 0700 directory for cookie jars; stale ones from dead runs are swept."""
    global _jar_dir
    with _jar_lock:
        if _jar_dir is None or not _jar_dir.is_dir():
            sweep_stale_jar_dirs()
            _jar_dir = Path(tempfile.mkdtemp(prefix=f"{_JAR_PREFIX}{os.getpid()}-"))
        return _jar_dir


def remove_private_jar_dir() -> None:
    global _jar_dir
    with _jar_lock:
        if _jar_dir is not None:
            shutil.rmtree(_jar_dir, ignore_errors=True)
            _jar_dir = None


def sweep_stale_jar_dirs() -> None:
    """Delete jar dirs left by processes that no longer exist (e.g. after SIGKILL)."""
    for entry in Path(tempfile.gettempdir()).glob(f"{_JAR_PREFIX}*"):
        try:
            pid = int(entry.name[len(_JAR_PREFIX):].split("-", 1)[0])
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            shutil.rmtree(entry, ignore_errors=True)
        except PermissionError:
            continue  # alive, owned by someone else


# --------------------------------------------------------------------------------------
# Shared, bounded authorisation refresh
# --------------------------------------------------------------------------------------


class _RefreshExhausted(Exception):
    pass


class AuthRefresher:
    """At most ``limit`` ``session.refresh()`` calls per run, shared by all workers.

    A worker that hit a 403 with credentials of generation *g* calls
    :meth:`refreshed_since`. If another worker already refreshed (generation > *g*)
    it simply retries with the new credentials.
    """

    def __init__(self, session: ContentSession, limit: int = 1) -> None:
        self.session = session
        self.limit = limit
        self.calls = 0
        self.generation = 0
        self._auth = session.auth
        self._lock = threading.Lock()
        self._error: AuthError | None = None

    def current(self) -> tuple[int, AuthContext]:
        with self._lock:
            return self.generation, self._auth

    def refreshed_since(self, generation: int) -> AuthContext:
        with self._lock:
            if self.generation > generation:
                return self._auth
            if self._error is not None:
                raise self._error
            if self.calls >= self.limit:
                raise _RefreshExhausted
            self.calls += 1
            try:
                self._auth = self.session.refresh()
            except AuthError as exc:
                self._error = exc
                raise
            except Echo360DLError as exc:
                self._error = UnauthorizedError(scrub(exc.message), hint=exc.hint)
                raise self._error from None
            self.generation += 1
            return self._auth


# --------------------------------------------------------------------------------------
# Playlist helpers
# --------------------------------------------------------------------------------------


def _attrs(text: str) -> dict[str, str]:
    return {k: v.strip('"') for k, v in re.findall(r'([A-Z0-9-]+)=("[^"]*"|[^,]*)', text)}


def playlist_total_bytes(playlist: str) -> int | None:
    """Exact byte size when every segment (and the MAP) has a byte range."""
    total = 0
    pending_range: int | None = None
    segments = 0
    for raw in playlist.splitlines():
        line = raw.strip()
        if line.startswith("#EXT-X-MAP"):
            rng = _attrs(line.partition(":")[2]).get("BYTERANGE")
            if not rng:
                return None
            total += int(rng.split("@")[0])
        elif line.startswith("#EXT-X-BYTERANGE:"):
            pending_range = int(line.partition(":")[2].split("@")[0])
        elif line and not line.startswith("#"):
            if pending_range is None:
                return None
            total += pending_range
            pending_range = None
            segments += 1
    return total if segments else None


def resume_playlist(playlist: str) -> str:
    """Rewrite ``#EXT-X-MAP`` as an ordinary byte-range segment (same fragment index)
    so yt-dlp skips it on resume instead of appending it again."""
    def convert(match: re.Match[str]) -> str:
        attrs = _attrs(match.group(1))
        uri = attrs.get("URI")
        if not uri:
            return match.group(0)
        rng = attrs.get("BYTERANGE")
        return (f"#EXT-X-BYTERANGE:{rng}\n{uri}" if rng else uri)

    return re.sub(r"(?m)^#EXT-X-MAP:(.*)$", convert, playlist)


def _ytdl_resume_index(path: Path) -> int:
    try:
        data = json.loads((path.parent / (path.name + ".ytdl")).read_text("utf-8"))
        return int(data["downloader"]["current_fragment"]["index"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0


# --------------------------------------------------------------------------------------
# yt-dlp integration
# --------------------------------------------------------------------------------------


class _HTTPStatusError(Exception):
    def __init__(self, status: int | None, message: str) -> None:
        super().__init__(message)
        self.status = status


class _SizeMismatch(Exception):
    def __init__(self, size: int, expected: int) -> None:
        super().__init__(f"size {size} != expected {expected}")
        self.size = size
        self.expected = expected


class _CancelSignal(Exception):
    """Raised from the progress hook to stop yt-dlp."""


class _Logger:
    def __init__(self, progress: ProgressSink, label: str) -> None:
        self.progress = progress
        self.label = label
        self.errors: list[str] = []

    def debug(self, msg: str) -> None:
        if not msg.startswith("[download]"):
            self.progress.debug(f"{self.label}: {scrub(msg)}")

    def info(self, msg: str) -> None:
        self.debug(msg)

    def warning(self, msg: str) -> None:
        self.progress.debug(f"{self.label}: {scrub(msg)}")

    def error(self, msg: str) -> None:
        self.errors.append(scrub(msg))


def _status_of(exc: BaseException) -> int | None:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = getattr(current, "status", None) or getattr(current, "code", None)
        if isinstance(status, int):
            return status
        info = getattr(current, "exc_info", None)
        if info and len(info) > 1 and isinstance(info[1], BaseException):
            current = info[1]
            continue
        current = current.__cause__ or current.__context__
    match = re.search(r"HTTP Error (\d{3})", str(exc))
    return int(match.group(1)) if match else None


def _http_headers(auth: AuthContext) -> dict[str, str]:
    return {_HEADER_CASE.get(k, k): v for k, v in auth.headers.items() if k in _HEADER_CASE}


@dataclass
class DownloadContext:
    work: WorkDir
    refresher: AuthRefresher
    progress: ProgressSink
    cancel: threading.Event
    retries: int = 3
    #: Media playlists read ahead by :func:`prefetch_size`, used once by the download.
    playlists: dict[str, str] = field(default_factory=dict)
    started: set[str] = field(default_factory=set)
    _locks: dict[str, threading.Lock] = field(default_factory=dict)
    _locks_guard: threading.Lock = field(default_factory=threading.Lock)

    def playlist_lock(self, key: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())


def _ydl_options(jar: Path, auth: AuthContext, ctx: DownloadContext, component: Component,
                 logger: _Logger, hook: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    return {
        "paths": {"home": str(ctx.work.path)},
        "outtmpl": {"default": f"{component.key}.%(ext)s"},
        "cookiefile": str(jar),
        "http_headers": _http_headers(auth),
        "continuedl": True,
        "nopart": False,
        "retries": ctx.retries,
        "fragment_retries": ctx.retries,
        "file_access_retries": 3,
        "skip_unavailable_fragments": False,  # a missing fragment must fail, not be skipped
        "concurrent_fragment_downloads": 1,
        "fixup": "never",
        "hls_use_mpegts": False,
        "format": "media",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": logger,
        "progress_hooks": [hook],
        "cachedir": False,
        "noplaylist": True,
        "socket_timeout": 30,
        "color": "no_color",
    }


def _fetch_playlist(ydl: Any, url: str, ctx: DownloadContext, label: str) -> str:
    from yt_dlp.networking.exceptions import HTTPError, TransportError

    last: Exception | None = None
    for attempt in range(ctx.retries + 1):
        if ctx.cancel.is_set():
            raise CancelledError("Download cancelled.")
        try:
            with ydl.urlopen(url) as response:
                return response.read().decode("utf-8", "replace")
        except HTTPError as exc:
            status = exc.status
            if status in (401, 403, 404, 410):
                raise _HTTPStatusError(status, f"HTTP {status}") from None
            last = exc
        except TransportError as exc:
            last = exc
        if attempt < ctx.retries:
            time.sleep(RETRY_BACKOFF * (attempt + 1))
    raise NetworkError(f"{label}: could not fetch the playlist after {ctx.retries + 1} "
                       f"attempts ({scrub(str(last))}).")


@contextlib.contextmanager
def _ydl_session(component: Component, ctx: DownloadContext, auth: AuthContext,
                 logger: _Logger, hook: Callable[[dict[str, Any]], None]):
    """A YoutubeDL instance with this component's cookie jar (deleted afterwards)."""
    import yt_dlp

    with ctx.work.cookie_jar(component, auth) as jar, \
            yt_dlp.YoutubeDL(_ydl_options(jar, auth, ctx, component, logger, hook)) as ydl:
        # Python's default policy lets host-only cookies match subdomains; be strict.
        ydl.cookiejar.set_policy(http.cookiejar.DefaultCookiePolicy(
            strict_ns_domain=http.cookiejar.DefaultCookiePolicy.DomainStrictNonDomain))
        yield ydl


def prefetch_size(component: Component, ctx: DownloadContext) -> int | None:
    """Read the media playlist of a component that is still waiting for a worker, to learn
    its exact size early. The text is kept and used by the download itself, so this adds
    no request (except when this read fails and the download asks again). Best effort:
    one attempt, any failure returns None and the download proceeds as usual."""
    if ctx.cancel.is_set() or ctx.work.completed(component) is not None:
        return None
    with ctx.playlist_lock(component.key):
        if component.key in ctx.started or component.key in ctx.playlists:
            return None
        _generation, auth = ctx.refresher.current()
        once = dataclasses.replace(ctx, retries=0)
        try:
            with _ydl_session(component, ctx, auth, _Logger(ctx.progress, component.label),
                              lambda _d: None) as ydl:
                text = _fetch_playlist(ydl, component.url, once, component.label)
        except Exception as exc:  # best effort; the download reports real problems
            ctx.progress.debug(f"{component.label}: size not read ahead ({scrub(str(exc))})")
            return None
        if "#EXTM3U" not in text or "#EXT-X-STREAM-INF" in text:
            return None
        ctx.playlists[component.key] = text
        return playlist_total_bytes(text)


def _attempt(component: Component, ctx: DownloadContext, auth: AuthContext,
             task_id: str) -> Path:
    from yt_dlp.utils import DownloadError

    target = ctx.work.file_for(component)
    logger = _Logger(ctx.progress, component.label)
    totals: dict[str, int | None] = {"total": None}

    def hook(d: dict[str, Any]) -> None:
        if ctx.cancel.is_set():
            raise _CancelSignal
        if d.get("status") not in ("downloading", "finished"):
            return
        done = d.get("downloaded_bytes")
        total = totals["total"] or d.get("total_bytes")
        speed = d.get("speed")
        eta = (total - done) / speed if total and done is not None and speed else None
        ctx.progress.task_update(task_id, completed=done, total=total, rate=speed, eta=eta)

    with _ydl_session(component, ctx, auth, logger, hook) as ydl:
        with ctx.playlist_lock(component.key):
            ctx.started.add(component.key)
            playlist = ctx.playlists.pop(component.key, None)  # read ahead, used once
        if playlist is None:
            playlist = _fetch_playlist(ydl, component.url, ctx, component.label)
        if "#EXTM3U" not in playlist:
            raise UnsupportedError(f"{component.label}: the server did not return an HLS "
                                   "playlist.")
        if "#EXT-X-STREAM-INF" in playlist:
            raise UnsupportedError(f"{component.label}: expected a media playlist but got a "
                                   "master playlist.")
        total = playlist_total_bytes(playlist)
        totals["total"] = total
        ctx.progress.task_update(task_id, total=total)
        data = playlist
        if ctx.work.has_partial(component) and _ytdl_resume_index(target) >= 1:
            data = resume_playlist(playlist)
        info = {
            "id": component.key, "title": component.key, "extractor": "3ch0",
            "extractor_key": "Echo360DL", "webpage_url": public_url(component.url),
            "formats": [{
                "format_id": "media", "url": component.url, "protocol": "m3u8_native",
                "ext": "mp4", "hls_media_playlist_data": data,
                "vcodec": "none" if component.kind == "audio" else None,
                "acodec": "none" if component.kind == "video" else None,
            }],
        }
        try:
            ydl.process_ie_result(info, download=True)
        except _CancelSignal:
            raise CancelledError("Download cancelled.") from None
        except DownloadError as exc:
            status = _status_of(exc)
            message = scrub(str(exc)) or "; ".join(logger.errors)
            if status in (401, 403, 404, 410):
                raise _HTTPStatusError(status, message) from None
            raise NetworkError(f"{component.label}: download failed: "
                               f"{message.removeprefix('ERROR: ')}") from None
    if not target.is_file():
        raise NetworkError(f"{component.label}: download finished without an output file.")
    size = target.stat().st_size
    if total is not None and size != total:
        raise _SizeMismatch(size, total)
    return target


def component_task_id(component: Component) -> str:
    """Progress task id of a component download (``download:<kind>-<rendition>``)."""
    return f"download:{component.key}"


def download_component(component: Component, ctx: DownloadContext) -> Path:
    """Download (or reuse/resume) one component. Raises Echo360DLError subclasses:
    AuthError (refresh failed), UnavailableError (403/404 after refresh),
    NetworkError, UnsupportedError or CancelledError."""
    task_id = component_task_id(component)
    reused = ctx.work.completed(component)
    ctx.progress.task_start(task_id, component.label, total=None, unit="bytes")
    if reused is not None:
        size = reused.stat().st_size
        ctx.progress.task_update(task_id, completed=size, total=size)
        ctx.progress.task_done(task_id, ok=True, message="reused from an earlier run")
        return reused
    refreshed_for_me = False
    restarted = False
    try:
        while True:
            if ctx.cancel.is_set():
                raise CancelledError("Download cancelled.")
            generation, auth = ctx.refresher.current()
            try:
                path = _attempt(component, ctx, auth, task_id)
            except _HTTPStatusError as exc:
                if exc.status in (404, 410):
                    raise UnavailableError(
                        f"{component.label}: not found on the server (HTTP {exc.status}).",
                    ) from None
                if refreshed_for_me:
                    raise _forbidden(component, exc.status) from None
                try:
                    ctx.refresher.refreshed_since(generation)
                except _RefreshExhausted:
                    raise _forbidden(component, exc.status) from None
                refreshed_for_me = True
                ctx.progress.debug(f"{component.label}: HTTP {exc.status}; retrying with "
                                   "refreshed authorisation")
                continue
            except _SizeMismatch as exc:
                ctx.work.discard_partial(component)
                if restarted:
                    raise NetworkError(
                        f"{component.label}: downloaded size {exc.size} does not match the "
                        f"playlist ({exc.expected} bytes).") from None
                restarted = True
                ctx.progress.debug(f"{component.label}: incomplete/corrupt resume; "
                                   "restarting this component")
                continue
            size = path.stat().st_size
            ctx.work.mark_complete(component, size)
            ctx.progress.task_done(task_id, ok=True)
            return path
    except CancelledError:
        ctx.progress.task_done(task_id, ok=False, message="cancelled (partial data kept)")
        raise
    except Echo360DLError as exc:
        ctx.progress.task_done(task_id, ok=False, message=scrub(exc.message))
        raise


def _forbidden(component: Component, status: int | None) -> Echo360DLError:
    if status == 401:
        return UnauthorizedError(
            f"{component.label}: authorisation refused (HTTP 401) even after refreshing it.",
            hint="Run `3ch0 auth login` and try again.")
    return ForbiddenError(
        f"{component.label}: access refused (HTTP {status or 403}) even with refreshed "
        "authorisation.",
        hint="Echo360 answers 403 both for content this account may not access and for "
             "content that does not exist. If it plays in the browser, run "
             "`3ch0 auth login` and try again.")
