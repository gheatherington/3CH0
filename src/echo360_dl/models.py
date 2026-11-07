"""Shared data models and interface contracts.

This module is the contract between the three areas of the code base:

* ``echo360_dl.echo360`` (discovery/authentication) *produces* :class:`Recording`,
  :class:`AuthContext` and platform :class:`Transcript` objects.
* ``echo360_dl.media`` (download/subtitles/chapters/muxing) *consumes* a
  :class:`DownloadRequest` and produces a :class:`DownloadResult`.
* ``echo360_dl.cli`` builds requests, renders models and reports progress through
  the :class:`ProgressSink` protocol.

Rules:
* Models never hold credentials except :class:`AuthContext`, which is never
  serialised by :func:`to_jsonable` and has a redacted ``repr``.
* URLs stored on models may carry query strings needed for fetching; use
  :func:`echo360_dl.redact.public_url` before displaying or serialising them.
  :func:`to_jsonable` does this automatically for fields named ``*url``.
* Times are seconds as ``float`` relative to the start of the recording.
"""

from __future__ import annotations

import dataclasses
import enum
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------


class AccessState(enum.StrEnum):
    """Result of trying to access a resource. A 403 alone is never ``NOT_FOUND``."""

    AVAILABLE = "available"
    UNAUTHORIZED = "unauthorized"  # definitive auth refusal (e.g. 401, login redirect)
    EXPIRED = "expired"  # credentials known to be past expiry
    FORBIDDEN = "forbidden"  # 403: absent *or* unauthorized - cannot tell which
    NOT_FOUND = "not_found"  # 404/410, or absent according to authoritative metadata
    UNSUPPORTED = "unsupported"  # exists but unusable here (DRM, unknown format)
    TRANSIENT = "transient"  # timeout, 5xx, connection reset - retry may help
    UNKNOWN = "unknown"  # not probed


class DiscoveryMethod(enum.StrEnum):
    """How a piece of information was obtained (evidence strength, strongest first)."""

    METADATA = "metadata"  # parsed from a verified Echo360 metadata response
    OBSERVED_REQUEST = "observed-request"  # seen in the player's own network traffic
    PLAYLIST = "playlist"  # parsed from an HLS playlist we fetched
    NAMING_PROBE = "naming-probe"  # guessed from URL naming convention, then probed
    USER_INPUT = "user-input"  # supplied by the user (e.g. imported cURL request)


class TranscriptProvenance(enum.StrEnum):
    PLATFORM_CAPTIONS = "platform-captions"  # timed captions published by Echo360
    PLATFORM_TRANSCRIPT = "platform-transcript"  # Echo360 transcript (may be untimed)
    GENERATED_LOCAL = "generated-local"  # produced on this machine by speech recognition
    USER_FILE = "user-file"  # subtitle file supplied by the user


class ChapterProvenance(enum.StrEnum):
    NATIVE = "native"  # Echo360 lesson metadata
    USER_FILE = "user-file"  # --chapters-file
    GENERATED_INTERVAL = "generated-interval"  # fixed intervals, neutral titles
    GENERATED_HEURISTIC = "generated-heuristic"  # transcript-based, editable, not authoritative


class Container(enum.StrEnum):
    MP4 = "mp4"
    MKV = "mkv"


class SubtitleModeKind(enum.StrEnum):
    NONE = "none"
    AUTO = "auto"  # every platform caption/transcript language that is timed; report absence
    LANGUAGES = "languages"  # explicit list; missing language -> failure (documented)


class ChapterModeKind(enum.StrEnum):
    NONE = "none"
    AUTO = "auto"  # native chapters only, never fabricated
    FILE = "file"  # user chapter JSON (takes precedence over native)
    INTERVAL = "interval"  # generated fixed-interval chapters


class Layout(enum.StrEnum):
    """How selected video sources are written.

    SEPARATE keeps each source in its own stream-copied file (original quality). The
    composite layouts place two sources next to each other in ONE file; that requires
    re-encoding the video (audio, subtitles and chapters are carried over unchanged).
    """

    SEPARATE = "separate"
    SIDE_BY_SIDE = "side-by-side"  # first source left, second right
    TOP_BOTTOM = "top-bottom"  # first source on top, second below


class Encoder(enum.StrEnum):
    AUTO = "auto"  # Apple VideoToolbox when available, else libx264
    VIDEOTOOLBOX = "videotoolbox"  # hardware H.264 on macOS
    X264 = "x264"  # libx264 software encoder (portable, slower)


class IfExists(enum.StrEnum):
    SKIP = "skip"  # leave existing final file, report it as skipped
    OVERWRITE = "overwrite"  # atomically replace
    ERROR = "error"  # fail that output


class OutputKind(enum.StrEnum):
    VIDEO = "video"
    COMPOSITE = "composite"  # two sources in one re-encoded video (side by side / stacked)
    AUDIO = "audio"
    SUBTITLE = "subtitle"  # sidecar subtitle file
    TRANSCRIPT = "transcript"  # sidecar transcript (txt/json)
    CHAPTERS = "chapters"  # sidecar chapter JSON


class ResultStatus(enum.StrEnum):
    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Stage(enum.StrEnum):
    """User-visible pipeline stages, in order."""

    DISCOVER = "discover"
    SELECT = "select"
    DOWNLOAD = "download"
    SUBTITLES = "subtitles"
    CHAPTERS = "chapters"
    VERIFY = "verify"


# --------------------------------------------------------------------------------------
# Discovery results
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassroomRef:
    """A parsed, validated classroom URL. ``url`` has no query/fragment."""

    host: str  # e.g. "echo360.ca"
    lesson_id: str
    url: str


@dataclass(frozen=True)
class VideoRendition:
    """One video variant of a source (one ``#EXT-X-STREAM-INF`` entry)."""

    id: str  # stable within the recording, e.g. "s1q1"
    url: str  # media playlist URL (may contain query string)
    width: int | None = None
    height: int | None = None
    bandwidth: int | None = None
    average_bandwidth: int | None = None
    codecs: str | None = None
    frame_rate: float | None = None
    audio_group: str | None = None  # HLS AUDIO group id this variant pairs with

    @property
    def resolution(self) -> str | None:
        return f"{self.width}x{self.height}" if self.width and self.height else None


@dataclass(frozen=True)
class AudioRendition:
    """One audio variant (``#EXT-X-MEDIA:TYPE=AUDIO``)."""

    id: str  # e.g. "s0q1"
    url: str
    group_id: str | None = None
    name: str | None = None
    language: str | None = None  # BCP-47 / ISO code as published, if any
    default: bool = False
    bandwidth: int | None = None  # rarely published; may be inferred from paired video


@dataclass
class VideoSource:
    """One Echo360 video source (camera, screen capture, ...).

    ``index`` is the platform's source number (1-based, ``s1``/``s2`` on observed
    recordings). ``label`` is set only when metadata reliably names the source;
    otherwise UIs show :attr:`display_name` ("Source 1").
    """

    index: int
    playlist_url: str | None
    renditions: list[VideoRendition] = field(default_factory=list)
    label: str | None = None
    label_method: DiscoveryMethod | None = None
    method: DiscoveryMethod = DiscoveryMethod.PLAYLIST
    access: AccessState = AccessState.UNKNOWN
    access_detail: str | None = None  # e.g. "HTTP 403" - never contains secrets
    audio_track_id: str | None = None  # AudioTrack.id paired with this source, if known

    @property
    def display_name(self) -> str:
        return self.label or f"Source {self.index}"

    @property
    def max_height(self) -> int | None:
        heights = [r.height for r in self.renditions if r.height]
        return max(heights) if heights else None

    @property
    def has_audio(self) -> bool:
        return any(r.audio_group for r in self.renditions)


@dataclass
class AudioTrack:
    """A logical audio track (all renditions of one programme audio).

    ``shared`` is True when several video sources reference the same renditions
    (observed: ``s1_av`` and ``s2_av`` both point at ``s0q0``/``s0q1``). Shared
    audio is downloaded once.
    """

    id: str  # e.g. "s0"
    renditions: list[AudioRendition] = field(default_factory=list)
    language: str | None = None
    shared: bool = False
    used_by_sources: list[int] = field(default_factory=list)


@dataclass
class CaptionTrack:
    """A transcript/caption *resource* discovered on the platform (not yet fetched).

    ``formats`` maps a format name ("vtt", "srt", "txt", "json", ...) to a URL.
    ``timed`` is None when the format's timing is not yet known.
    """

    id: str
    language: str | None
    kind: TranscriptProvenance  # PLATFORM_CAPTIONS or PLATFORM_TRANSCRIPT
    formats: dict[str, str] = field(default_factory=dict)
    label: str | None = None
    timed: bool | None = None
    method: DiscoveryMethod = DiscoveryMethod.METADATA
    access: AccessState = AccessState.UNKNOWN


@dataclass(frozen=True)
class Chapter:
    start: float
    title: str
    end: float | None = None  # resolved from next start / media duration when None


@dataclass
class ChapterSet:
    chapters: list[Chapter]
    provenance: ChapterProvenance
    source_detail: str | None = None  # e.g. file name, "every 600 s", endpoint kind


@dataclass(frozen=True)
class DiscoveryNote:
    level: Literal["info", "warning"]
    message: str  # human readable; never contains secrets
    observed: bool = True  # False -> hypothesis / inference, labelled as such in UI


@dataclass
class Recording:
    """Everything known about one lesson's recording."""

    lesson_id: str
    classroom_url: str  # sanitised (no query)
    host: str
    media_id: str | None = None
    institution_id: str | None = None
    title: str | None = None
    course: str | None = None
    recorded_at: str | None = None  # ISO 8601 when known
    duration: float | None = None  # seconds
    sources: list[VideoSource] = field(default_factory=list)
    audio_tracks: list[AudioTrack] = field(default_factory=list)
    captions: list[CaptionTrack] = field(default_factory=list)
    chapters: ChapterSet | None = None
    notes: list[DiscoveryNote] = field(default_factory=list)
    #: Source numbers probed by naming convention that could not be accessed, with the
    #: access state (e.g. ``{3: FORBIDDEN}``). Not sources: they may not exist.
    unreachable_sources: dict[int, AccessState] = field(default_factory=dict)
    discovered_at: float = field(default_factory=time.time)

    def source(self, index: int) -> VideoSource | None:
        return next((s for s in self.sources if s.index == index), None)

    def audio_track(self, track_id: str | None) -> AudioTrack | None:
        if track_id is None:
            return self.audio_tracks[0] if len(self.audio_tracks) == 1 else None
        return next((a for a in self.audio_tracks if a.id == track_id), None)

    @property
    def short_id(self) -> str:
        """Stable 10-hex-digit digest of the full lesson id, used in file names.

        Observed echo360.ca lesson ids look like ``G_<uuid>_<uuid>_<start>_<end>`` (123
        chars). A prefix may be shared by every lesson of a course section, so a prefix
        would collide; a digest of the whole id does not.
        """
        import hashlib

        return hashlib.sha256(self.lesson_id.encode("utf-8")).hexdigest()[:10]

    @property
    def caption_languages(self) -> list[str]:
        return sorted({c.language for c in self.captions if c.language})


# --------------------------------------------------------------------------------------
# Transcripts
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cue:
    text: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None


@dataclass
class Transcript:
    """Normalised transcript or subtitle cues with provenance.

    ``raw``/``raw_format`` keep the original bytes so a sidecar can preserve styling
    or information lost by conversion. ``generator`` names the model/backend for
    generated transcripts (e.g. "faster-whisper small (int8, cpu)").
    """

    language: str | None
    provenance: TranscriptProvenance
    cues: list[Cue] = field(default_factory=list)
    label: str | None = None
    source_detail: str | None = None
    generator: str | None = None
    raw: bytes | None = field(default=None, repr=False)
    raw_format: str | None = None

    @property
    def timed(self) -> bool:
        return bool(self.cues) and all(c.start is not None and c.end is not None
                                       for c in self.cues)

    @property
    def generated(self) -> bool:
        return self.provenance is TranscriptProvenance.GENERATED_LOCAL


@dataclass
class TranscriptBundle:
    """Result of gathering transcripts for a recording (platform and/or generated).

    ``missing_languages`` lists explicitly requested languages that were not found.
    ``absent`` is True when the platform published no transcript/caption at all.
    """

    transcripts: list[Transcript] = field(default_factory=list)
    missing_languages: list[str] = field(default_factory=list)
    absent: bool = False
    notes: list[str] = field(default_factory=list)
    #: Tracks that exist on the platform but could not be fetched (403, expired, network).
    #: These are *not* missing languages.
    failed: list[Failure] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Authentication context (secret-bearing)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cookie:
    """A cookie with its scope. ``domain`` starting with "." means subdomains match;
    otherwise the cookie is host-only. ``expires`` is a Unix time or None (session)."""

    name: str
    value: str = field(repr=False)
    domain: str
    path: str = "/"
    secure: bool = True
    http_only: bool = False
    expires: float | None = None

    @property
    def host_only(self) -> bool:
        return not self.domain.startswith(".")

    def matches(self, url: str) -> bool:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if not host:
            return False
        if self.secure and parts.scheme != "https":
            return False
        domain = self.domain.lower().lstrip(".")
        if self.host_only:
            if host != domain:
                return False
        elif not (host == domain or host.endswith("." + domain)):
            return False
        path = parts.path or "/"
        cookie_path = self.path or "/"
        if cookie_path == "/":
            return True
        return path == cookie_path or path.startswith(cookie_path.rstrip("/") + "/")

    def expired(self, now: float | None = None) -> bool:
        return self.expires is not None and self.expires <= (now or time.time())


#: Request headers that may be replayed. Everything else (authorization, cookie,
#: sec-*, ...) is dropped. Values are still treated as private and never logged.
REPLAYABLE_HEADERS = frozenset({"user-agent", "referer", "origin", "accept-language"})


@dataclass
class AuthContext:
    """Credentials for fetching content. Secret-bearing: never serialise or log.

    ``source`` is "browser" or "request-file". ``expires_at`` is the earliest known
    authorisation expiry (e.g. decoded from a CloudFront policy), if any.
    """

    cookies: list[Cookie] = field(default_factory=list, repr=False)
    headers: dict[str, str] = field(default_factory=dict, repr=False)
    source: str = "browser"
    obtained_at: float = field(default_factory=time.time)
    expires_at: float | None = None

    def __post_init__(self) -> None:
        self.headers = {k.lower(): v for k, v in self.headers.items()
                        if k.lower() in REPLAYABLE_HEADERS}

    def __repr__(self) -> str:  # never expose values
        return (f"AuthContext(source={self.source!r}, cookies={len(self.cookies)}, "
                f"expires_at={self.expires_at!r})")

    __str__ = __repr__

    def cookies_for(self, url: str, now: float | None = None) -> list[Cookie]:
        """Cookies that a browser would send to ``url`` (scope, secure, expiry)."""
        return [c for c in self.cookies if c.matches(url) and not c.expired(now)]

    def cookie_header(self, url: str) -> str | None:
        cookies = self.cookies_for(url)
        return "; ".join(f"{c.name}={c.value}" for c in cookies) if cookies else None

    def is_expired(self, now: float | None = None, margin: float = 30.0) -> bool:
        return self.expires_at is not None and self.expires_at - margin <= (now or time.time())

    def write_netscape_cookiejar(self, path: Path, *, for_url: str | None = None) -> Path:
        """Write a scope-preserving Netscape cookie file (mode 0600) for yt-dlp.

        With ``for_url`` only the cookies a browser would send to that URL are written,
        so e.g. echo360.ca session cookies never reach a content-host download.
        """
        rows = ["# Netscape HTTP Cookie File", "# Generated by echo360-dl. Contains secrets."]
        selected = self.cookies_for(for_url) if for_url else self.cookies
        for c in selected:
            if c.expired():
                continue
            if any(ch in c.name + c.value for ch in "\t\r\n"):
                continue
            domain = c.domain if not c.host_only else c.domain.lstrip(".")
            rows.append("\t".join([
                domain,
                "FALSE" if c.host_only else "TRUE",
                c.path or "/",
                "TRUE" if c.secure else "FALSE",
                str(int(c.expires)) if c.expires else "0",
                c.name,
                c.value,
            ]))
        path.parent.mkdir(parents=True, exist_ok=True)
        fd_path = path
        # Create with restrictive permissions before writing any secret.
        fd_path.touch(mode=0o600, exist_ok=True)
        fd_path.chmod(0o600)
        fd_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return fd_path


# --------------------------------------------------------------------------------------
# Discovery / authentication options
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveryOptions:
    """How to authenticate and how long discovery may take.

    * ``auth_mode="browser"``: use the dedicated persistent Playwright profile.
      Headless when the saved session works; a visible window is opened for login
      only when ``interactive`` is True. Non-interactive runs never open a window.
    * ``auth_mode="request-file"``: import a saved "Copy as cURL" request (parsed as
      data, never executed). Troubleshooting path; cannot see lesson metadata.
    """

    auth_mode: Literal["browser", "request-file"] = "browser"
    request_file: Path | None = None
    interactive: bool = True
    headless: bool | None = None
    timeout: float = 90.0  # bound for the whole discovery window (seconds)
    login_timeout: float = 300.0  # bound for an interactive login (seconds)
    profile_dir: Path | None = None  # default: paths.browser_profile_dir()
    save_trace: Path | None = None  # write sanitised traffic summary (no secrets)


@dataclass
class AuthStatus:
    profile_dir: Path
    profile_exists: bool
    logged_in: bool | None  # None -> unknown / not checked
    detail: str
    host: str = "echo360.ca"
    checked_at: float = field(default_factory=time.time)


# --------------------------------------------------------------------------------------
# Download request / result
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QualitySpec:
    """``best``, ``worst`` or a height cap ("720p" = best available with height <= 720)."""

    kind: Literal["best", "worst", "max_height"] = "best"
    max_height: int | None = None

    def __str__(self) -> str:
        return f"{self.max_height}p" if self.kind == "max_height" else self.kind


@dataclass(frozen=True)
class SubtitleMode:
    kind: SubtitleModeKind = SubtitleModeKind.AUTO
    languages: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChapterMode:
    kind: ChapterModeKind = ChapterModeKind.AUTO
    file: Path | None = None
    interval: float | None = None  # seconds, for INTERVAL


@dataclass(frozen=True)
class TranscribeOptions:
    enabled: bool = False
    backend: str = "auto"  # "auto" | "faster-whisper" | "mlx-whisper"
    model: str = "small"
    language: str | None = None  # None -> detect
    allow_model_download: bool = False
    only_if_missing: bool = True  # generate only when no platform transcript applies


@dataclass(frozen=True)
class DownloadRequest:
    recording: Recording
    output_dir: Path
    source_indexes: tuple[int, ...] = ()  # empty with audio_only=True -> audio only
    audio_only: bool = False
    quality: QualitySpec = QualitySpec()
    container: Container = Container.MP4
    subtitles: SubtitleMode = SubtitleMode()
    chapters: ChapterMode = ChapterMode()
    transcribe: TranscribeOptions = TranscribeOptions()
    if_exists: IfExists = IfExists.SKIP
    write_sidecars: bool = True  # transcript/subtitle/chapter sidecars next to media
    default_subtitle: str | None = None  # language whose track gets default disposition
    retries: int = 3
    concurrency: int = 2
    keep_work_dir: bool = False
    layout: Layout = Layout.SEPARATE
    #: Source order inside a composite (first = left/top). Empty -> selection order.
    layout_order: tuple[int, ...] = ()
    #: With a composite layout, also write each source as its own stream-copied file.
    keep_separate: bool = False
    encoder: Encoder = Encoder.AUTO


@dataclass
class StreamInfo:
    """One stream as reported by ffprobe (subset)."""

    index: int
    codec_type: str
    codec_name: str | None = None
    language: str | None = None
    title: str | None = None
    default: bool = False
    width: int | None = None
    height: int | None = None
    duration: float | None = None
    start_time: float | None = None


@dataclass
class VerifyReport:
    ok: bool
    duration: float | None = None
    streams: list[StreamInfo] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


@dataclass
class OutputFile:
    path: Path
    kind: OutputKind
    source_index: int | None = None  # None for audio-only / shared outputs
    language: str | None = None
    provenance: str | None = None  # TranscriptProvenance / ChapterProvenance value
    skipped_existing: bool = False
    verify: VerifyReport | None = None
    size: int | None = None


@dataclass
class Failure:
    step: str  # e.g. "download", "subtitles", "mux", "verify"
    target: str  # e.g. "Source 2", "captions:fr", "audio"
    kind: str  # Echo360DLError.kind
    message: str
    hint: str | None = None


@dataclass
class DownloadResult:
    status: ResultStatus
    outputs: list[OutputFile] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    lesson_id: str | None = None

    @property
    def exit_code(self) -> int:
        from echo360_dl.errors import ExitCode

        return {
            ResultStatus.OK: ExitCode.OK,
            ResultStatus.PARTIAL: ExitCode.PARTIAL,
            ResultStatus.CANCELLED: ExitCode.CANCELLED,
        }.get(self.status, _failure_exit_code(self.failures))


def _failure_exit_code(failures: Iterable[Failure]) -> int:
    from echo360_dl import errors

    by_kind = {cls.kind: cls.exit_code for cls in _all_error_classes(errors.Echo360DLError)}
    codes = {by_kind.get(f.kind, errors.ExitCode.ERROR) for f in failures}
    return int(codes.pop()) if len(codes) == 1 else int(errors.ExitCode.ERROR)


def _all_error_classes(base: type) -> list[type]:
    out = [base]
    for sub in base.__subclasses__():
        out.extend(_all_error_classes(sub))
    return out


# --------------------------------------------------------------------------------------
# Progress reporting and pluggable services
# --------------------------------------------------------------------------------------


@runtime_checkable
class ProgressSink(Protocol):
    """Implemented by the CLI. Media/discovery code reports *real* progress only:
    pass ``total=None`` when the size is unknown - never invent percentages."""

    def stage(self, stage: Stage, message: str | None = None) -> None: ...

    def task_start(self, task_id: str, description: str, total: float | None = None,
                   unit: Literal["bytes", "seconds", "items"] = "bytes") -> None: ...

    def task_update(self, task_id: str, *, completed: float | None = None,
                    total: float | None = None, rate: float | None = None,
                    eta: float | None = None) -> None: ...

    def task_done(self, task_id: str, *, ok: bool = True, message: str | None = None) -> None: ...

    def info(self, message: str) -> None: ...

    def warn(self, message: str) -> None: ...

    def debug(self, message: str) -> None: ...


class NullProgress:
    """ProgressSink that discards everything (tests, --json without progress)."""

    def stage(self, stage: Stage, message: str | None = None) -> None:
        pass

    def task_start(self, task_id: str, description: str, total: float | None = None,
                   unit: str = "bytes") -> None:
        pass

    def task_update(self, task_id: str, **kwargs: Any) -> None:
        pass

    def task_done(self, task_id: str, *, ok: bool = True, message: str | None = None) -> None:
        pass

    def info(self, message: str) -> None:
        pass

    def warn(self, message: str) -> None:
        pass

    def debug(self, message: str) -> None:
        pass


#: Fetch a platform caption/transcript resource. Implemented by the Echo360 session.
TranscriptFetcher = Callable[[CaptionTrack], Transcript]


class Transcriber(Protocol):
    """Local speech recognition (see :mod:`echo360_dl.transcribe`)."""

    def transcribe(self, audio_path: Path, *, language: str | None,
                   progress: ProgressSink) -> Transcript: ...


class ContentSession(Protocol):
    """Authenticated access used by the media pipeline. Implemented in echo360/."""

    @property
    def auth(self) -> AuthContext: ...

    def refresh(self) -> AuthContext:
        """Obtain fresh authorisation (bounded attempts). Raises AuthError."""
        ...

    def fetch_transcript(self, track: CaptionTrack, fmt: str | None = None) -> Transcript: ...

    def probe(self, url: str) -> AccessState: ...

    def close(self) -> None:
        """Release network resources. Optional: callers use ``getattr(s, "close", None)``."""
        ...


# --------------------------------------------------------------------------------------
# JSON serialisation
# --------------------------------------------------------------------------------------


def to_jsonable(obj: Any) -> Any:
    """Convert models to JSON-safe data. Strips query strings from URL fields,
    drops ``raw`` transcript bytes and refuses to serialise :class:`AuthContext`."""
    from echo360_dl.redact import public_url

    if isinstance(obj, AuthContext | Cookie):
        raise TypeError("AuthContext/Cookie must never be serialised")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            if f.name == "raw":
                continue
            value = getattr(obj, f.name)
            if isinstance(value, str) and f.name.endswith("url"):
                value = public_url(value)
            elif f.name == "formats" and isinstance(value, dict):
                value = sorted(value)  # list available format names, not URLs
            out[f.name] = to_jsonable(value)
        for prop in ("display_name", "timed", "resolution"):
            attr = getattr(type(obj), prop, None)
            if isinstance(attr, property):
                out[prop] = to_jsonable(getattr(obj, prop))
        return out
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, bytes):
        return None
    return obj
