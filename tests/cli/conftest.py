"""Fakes for CLI tests: sanitised, fictional recordings and a recording Services double.

Secret-looking values are planted in URLs, the AuthContext and debug messages so tests
can assert that none of them ever reaches stdout or stderr.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from threech0.cli import output as cli_output
from threech0.cli.app import app
from threech0.errors import LoginRequiredError, ValidationError
from threech0.models import (
    AccessState,
    AudioRendition,
    AudioTrack,
    AuthContext,
    AuthStatus,
    CaptionTrack,
    Chapter,
    ChapterProvenance,
    ChapterSet,
    ClassroomRef,
    Cookie,
    Cue,
    DiscoveryMethod,
    DiscoveryNote,
    DiscoveryOptions,
    DownloadRequest,
    DownloadResult,
    OutputFile,
    OutputKind,
    ProgressSink,
    Recording,
    ResultStatus,
    Stage,
    StreamInfo,
    SubtitleMode,
    TranscribeOptions,
    Transcript,
    TranscriptBundle,
    TranscriptProvenance,
    VerifyReport,
    VideoRendition,
    VideoSource,
)
from threech0.services import Services, set_services

SECRET_SIG = "SeCrEtSiGnAtUrE9876"
SECRET_UID = "UidSecret424242"
SECRET_COOKIE = "CookieSecretValue1357"
SECRET_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmYWtlLXN0dWRlbnQifQ.c2lnbmF0dXJlLWZha2U"
MODEL_MESSAGE = "Speech model 'small' (about 460 MB) is not downloaded yet."
SECRETS = [SECRET_SIG, SECRET_UID, SECRET_COOKIE, SECRET_JWT]

LESSON = "G_fake-lesson-0001"
CLASSROOM = f"https://echo360.ca/lesson/{LESSON}/classroom"
URL = CLASSROOM + "?timestamp=123"
CONTENT = "https://content.echo360.ca/0000.inst-fake/media-fake-0001/1"
QUERY = f"?x-uid={SECRET_UID}&x-lid={LESSON}&Signature={SECRET_SIG}"


def _rendition(source: int, q: int, height: int, width: int, bw: int) -> VideoRendition:
    return VideoRendition(id=f"s{source}q{q}", url=f"{CONTENT}/s{source}q{q}.m3u8{QUERY}",
                          width=width, height=height, bandwidth=bw,
                          codecs="avc1.640029,mp4a.40.2", frame_rate=30.0,
                          audio_group=f"q{q}")


def _source(index: int, access: AccessState = AccessState.AVAILABLE) -> VideoSource:
    return VideoSource(
        index=index, playlist_url=f"{CONTENT}/s{index}_av.m3u8{QUERY}",
        renditions=[_rendition(index, 0, 360, 640, 93350 * index),
                    _rendition(index, 1, 720, 1280, 272955 * index)],
        method=DiscoveryMethod.OBSERVED_REQUEST, access=access,
        access_detail="HTTP 403" if access is AccessState.FORBIDDEN else None,
        audio_track_id="s0")


def _audio(sources: list[int]) -> AudioTrack:
    return AudioTrack(id="s0", renditions=[
        AudioRendition(id="s0q0", url=f"{CONTENT}/s0q0.m3u8{QUERY}", group_id="q0",
                       name="Default", default=True),
        AudioRendition(id="s0q1", url=f"{CONTENT}/s0q1.m3u8{QUERY}", group_id="q1",
                       name="Default", default=True),
    ], shared=len(sources) > 1, used_by_sources=sources)


def make_recording(*, sources: int = 2, forbidden_second: bool = True, captions: bool = True,
                   chapters: bool = True) -> Recording:
    """Fictional lesson. ``sources=2, forbidden_second=True`` mirrors the observed shape
    with source 2 returning 403."""
    srcs = [_source(1)]
    if sources >= 2:
        srcs.append(_source(2, AccessState.FORBIDDEN if forbidden_second
                            else AccessState.AVAILABLE))
    rec = Recording(
        lesson_id=LESSON, classroom_url=CLASSROOM, host="echo360.ca",
        media_id="media-fake-0001", title="Lecture 7: Thermodynamics [draft]",
        course="PHYS 201", recorded_at="2026-10-14T10:30:00", duration=3240.0,
        sources=srcs, audio_tracks=[_audio([s.index for s in srcs])],
        notes=[DiscoveryNote("info", "Stream URLs observed in player traffic."),
               DiscoveryNote("info", "Source labels may be camera and screen.", observed=False)],
    )
    if captions:
        rec.captions = [CaptionTrack(
            id="cap-en", language="en", kind=TranscriptProvenance.PLATFORM_CAPTIONS,
            formats={"vtt": f"https://echo360.ca/api/captions/en.vtt{QUERY}"},
            timed=True, access=AccessState.AVAILABLE)]
    if chapters:
        rec.chapters = ChapterSet([Chapter(0.0, "Introduction"), Chapter(510.0, "Entropy"),
                                   Chapter(1800.0, "Worked example")], ChapterProvenance.NATIVE)
    return rec


class FakeSession:
    def __init__(self) -> None:
        self._auth = AuthContext(
            cookies=[Cookie("CloudFront-Signature", SECRET_COOKIE, "content.echo360.ca"),
                     Cookie("ECHO_JWT", SECRET_JWT, ".echo360.ca")],
            headers={"user-agent": "test-agent", "authorization": f"Bearer {SECRET_JWT}"})

    @property
    def auth(self) -> AuthContext:
        return self._auth

    def refresh(self) -> AuthContext:
        return self._auth

    def fetch_transcript(self, track: CaptionTrack, fmt: str | None = None) -> Transcript:
        raise AssertionError("CLI must not fetch transcripts directly")

    def probe(self, url: str) -> AccessState:
        return AccessState.AVAILABLE


def ok_result(request: DownloadRequest) -> DownloadResult:
    outputs: list[OutputFile] = []
    for index in request.source_indexes:
        path = request.output_dir / f"Lecture-7-{request.recording.short_id}-s{index}.mp4"
        outputs.append(OutputFile(
            path=path, kind=OutputKind.VIDEO, source_index=index, size=27_000_000,
            verify=VerifyReport(ok=True, duration=3240.0, streams=[
                StreamInfo(0, "video", "h264", width=1280, height=720),
                StreamInfo(1, "audio", "aac"),
                StreamInfo(2, "subtitle", "mov_text", language="eng")],
                chapters=[Chapter(0, "Introduction"), Chapter(510, "Entropy"),
                          Chapter(1800, "Worked example")])))
    if request.audio_only:
        outputs.append(OutputFile(path=request.output_dir / "Lecture-7-audio.m4a",
                                  kind=OutputKind.AUDIO, size=3_000_000))
    return DownloadResult(ResultStatus.OK, outputs=outputs, lesson_id=LESSON)


@dataclass
class FakeServices(Services):
    recording: Recording = field(default_factory=make_recording)
    login_required: bool = False
    discover_error: Exception | None = None
    download_result: DownloadResult | Callable[[DownloadRequest], DownloadResult] | None = None
    download_error: BaseException | None = None
    bundle: TranscriptBundle | None = None
    versions: dict[str, str | None] = field(default_factory=lambda: {
        "ffmpeg": "9.0", "ffprobe": "9.0", "yt-dlp": "2026.10.26"})
    versions_error: Exception | None = None
    browser: tuple[bool, str] = (True, "chromium 140 installed")
    backends: list[Any] | Exception = field(default_factory=list)
    auth: AuthStatus | None = None
    probe_report: VerifyReport | None = None
    # recorded calls
    discover_calls: list[tuple[ClassroomRef, DiscoveryOptions]] = field(default_factory=list)
    requests: list[DownloadRequest] = field(default_factory=list)
    transcript_calls: list[dict[str, Any]] = field(default_factory=list)
    login_calls: list[Path] = field(default_factory=list)
    confirms: list[Callable[[str], bool] | None] = field(default_factory=list)
    cancels: list[threading.Event | None] = field(default_factory=list)
    confirm_answers: list[bool] = field(default_factory=list)

    def parse_url(self, url: str) -> ClassroomRef:
        from threech0.errors import InvalidURLError

        if "/lesson/" not in url:
            raise InvalidURLError(f"Not an Echo360 classroom URL: {url.split('?')[0]}")
        return ClassroomRef(host="echo360.ca", lesson_id=LESSON, url=CLASSROOM)

    def discover(self, ref: ClassroomRef, options: DiscoveryOptions,
                 progress: ProgressSink) -> tuple[Recording, FakeSession]:
        self.discover_calls.append((ref, options))
        progress.stage(Stage.DISCOVER, "probing")
        progress.debug(f"GET {CONTENT}/s1_av.m3u8{QUERY} Cookie: ECHO_JWT={SECRET_JWT}")
        progress.task_start("meta", "lesson metadata", total=None, unit="items")
        progress.task_done("meta")
        if self.discover_error is not None:
            raise self.discover_error
        if self.login_required:
            raise LoginRequiredError("No saved Echo360 session.")
        return self.recording, FakeSession()

    def login(self, profile_dir: Path, *, start_url: str | None, timeout: float,
              progress: ProgressSink) -> AuthStatus:
        self.login_calls.append(profile_dir)
        return AuthStatus(profile_dir, True, True, "signed in")

    def auth_status(self, profile_dir: Path, *, timeout: float = 30.0) -> AuthStatus:
        return self.auth or AuthStatus(profile_dir, profile_dir.exists(), True, "session valid")

    def download(self, request: DownloadRequest, session: Any, *, progress: ProgressSink,
                 confirm_download: Callable[[str], bool] | None,
                 cancel: threading.Event | None = None) -> DownloadResult:
        self.requests.append(request)
        self.confirms.append(confirm_download)
        self.cancels.append(cancel)
        if request.transcribe.enabled and confirm_download is not None:
            self.confirm_answers.append(confirm_download(MODEL_MESSAGE))
        progress.stage(Stage.DOWNLOAD)
        progress.task_start("s1", "Source 1 video", total=27_000_000)
        progress.task_update("s1", completed=13_500_000, rate=2_000_000.0)
        progress.task_done("s1")
        progress.stage(Stage.VERIFY)
        if self.download_error is not None:
            raise self.download_error
        if callable(self.download_result):
            return self.download_result(request)
        return self.download_result or ok_result(request)

    def transcripts(self, recording: Recording, session: Any, *, subtitles: SubtitleMode,
                    transcribe: TranscribeOptions, work_dir: Path, progress: ProgressSink,
                    confirm_download: Callable[[str], bool] | None,
                    cancel: threading.Event | None = None) -> TranscriptBundle:
        self.transcript_calls.append({"subtitles": subtitles, "transcribe": transcribe,
                                      "work_dir": work_dir, "confirm": confirm_download})
        if self.bundle is not None:
            return self.bundle
        return TranscriptBundle(transcripts=[Transcript(
            language="en", provenance=TranscriptProvenance.PLATFORM_CAPTIONS,
            cues=[Cue("Welcome back.", 0.0, 2.5), Cue("Today: entropy.", 2.5, 5.0)])])

    def load_chapters(self, path: Path) -> ChapterSet:
        try:
            data = json.loads(path.read_text())
        except ValueError as exc:
            raise ValidationError(f"{path.name}: not valid JSON ({exc.msg}).") from None
        from threech0.timecode import parse_time

        return ChapterSet([Chapter(parse_time(c["start"]), c["title"],
                                   parse_time(c["end"]) if "end" in c else None)
                           for c in data["chapters"]], ChapterProvenance.USER_FILE, path.name)

    def validate_chapters(self, chapters: ChapterSet, duration: float | None) -> list[str]:
        problems: list[str] = []
        starts = [c.start for c in chapters.chapters]
        if starts != sorted(set(starts)):
            problems.append("chapter starts must be strictly increasing")
        if duration is not None and any(s >= duration for s in starts):
            problems.append("a chapter starts after the end of the media")
        if any(not c.title.strip() for c in chapters.chapters):
            problems.append("chapter titles must not be empty")
        return problems

    def probe(self, path: Path) -> VerifyReport:
        return self.probe_report or VerifyReport(ok=True, duration=3240.0)

    def tool_versions(self) -> dict[str, str | None]:
        if self.versions_error is not None:
            raise self.versions_error
        return self.versions

    def browser_installed(self) -> tuple[bool, str]:
        return self.browser

    def transcription_backends(self) -> list[Any]:
        if isinstance(self.backends, Exception):
            raise self.backends
        return self.backends


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeServices]:
    monkeypatch.setenv("THREECH0_HOME", str(tmp_path / "home"))
    services = FakeServices()
    set_services(services)
    yield services
    set_services(None)


@pytest.fixture
def tty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend stdin and stderr are terminals (interactive session)."""
    monkeypatch.setattr(cli_output, "isatty", lambda stream: True)


BASE_ENV = {"COLUMNS": "100", "NO_COLOR": None, "FORCE_COLOR": None, "TTY_COMPATIBLE": None,
            "TTY_INTERACTIVE": None, "TERM": "xterm-256color"}


def invoke(args: list[str], *, input: str | None = None, env: dict[str, str | None] | None = None):
    runner = CliRunner()
    merged = dict(BASE_ENV)
    merged.update(env or {})
    return runner.invoke(app, args, input=input, env=merged)


def assert_clean(text: str) -> None:
    """No escape codes and no planted secrets."""
    assert "\x1b" not in text
    for secret in SECRETS:
        assert secret not in text, f"secret leaked: {secret[:6]}..."
