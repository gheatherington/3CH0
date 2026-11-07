"""Media test fixtures: synthetic Echo360-shaped HLS, a local server, fake session.

Fixture media is generated once per session with ffmpeg lavfi (testsrc/sine, 6 s,
H.264/AAC) and packaged like the observed Echo360 content: fMP4, single file with
byte ranges (``-hls_segment_type fmp4 -hls_flags single_file``), separate video-only
(``s1q*``, ``s2q*``) and audio-only (``s0q*``) media playlists, and master playlists
``s1_av``/``s2_av`` that share the ``s0`` audio. Source 1 is encoded with B-frames,
which reproduces the observed video ``start_time=0.066016`` (audio 0.0); source 2 has
no B-frames (start 0.0), giving a second, different start offset.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from echo360_dl.errors import AuthExpiredError
from echo360_dl.media.subtitles import parse_subtitle_bytes
from echo360_dl.models import (
    AccessState,
    AudioRendition,
    AudioTrack,
    AuthContext,
    CaptionTrack,
    ChapterSet,
    Cookie,
    Cue,
    Recording,
    Stage,
    Transcript,
    TranscriptProvenance,
    VideoRendition,
    VideoSource,
)
from tests.media.hls_server import COOKIE_NAME, HLSServer

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "media"

#: Secret-looking values that must never appear in results, errors or files.
URL_SECRETS = ("SECRETUID123", "SECRETSID456", "SIGSECRET789")
JWT_SECRET = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzZWNyZXQifQ.c2lnbmF0dXJlc2VjcmV0"
FORBIDDEN_HEADER_VALUES = ("SHOULD-NOT-BE-SENT", "Bearer nope")
SIGNED_QUERY = "?x-uid=SECRETUID123&x-sid=SECRETSID456&Signature=SIGSECRET789"
DURATION = 6
LESSON_ID = "G_8f1c2a9e-1234-5678-9abc-def012345678"

MASTER = """#EXTM3U
#EXT-X-VERSION:7
#EXT-X-INDEPENDENT-SEGMENTS

#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="q0",NAME="Default",DEFAULT=YES,AUTOSELECT=YES,URI="s0q0.m3u8"
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="q1",NAME="Default",DEFAULT=YES,AUTOSELECT=YES,URI="s0q1.m3u8"

#EXT-X-STREAM-INF:BANDWIDTH=93350,RESOLUTION=160x90,CODECS="avc1.640029",AUDIO="q0",CODECS="avc1.640029,mp4a.40.2",FRAME-RATE=30.0
s{n}q0.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=272955,RESOLUTION=320x180,CODECS="avc1.640029",AUDIO="q1",CODECS="avc1.640029,mp4a.40.2",FRAME-RATE=30.0
s{n}q1.m3u8
"""


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def _hls(out: Path, name: str, args: list[str]) -> None:
    _ffmpeg(*args, "-f", "hls", "-hls_time", "2", "-hls_playlist_type", "vod",
            "-hls_segment_type", "fmp4", "-hls_flags", "single_file",
            "-hls_segment_filename", str(out / f"{name}.mp4"), str(out / f"{name}.m3u8"))


def build_fixture_tree(root: Path) -> Path:
    lesson = root / "lesson"
    lesson.mkdir(parents=True, exist_ok=True)
    for n, pattern, bframes in ((1, "testsrc", "2"), (2, "testsrc2", "0")):
        for q, size in (("q0", "160x90"), ("q1", "320x180")):
            _hls(lesson, f"s{n}{q}", [
                "-f", "lavfi", "-i", f"{pattern}=size={size}:rate=30:duration={DURATION}",
                "-c:v", "libx264", "-preset", "ultrafast", "-bf", bframes, "-g", "30",
                "-pix_fmt", "yuv420p", "-an"])
        (lesson / f"s{n}_av.m3u8").write_text(MASTER.replace("{n}", str(n)), "utf-8")
    for q, rate in (("q0", "48k"), ("q1", "128k")):
        _hls(lesson, f"s0{q}", [
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={DURATION}:sample_rate=44100",
            "-c:a", "aac", "-b:a", rate])
    return root


@pytest.fixture(scope="session")
def hls_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not installed")
    return build_fixture_tree(tmp_path_factory.mktemp("echo360-hls"))


@pytest.fixture
def server(hls_root: Path):
    srv = HLSServer(hls_root).start()
    try:
        yield srv
    finally:
        srv.stop()


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    from echo360_dl.media import download

    monkeypatch.setattr(download, "RETRY_BACKOFF", 0.01)


# --------------------------------------------------------------------------------------
# Recording / session fakes
# --------------------------------------------------------------------------------------


def make_recording(base_url: str, *, sources: tuple[int, ...] = (1, 2),
                   captions: list[CaptionTrack] | None = None,
                   chapters: ChapterSet | None = None,
                   title: str | None = "Limits: Part 1",
                   course: str | None = "MATH 101") -> Recording:
    def url(name: str) -> str:
        return f"{base_url}/lesson/{name}.m3u8{SIGNED_QUERY}"

    audio = AudioTrack(id="s0", shared=len(sources) > 1, used_by_sources=list(sources),
                       renditions=[AudioRendition(id="s0q0", url=url("s0q0"), group_id="q0"),
                                   AudioRendition(id="s0q1", url=url("s0q1"), group_id="q1")])
    video_sources = [
        VideoSource(index=n, playlist_url=url(f"s{n}_av"), audio_track_id="s0", renditions=[
            VideoRendition(id=f"s{n}q0", url=url(f"s{n}q0"), width=160, height=90,
                           bandwidth=93350, codecs="avc1.640029", audio_group="q0"),
            VideoRendition(id=f"s{n}q1", url=url(f"s{n}q1"), width=320, height=180,
                           bandwidth=272955, codecs="avc1.640029", audio_group="q1"),
        ]) for n in sources]
    return Recording(lesson_id=LESSON_ID,
                     classroom_url=f"https://echo360.ca/lesson/{LESSON_ID}/classroom",
                     host="echo360.ca", title=title, course=course,
                     recorded_at="2026-11-05T14:00:00Z", duration=float(DURATION),
                     sources=video_sources, audio_tracks=[audio], captions=captions or [],
                     chapters=chapters)


def caption(track_id: str, language: str, fmt: str,
            kind: TranscriptProvenance = TranscriptProvenance.PLATFORM_CAPTIONS,
            timed: bool | None = True) -> CaptionTrack:
    return CaptionTrack(id=track_id, language=language, kind=kind,
                        formats={fmt: f"https://echo360.ca/api/{track_id}.{fmt}{SIGNED_QUERY}"},
                        timed=timed)


class FakeSession:
    """ContentSession double. ``refresh()`` obtains the server's next token."""

    def __init__(self, srv: HLSServer, transcripts: dict[str, bytes] | None = None,
                 refresh_error: Exception | None = None) -> None:
        self.server = srv
        self.transcripts = transcripts or {}
        self.refresh_error = refresh_error
        self.refresh_calls = 0
        self.fetch_calls: list[tuple[str, str | None]] = []
        self._auth = self._make(srv.state.token)

    @staticmethod
    def _make(token: str) -> AuthContext:
        return AuthContext(
            cookies=[Cookie(name=COOKIE_NAME, value=token, domain="127.0.0.1", secure=False),
                     Cookie(name="ECHO_JWT", value=JWT_SECRET, domain="echo360.ca")],
            headers={"User-Agent": "echo360-dl-test/1.0",
                     "Referer": "https://echo360.ca/lesson/x/classroom",
                     "Cookie": "SHOULD-NOT-BE-SENT=1", "Authorization": "Bearer nope"},
            source="browser")

    @property
    def auth(self) -> AuthContext:
        return self._auth

    def refresh(self) -> AuthContext:
        self.refresh_calls += 1
        if self.refresh_error is not None:
            raise self.refresh_error
        self._auth = self._make(self.server.issue_refreshed_token())
        return self._auth

    def fetch_transcript(self, track: CaptionTrack, fmt: str | None = None) -> Transcript:
        self.fetch_calls.append((track.id, fmt))
        data = self.transcripts[track.id]
        cues = parse_subtitle_bytes(data, fmt)
        return Transcript(language=track.language, provenance=track.kind, cues=cues,
                          label=track.label, raw=data, raw_format=fmt)

    def probe(self, url: str) -> AccessState:
        return AccessState.AVAILABLE


def expired_refresh_error() -> AuthExpiredError:
    return AuthExpiredError(f"Session expired (cookie {COOKIE_NAME}=tok-SECRETVALUE9)",
                            hint="Run `echo360-dl auth login`.")


class FakeTranscriber:
    def __init__(self, language: str = "en") -> None:
        self.language = language
        self.calls: list[Path] = []

    def transcribe(self, audio_path: Path, *, language: str | None, progress: Any) -> Transcript:
        assert audio_path.is_file()
        self.calls.append(audio_path)
        return Transcript(language=language or self.language,
                          provenance=TranscriptProvenance.GENERATED_LOCAL,
                          cues=[Cue("generated one", 0.5, 1.5), Cue("generated two", 2.0, 3.0)],
                          generator="fake-whisper tiny")


@dataclass
class RecordingProgress:
    """ProgressSink that records everything (thread-safe)."""

    stages: list[Stage] = field(default_factory=list)
    updates: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    starts: list[tuple[str, str]] = field(default_factory=list)
    done: list[tuple[str, bool, str | None]] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    on_update: Any = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def stage(self, stage: Stage, message: str | None = None) -> None:
        self.stages.append(stage)

    def task_start(self, task_id: str, description: str, total: float | None = None,
                   unit: str = "bytes") -> None:
        with self.lock:
            self.starts.append((task_id, description))
            self.messages.append(description)

    def task_update(self, task_id: str, **kwargs: Any) -> None:
        with self.lock:
            self.updates.append((task_id, kwargs))
        if self.on_update:
            self.on_update(task_id, kwargs)

    def task_done(self, task_id: str, *, ok: bool = True, message: str | None = None) -> None:
        with self.lock:
            self.done.append((task_id, ok, message))
            self.messages.append(message or "")

    def info(self, message: str) -> None:
        self.messages.append(message)

    def warn(self, message: str) -> None:
        self.messages.append(message)

    def debug(self, message: str) -> None:
        self.messages.append(message)
