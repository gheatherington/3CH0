from __future__ import annotations

import shutil
import threading

import pytest

from tests.media.conftest import (
    SIGNED_QUERY,
    URL_SECRETS,
    FakeSession,
    RecordingProgress,
    expired_refresh_error,
)
from threech0.errors import (
    AuthExpiredError,
    CancelledError,
    DependencyError,
    ForbiddenError,
)
from threech0.media import ffmpeg
from threech0.media.download import (
    AuthRefresher,
    Component,
    DownloadContext,
    WorkDir,
    download_component,
    playlist_total_bytes,
    resume_playlist,
)
from threech0.models import NullProgress

PLAYLIST = """#EXTM3U
#EXT-X-VERSION:7
#EXT-X-MAP:URI="s1q1.mp4",BYTERANGE="845@0"
#EXTINF:10.000000,
#EXT-X-BYTERANGE:15652@845
s1q1.mp4
#EXTINF:10.000000,
#EXT-X-BYTERANGE:1000@16497
s1q1.mp4
#EXT-X-ENDLIST
"""


def test_playlist_total_and_resume_rewrite():
    assert playlist_total_bytes(PLAYLIST) == 845 + 15652 + 1000
    assert playlist_total_bytes("#EXTM3U\n#EXTINF:1,\nseg1.mp4\n") is None
    rewritten = resume_playlist(PLAYLIST)
    assert "#EXT-X-MAP" not in rewritten
    assert rewritten.splitlines()[2:4] == ["#EXT-X-BYTERANGE:845@0", "s1q1.mp4"]
    assert playlist_total_bytes(rewritten) == playlist_total_bytes(PLAYLIST)


def test_require_tools_hint(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(DependencyError) as exc:
        ffmpeg.require_tools("ffmpeg", "ffprobe")
    assert "ffmpeg, ffprobe" in exc.value.message
    assert "brew install ffmpeg" in exc.value.hint
    assert ffmpeg.tool_versions()["ffmpeg"] is None


@pytest.mark.ffmpeg
def test_tool_versions_and_probe_report(hls_root, tmp_path):
    versions = ffmpeg.tool_versions()
    assert versions["ffmpeg"] and versions["ffprobe"] and versions["yt-dlp"]
    report = ffmpeg.probe_report(hls_root / "lesson" / "s1q1.mp4")
    assert report.ok and report.streams[0].codec_name == "h264"
    assert report.streams[0].start_time == pytest.approx(0.066016, abs=1e-6)
    audio = ffmpeg.probe_report(hls_root / "lesson" / "s0q1.mp4").streams[0]
    assert audio.codec_type == "audio" and audio.start_time == 0.0
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not media")
    bad = ffmpeg.probe_report(junk)
    assert not bad.ok and bad.problems
    assert not ffmpeg.probe_report(tmp_path / "missing.mp4").ok


def _component(server, name="s1q1", kind="video"):
    return Component(f"{kind}-{name}", kind, name,
                     f"{server.base_url}/lesson/{name}.m3u8{SIGNED_QUERY}", f"{name} {kind}")


def _ctx(tmp_path, session, progress=None, cancel=None, retries=2):
    return DownloadContext(WorkDir(tmp_path / "work", "lesson"), AuthRefresher(session),
                           progress or NullProgress(), cancel or threading.Event(), retries)


@pytest.mark.ffmpeg
def test_download_is_byte_identical_with_real_progress(server, hls_root, tmp_path):
    session = FakeSession(server)
    progress = RecordingProgress()
    ctx = _ctx(tmp_path, session, progress)
    path = download_component(_component(server), ctx)
    original = (hls_root / "lesson" / "s1q1.mp4").read_bytes()
    assert path.read_bytes() == original
    totals = {u.get("total") for _, u in progress.updates if u.get("total")}
    assert totals == {len(original)}  # exact total from the byte-range playlist
    assert not list(ctx.work.path.glob(".cookies-*"))
    headers = server.state.headers_seen
    assert all(h.get("user-agent") == "3ch0-test/1.0" for h in headers)
    assert all("SHOULD-NOT-BE-SENT" not in h.get("cookie", "") for h in headers)
    assert all("authorization" not in h for h in headers)
    # A second call reuses the completed component without any request.
    server.reset_log()
    assert download_component(_component(server), ctx) == path
    assert sum(server.state.requests.values()) == 0


@pytest.mark.ffmpeg
def test_cancel_then_resume_is_byte_identical(server, hls_root, tmp_path):
    session = FakeSession(server)
    cancel = threading.Event()
    progress = RecordingProgress()
    progress.on_update = lambda task, u: (u.get("completed") or 0) > 900 and cancel.set()
    server.state.throttle = 0.05
    ctx = _ctx(tmp_path, session, progress, cancel)
    with pytest.raises(CancelledError):
        download_component(_component(server), ctx)
    work = ctx.work.path
    assert (work / "video-s1q1.mp4.part").exists() and not (work / "video-s1q1.mp4").exists()
    path = download_component(_component(server), _ctx(tmp_path, session))
    assert path.read_bytes() == (hls_root / "lesson" / "s1q1.mp4").read_bytes()


@pytest.mark.ffmpeg
def test_expiry_mid_download_refreshes_once_and_resumes(server, hls_root, tmp_path):
    session = FakeSession(server)
    server.state.expire_after = 2
    path = download_component(_component(server), _ctx(tmp_path, session))
    assert session.refresh_calls == 1
    assert path.read_bytes() == (hls_root / "lesson" / "s1q1.mp4").read_bytes()


@pytest.mark.ffmpeg
def test_transient_failures_are_retried(server, hls_root, tmp_path):
    server.state.fail_next = {"/lesson/s0q1.m3u8": 2, "/lesson/s0q1.mp4": 2}
    path = download_component(_component(server, "s0q1", "audio"),
                              _ctx(tmp_path, FakeSession(server), retries=3))
    assert path.read_bytes() == (hls_root / "lesson" / "s0q1.mp4").read_bytes()


@pytest.mark.ffmpeg
def test_refresh_failure_is_an_auth_error_without_secrets(server, tmp_path):
    server.expire()
    session = FakeSession(server, refresh_error=expired_refresh_error())
    ctx = _ctx(tmp_path, session)
    with pytest.raises(AuthExpiredError) as exc:
        download_component(_component(server), ctx)
    assert session.refresh_calls == 1
    assert not list(ctx.work.path.glob(".cookies-*"))
    for secret in URL_SECRETS:
        assert secret not in str(exc.value)


@pytest.mark.ffmpeg
def test_persistent_403_after_refresh_is_forbidden(server, tmp_path):
    server.state.forbid = {"/lesson/s2"}
    session = FakeSession(server)
    with pytest.raises(ForbiddenError) as exc:
        download_component(_component(server, "s2q1"), _ctx(tmp_path, session))
    assert "403" in exc.value.message and "403" in exc.value.hint
    assert session.refresh_calls == 1
