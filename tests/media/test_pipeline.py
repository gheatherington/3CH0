"""End-to-end pipeline tests: real ffmpeg/ffprobe, synthetic HLS over localhost."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

import pytest

from echo360_dl.media import mux as mux_module
from echo360_dl.media import pipeline
from echo360_dl.media.ffmpeg import count_packets, ffprobe_json
from echo360_dl.media.pipeline import obtain_transcripts, run_download, work_dir_for
from echo360_dl.models import (
    Chapter,
    ChapterMode,
    ChapterModeKind,
    ChapterProvenance,
    ChapterSet,
    Container,
    DownloadRequest,
    IfExists,
    NullProgress,
    OutputKind,
    QualitySpec,
    ResultStatus,
    SubtitleMode,
    SubtitleModeKind,
    TranscribeOptions,
    TranscriptProvenance,
    to_jsonable,
)
from tests.media.conftest import (
    FIXTURES,
    FORBIDDEN_HEADER_VALUES,
    JWT_SECRET,
    URL_SECRETS,
    FakeSession,
    FakeTranscriber,
    RecordingProgress,
    caption,
    expired_refresh_error,
    make_recording,
)

pytestmark = pytest.mark.ffmpeg

CHAPTERS = ChapterSet([Chapter(0.0, "Intro = ; # \\ é"), Chapter(2.5, "Second — 第二"),
                       Chapter(4.0, "Wrap\\up; #3")], ChapterProvenance.NATIVE, "metadata")
TRANSCRIPTS = {
    "en": (FIXTURES / "captions_en.vtt").read_bytes(),
    "fr": (FIXTURES / "captions_fr.srt").read_bytes(),
    "en-txt": (FIXTURES / "transcript_untimed.txt").read_bytes(),
}


def _captions():
    return [caption("en", "en", "vtt"), caption("fr", "fr", "srt")]


def _request(rec, tmp_path, **kw) -> DownloadRequest:
    kw.setdefault("output_dir", tmp_path / "out")
    return DownloadRequest(recording=rec, **kw)


def _probe(path: Path) -> dict:
    return ffprobe_json(path)


def _streams(data: dict, kind: str) -> list[dict]:
    return [s for s in data["streams"] if s["codec_type"] == kind]


def _first_cue_time(path: Path, index: int) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", f"s:{index}",
                          "-show_entries", "packet=pts_time,size", "-of", "json", str(path)],
                         capture_output=True, check=True)
    packets = json.loads(out.stdout)["packets"]
    return next(float(p["pts_time"]) for p in packets if int(p["size"]) > 2)


def _no_secrets(result, progress: RecordingProgress | None, out_dir: Path, server) -> None:
    dumped = json.dumps(to_jsonable(result), ensure_ascii=False)
    texts = [dumped] + (progress.messages if progress else [])
    secrets = [*URL_SECRETS, JWT_SECRET, server.state.token, "tok-initial-SECRETVALUE1",
               "SECRETVALUE"]
    for text in texts:
        for secret in secrets:
            assert secret not in text, (secret, text[:200])
    for path in out_dir.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".txt", ".vtt", ".srt"}:
            content = path.read_text("utf-8", "replace")
            for secret in secrets:
                assert secret not in content, path
        assert not path.name.startswith(".cookies-"), path
    for headers in server.state.headers_seen:
        for bad in FORBIDDEN_HEADER_VALUES:
            assert bad not in " ".join(headers.values())


def _video_outputs(result):
    return [o for o in result.outputs if o.kind is OutputKind.VIDEO]


# --------------------------------------------------------------------------------------


def test_one_source_mp4_subtitles_chapters_sync(server, hls_root, tmp_path):
    rec = make_recording(server.base_url, sources=(1,), captions=_captions(), chapters=CHAPTERS)
    session = FakeSession(server, TRANSCRIPTS)
    progress = RecordingProgress()
    result = run_download(_request(rec, tmp_path, source_indexes=(1,)), session,
                          progress=progress, transcriber_factory=None, cancel=None)
    assert result.status is ResultStatus.OK, result.failures
    assert progress.stages == ["select", "download", "subtitles", "chapters", "verify"]
    (video,) = _video_outputs(result)
    assert video.path.name == (f"MATH 101 - 2026-11-05 - Limits - Part 1 [{rec.short_id}]"
                               " - Source 1.mp4")
    assert video.size == video.path.stat().st_size and video.verify.ok

    data = _probe(video.path)
    v, a, subs = _streams(data, "video"), _streams(data, "audio"), _streams(data, "subtitle")
    assert [s["codec_name"] for s in v + a + subs] == ["h264", "aac", "mov_text", "mov_text"]
    assert [s["tags"]["language"] for s in subs] == ["eng", "fre"]
    assert [s["tags"]["handler_name"] for s in subs] == ["English — Echo360 captions",
                                                         "French — Echo360 captions"]
    assert sum(s["disposition"]["default"] for s in subs) <= 1
    # Video was stream-copied: same codec, size and packet count as the source.
    src = hls_root / "lesson" / "s1q1.mp4"
    assert (v[0]["width"], v[0]["height"]) == (320, 180)
    assert count_packets(video.path) == count_packets(src) == 180
    # Timing: original start offsets preserved (video 0.066016, audio 0), cues unshifted.
    assert float(v[0]["start_time"]) == pytest.approx(0.066016, abs=1e-6)
    assert float(a[0]["start_time"]) == 0.0
    assert _first_cue_time(video.path, 0) == pytest.approx(1.0, abs=0.001)
    assert _first_cue_time(video.path, 1) == pytest.approx(1.2, abs=0.001)
    # Chapters embedded with exact titles (escaping) and boundaries.
    chapters = [(float(c["start_time"]), float(c["end_time"]), c["tags"]["title"])
                for c in data["chapters"]]
    assert [c[2] for c in chapters] == [c.title for c in CHAPTERS.chapters]
    assert [round(c[0], 3) for c in chapters] == [0.0, 2.5, 4.0]
    assert [round(c[1], 3) for c in chapters[:2]] == [2.5, 4.0]
    assert data["format"]["tags"]["title"] == "Limits: Part 1"

    names = sorted(o.path.name.split("] ")[-1].split("]")[-1] for o in result.outputs
                   if o.kind is not OutputKind.VIDEO)
    assert names == [".chapters.json", ".en.platform.vtt", ".en.srt", ".en.txt", ".en.vtt",
                     ".fr.srt", ".fr.txt", ".fr.vtt"]
    out_dir = tmp_path / "out"
    styled = next(out_dir.glob("*.en.platform.vtt"))
    assert styled.read_bytes() == TRANSCRIPTS["en"]  # original kept (styling would be lost)
    assert "Dr. Alice Smith: Welcome & hello — café" in next(out_dir.glob("*.en.txt")).read_text()
    assert "<v Dr. Alice Smith>" in next(out_dir.glob("*].en.vtt")).read_text()
    side = json.loads(next(out_dir.glob("*.chapters.json")).read_text())
    assert side["provenance"] == "native" and side["chapters"][0]["title"] == "Intro = ; # \\ é"
    assert not work_dir_for(out_dir, rec).exists()  # removed after full success
    assert not list(out_dir.glob(".*echo360-dl-tmp*"))
    _no_secrets(result, progress, out_dir, server)


def test_two_sources_mkv_shared_audio_downloaded_once(server, tmp_path):
    rec = make_recording(server.base_url, captions=_captions(), chapters=CHAPTERS)
    session = FakeSession(server, TRANSCRIPTS)
    result = run_download(_request(rec, tmp_path, container=Container.MKV, default_subtitle="fr",
                                   concurrency=3), session, progress=NullProgress())
    assert result.status is ResultStatus.OK, result.failures
    videos = _video_outputs(result)
    assert [o.source_index for o in videos] == [1, 2]
    assert [o.path.name.rsplit(" - ", 1)[1] for o in videos] == ["Source 1.mkv", "Source 2.mkv"]
    # Shared audio: one download of s0q1 (init + 4 segments), s0q0 never fetched.
    assert server.count("/lesson/s0q1.mp4") == 5
    assert server.count("/lesson/s0q0.mp4") == 0
    assert server.count("/lesson/s1q1.mp4") == 4 and server.count("/lesson/s2q1.mp4") == 4
    for out in videos:
        data = _probe(out.path)
        subs = _streams(data, "subtitle")
        assert [s["codec_name"] for s in subs] == ["webvtt", "webvtt"]
        assert [s["tags"]["language"] for s in subs] == ["eng", "fre"]
        assert [s["tags"]["title"] for s in subs] == ["English — Echo360 captions",
                                                      "French — Echo360 captions"]
        assert [s["disposition"]["default"] for s in subs] == [0, 1]
        assert [c["tags"]["title"] for c in data["chapters"]] == \
            [c.title for c in CHAPTERS.chapters]
        assert [round(float(c["start_time"]), 3) for c in data["chapters"]] == [0.0, 2.5, 4.0]
        assert _first_cue_time(out.path, 0) == pytest.approx(1.0, abs=0.001)
    v1 = _streams(_probe(videos[0].path), "video")[0]
    v2 = _streams(_probe(videos[1].path), "video")[0]
    assert float(v1["start_time"]) == pytest.approx(0.066, abs=0.001)  # B-frame source
    assert float(v2["start_time"]) == pytest.approx(0.0, abs=0.001)  # no B-frames
    extracted = subprocess.run(["ffmpeg", "-v", "error", "-i", str(videos[0].path), "-map",
                                "0:s:0", "-c", "copy", "-f", "webvtt", "-"],
                               capture_output=True, check=True).stdout.decode()
    assert "<v Dr. Alice Smith>Welcome &amp; hello" in extracted  # speaker kept in MKV


def test_quality_and_source_selection(server, tmp_path):
    rec = make_recording(server.base_url)
    result = run_download(_request(rec, tmp_path, source_indexes=(2,),
                                   quality=QualitySpec("max_height", 120)),
                          FakeSession(server), progress=NullProgress())
    assert result.status is ResultStatus.OK, result.failures
    (video,) = _video_outputs(result)
    assert video.source_index == 2
    v = _streams(_probe(video.path), "video")[0]
    assert (v["width"], v["height"]) == (160, 90)
    assert server.count("/lesson/s0q0.mp4") > 0 and server.count("/lesson/s0q1.mp4") == 0
    assert server.count("/lesson/s1q0.mp4") == 0
    bad = run_download(_request(rec, tmp_path / "b", quality=QualitySpec("max_height", 60)),
                       FakeSession(server), progress=NullProgress())
    assert bad.status is ResultStatus.FAILED
    assert {f.kind for f in bad.failures} == {"quality-unavailable"}


def test_audio_only_m4a_with_chapters_and_sidecar_subtitles(server, tmp_path):
    rec = make_recording(server.base_url, captions=_captions(), chapters=CHAPTERS)
    result = run_download(_request(rec, tmp_path, audio_only=True),
                          FakeSession(server, TRANSCRIPTS), progress=NullProgress())
    assert result.status is ResultStatus.OK, result.failures
    (audio,) = [o for o in result.outputs if o.kind is OutputKind.AUDIO]
    assert audio.path.name.endswith(f"[{rec.short_id}] - audio.m4a")
    data = _probe(audio.path)
    assert [s["codec_type"] for s in data["streams"] if s["codec_type"] != "data"] == ["audio"]
    assert [c["tags"]["title"] for c in data["chapters"]] == [c.title for c in CHAPTERS.chapters]
    assert any(o.path.name.endswith(".en.vtt") for o in result.outputs)
    assert server.count("s1q1.mp4") == 0 and server.count("s2q1.mp4") == 0


def test_403_on_one_source_gives_partial(server, tmp_path):
    server.state.forbid = {"/lesson/s2"}
    rec = make_recording(server.base_url)
    session = FakeSession(server)
    progress = RecordingProgress()
    result = run_download(_request(rec, tmp_path), session, progress=progress)
    assert result.status is ResultStatus.PARTIAL
    assert [o.source_index for o in _video_outputs(result)] == [1]
    (failure,) = result.failures
    assert (failure.step, failure.target, failure.kind) == ("download", "Source 2", "forbidden")
    assert "403" in failure.message
    assert session.refresh_calls == 1
    assert result.exit_code == 5
    work = work_dir_for(tmp_path / "out", rec)
    assert work.exists() and any("run the same command" in w for w in result.warnings)
    _no_secrets(result, progress, tmp_path / "out", server)
    # A rerun reuses the completed components and fetches only source 2.
    server.state.forbid = set()
    server.reset_log()
    again = run_download(_request(rec, tmp_path), FakeSession(server), progress=NullProgress())
    assert again.status is ResultStatus.OK, again.failures
    assert server.count("s1q1.mp4") == 0 and server.count("s0q1.mp4") == 0
    assert server.count("s2q1.mp4") > 0
    assert not work.exists()


def test_expired_auth_mid_download_refreshes_exactly_once(server, tmp_path):
    server.state.expire_after = 3
    rec = make_recording(server.base_url)
    session = FakeSession(server)
    result = run_download(_request(rec, tmp_path, concurrency=3), session,
                          progress=NullProgress())
    assert result.status is ResultStatus.OK, result.failures
    assert session.refresh_calls == 1
    assert all(o.verify.ok for o in _video_outputs(result))


def test_refresh_failure_is_auth_failure(server, tmp_path):
    server.expire()
    rec = make_recording(server.base_url)
    session = FakeSession(server, refresh_error=expired_refresh_error())
    progress = RecordingProgress()
    result = run_download(_request(rec, tmp_path), session, progress=progress)
    assert result.status is ResultStatus.FAILED
    assert {f.kind for f in result.failures} == {"auth-expired"}
    assert session.refresh_calls == 1
    assert result.exit_code == 3
    _no_secrets(result, progress, tmp_path / "out", server)


def test_missing_language_is_partial_and_video_still_downloads(server, tmp_path):
    rec = make_recording(server.base_url, sources=(1,), captions=_captions())
    result = run_download(
        _request(rec, tmp_path, subtitles=SubtitleMode(SubtitleModeKind.LANGUAGES, ("en", "de"))),
        FakeSession(server, TRANSCRIPTS), progress=NullProgress())
    assert result.status is ResultStatus.PARTIAL
    (failure,) = result.failures
    assert (failure.kind, failure.target) == ("language-unavailable", "captions:de")
    (video,) = _video_outputs(result)
    subs = _streams(_probe(video.path), "subtitle")
    assert [s["tags"]["language"] for s in subs] == ["eng"]
    assert not work_dir_for(tmp_path / "out", rec).exists()  # nothing a rerun could fix


def test_no_captions_reports_absence(server, tmp_path):
    rec = make_recording(server.base_url, sources=(1,))
    result = run_download(_request(rec, tmp_path), FakeSession(server), progress=NullProgress())
    assert result.status is ResultStatus.OK
    assert any("No Echo360 captions or transcript" in w for w in result.warnings)
    assert any("No native chapters" in w for w in result.warnings)


def test_untimed_transcript_is_txt_only(server, tmp_path):
    track = caption("en-txt", "en", "txt", TranscriptProvenance.PLATFORM_TRANSCRIPT, timed=False)
    rec = make_recording(server.base_url, sources=(1,), captions=[track])
    result = run_download(_request(rec, tmp_path), FakeSession(server, TRANSCRIPTS),
                          progress=NullProgress())
    assert result.status is ResultStatus.OK, result.failures
    (video,) = _video_outputs(result)
    assert _streams(_probe(video.path), "subtitle") == []
    sidecars = sorted(o.path.name.rsplit("]", 1)[1] for o in result.outputs
                      if o.kind is not OutputKind.VIDEO)
    assert sidecars == [".en.transcript.txt"]
    assert any("no timestamps" in w for w in result.warnings)


def test_generated_transcript_once_for_all_sources(server, tmp_path):
    rec = make_recording(server.base_url)
    transcriber = FakeTranscriber()
    made: list[TranscribeOptions] = []

    def factory(options):
        made.append(options)
        return transcriber

    result = run_download(
        _request(rec, tmp_path, transcribe=TranscribeOptions(enabled=True, language="en")),
        FakeSession(server), progress=NullProgress(), transcriber_factory=factory)
    assert result.status is ResultStatus.OK, result.failures
    assert len(made) == 1 and len(transcriber.calls) == 1
    assert transcriber.calls[0].name == "audio-s0q1.mp4"
    for out in _video_outputs(result):
        subs = _streams(_probe(out.path), "subtitle")
        assert [s["tags"]["handler_name"] for s in subs] == ["English — generated (local)"]
    generated = [o for o in result.outputs if o.kind is OutputKind.SUBTITLE]
    assert {o.path.name.rsplit("]", 1)[1] for o in generated} == {".en.generated.vtt",
                                                                  ".en.generated.srt"}
    assert {o.provenance for o in generated} == {"generated-local"}


def test_transcription_skipped_when_platform_captions_exist(server, tmp_path):
    rec = make_recording(server.base_url, sources=(1,), captions=_captions())
    transcriber = FakeTranscriber()
    result = run_download(_request(rec, tmp_path, transcribe=TranscribeOptions(enabled=True)),
                          FakeSession(server, TRANSCRIPTS), progress=NullProgress(),
                          transcriber_factory=lambda o: transcriber)
    assert result.status is ResultStatus.OK and transcriber.calls == []
    forced = obtain_transcripts(
        rec, FakeSession(server, TRANSCRIPTS), subtitles=SubtitleMode(),
        transcribe=TranscribeOptions(enabled=True, only_if_missing=False),
        work_dir=tmp_path / "w", progress=NullProgress(),
        transcriber_factory=lambda o: transcriber, cancel=None)
    assert len(transcriber.calls) == 1
    assert [t.provenance for t in forced.transcripts][-1] is TranscriptProvenance.GENERATED_LOCAL
    assert forced.transcripts[-1].label == "English (generated)"


def test_cancel_mid_download_then_rerun_completes(server, tmp_path):
    rec = make_recording(server.base_url, sources=(1,))
    cancel = threading.Event()
    progress = RecordingProgress()
    progress.on_update = lambda task, u: (u.get("completed") or 0) > 2000 and cancel.set()
    server.state.throttle = 0.05
    out_dir = tmp_path / "out"
    result = run_download(_request(rec, tmp_path), FakeSession(server), progress=progress,
                          cancel=cancel)
    assert result.status is ResultStatus.CANCELLED and result.exit_code == 130
    assert _video_outputs(result) == []
    assert not list(out_dir.glob("*.mp4")) and not list(out_dir.glob(".*tmp*"))
    work = work_dir_for(out_dir, rec)
    assert list(work.glob("*.part")), "resumable partial data kept"
    assert any("resume" in w for w in result.warnings)
    server.state.throttle = 0.0
    again = run_download(_request(rec, tmp_path), FakeSession(server), progress=NullProgress())
    assert again.status is ResultStatus.OK, again.failures
    assert again.outputs[0].verify.ok and not work.exists()


@pytest.mark.parametrize("error", [KeyboardInterrupt, OSError])
def test_interrupted_finalisation_leaves_no_final_file(server, tmp_path, monkeypatch, error):
    def boom(src, dst):
        raise error("interrupted before os.replace")

    monkeypatch.setattr(mux_module, "_atomic_replace", boom)
    rec = make_recording(server.base_url, sources=(1,))
    result = run_download(_request(rec, tmp_path), FakeSession(server), progress=NullProgress())
    out_dir = tmp_path / "out"
    assert not list(out_dir.glob("*.mp4")) and not list(out_dir.glob(".*echo360-dl-tmp*"))
    if error is KeyboardInterrupt:
        assert result.status is ResultStatus.CANCELLED
    else:
        assert result.status is ResultStatus.FAILED
        assert result.failures[0].step == "finalize"
    assert work_dir_for(out_dir, rec).exists()  # components kept for a rerun
    monkeypatch.undo()
    server.reset_log()
    again = run_download(_request(rec, tmp_path), FakeSession(server), progress=NullProgress())
    assert again.status is ResultStatus.OK and server.count(".mp4") == 0  # reused


def test_verification_failure_keeps_marked_file(server, tmp_path, monkeypatch):
    real_verify = pipeline.verify

    def failing(path, job):
        report = real_verify(path, job)
        report.problems.append("simulated mismatch")
        report.ok = False
        return report

    monkeypatch.setattr(pipeline, "verify", failing)
    rec = make_recording(server.base_url, sources=(1,))
    result = run_download(_request(rec, tmp_path), FakeSession(server), progress=NullProgress())
    assert result.status is ResultStatus.FAILED
    (failure,) = result.failures
    assert failure.kind == "verification" and "simulated mismatch" in failure.message
    out_dir = tmp_path / "out"
    assert [p.name.endswith(" - Source 1.UNVERIFIED.mp4") for p in out_dir.glob("*.mp4")] == [True]
    assert "UNVERIFIED" in failure.hint


@pytest.mark.parametrize("policy", list(IfExists))
def test_if_exists(server, tmp_path, policy):
    rec = make_recording(server.base_url, sources=(1,))
    first = run_download(_request(rec, tmp_path), FakeSession(server), progress=NullProgress())
    final = _video_outputs(first)[0].path
    final.write_bytes(b"old")
    server.reset_log()
    result = run_download(_request(rec, tmp_path, if_exists=policy), FakeSession(server),
                          progress=NullProgress())
    if policy is IfExists.SKIP:
        assert result.status is ResultStatus.OK
        assert result.outputs[0].skipped_existing and final.read_bytes() == b"old"
        assert sum(server.state.requests.values()) == 0
    elif policy is IfExists.ERROR:
        assert result.status is ResultStatus.FAILED
        assert result.failures[0].kind == "usage" and final.read_bytes() == b"old"
    else:
        assert result.status is ResultStatus.OK
        assert final.stat().st_size > 1000 and _video_outputs(result)[0].verify.ok


def test_interval_and_file_chapters(server, tmp_path):
    native = ChapterSet([Chapter(0.0, "Native")], ChapterProvenance.NATIVE)
    rec = make_recording(server.base_url, sources=(1,), chapters=native)
    result = run_download(
        _request(rec, tmp_path, chapters=ChapterMode(ChapterModeKind.FILE,
                                                     file=FIXTURES / "chapters_valid.json")),
        FakeSession(server), progress=NullProgress())
    assert result.status is ResultStatus.OK, result.failures
    assert any("overridden" in w for w in result.warnings)
    data = _probe(_video_outputs(result)[0].path)
    assert [c["tags"]["title"] for c in data["chapters"]] == ["Intro = ; # \\ é",
                                                             "Second — 第二", "Wrap-up"]
    assert [round(float(c["end_time"]), 3) for c in data["chapters"][:2]] == [2.5, 4.0]

    interval = run_download(
        _request(rec, tmp_path / "i", chapters=ChapterMode(ChapterModeKind.INTERVAL,
                                                           interval=2.0)),
        FakeSession(server), progress=NullProgress())
    data = _probe(_video_outputs(interval)[0].path)
    assert [c["tags"]["title"] for c in data["chapters"]] == ["Part 1", "Part 2", "Part 3"]


def test_invalid_chapter_file_fails_before_downloading(server, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"version": 1, "chapters": [{"start": "00:00:05", "title": "b"}, '
                   '{"start": "00:00:02", "title": "a"}]}')
    rec = make_recording(server.base_url, sources=(1,))
    result = run_download(_request(rec, tmp_path, chapters=ChapterMode(ChapterModeKind.FILE,
                                                                       file=bad)),
                          FakeSession(server), progress=NullProgress())
    assert result.status is ResultStatus.FAILED
    assert result.failures[0].kind == "validation"
    assert sum(server.state.requests.values()) == 0


def test_chapter_file_beyond_measured_duration_is_caught_after_download(server, tmp_path):
    """The pre-download check ignores the (unverified) metadata duration; the measured
    media duration is authoritative."""
    late = tmp_path / "late.json"
    late.write_text('{"version": 1, "chapters": [{"start": "00:00:09", "title": "late"}]}')
    rec = make_recording(server.base_url, sources=(1,))
    rec.duration = 1.0  # deliberately wrong metadata must not abort the run
    result = run_download(_request(rec, tmp_path, chapters=ChapterMode(ChapterModeKind.FILE,
                                                                       file=late)),
                          FakeSession(server), progress=NullProgress())
    assert result.status is ResultStatus.PARTIAL
    assert [o.kind for o in result.outputs if o.kind is OutputKind.VIDEO]
    assert any(f.kind == "validation" and "after the end" in f.message
               for f in result.failures)


def test_all_media_failing_is_failed_not_partial(server, tmp_path):
    """Sidecars alone are not a partial success."""
    server.state.forbid = {"/lesson/s1"}
    rec = make_recording(server.base_url, sources=(1,), captions=_captions())
    result = run_download(_request(rec, tmp_path, source_indexes=(1,)),
                          FakeSession(server, TRANSCRIPTS), progress=NullProgress())
    assert not [o for o in result.outputs if o.kind in (OutputKind.VIDEO, OutputKind.AUDIO)]
    assert result.status is ResultStatus.FAILED
    assert result.exit_code == 3  # the only failure kind is "forbidden" (auth family)


def test_malformed_platform_subtitle_is_reported_not_fatal(server, tmp_path):
    rec = make_recording(server.base_url, sources=(1,), captions=[caption("en", "en", "vtt")])
    session = FakeSession(server, {"en": b"\x00\x01garbage\xff"})
    result = run_download(_request(rec, tmp_path), session, progress=NullProgress())
    # The video is still produced; the unusable caption track is a reported failure.
    assert result.status is ResultStatus.PARTIAL
    assert any("Could not fetch English captions" in f.message for f in result.failures)


# -- review fixes: fail-fast transcription, native chapter warnings ------------------------


def test_missing_speech_model_fails_before_downloading(server, tmp_path):
    from echo360_dl.errors import ModelDownloadRequiredError
    from echo360_dl.models import TranscribeOptions

    def factory(options):
        raise ModelDownloadRequiredError("Speech model 'small' is not downloaded yet.")

    rec = make_recording(server.base_url, sources=(1,))  # no platform captions
    result = run_download(_request(rec, tmp_path, transcribe=TranscribeOptions(enabled=True)),
                          FakeSession(server), progress=NullProgress(),
                          transcriber_factory=factory)
    assert result.status is ResultStatus.FAILED and result.exit_code == 6
    assert sum(server.state.requests.values()) == 0  # nothing downloaded first


def test_transcriber_resolved_once_and_reused(server, tmp_path):
    from echo360_dl.models import TranscribeOptions
    from tests.media.conftest import FakeTranscriber

    made: list[FakeTranscriber] = []

    def factory(options):
        made.append(FakeTranscriber())
        return made[-1]

    rec = make_recording(server.base_url, sources=(1, 2))
    result = run_download(_request(rec, tmp_path, transcribe=TranscribeOptions(enabled=True)),
                          FakeSession(server), progress=NullProgress(),
                          transcriber_factory=factory)
    assert result.status is ResultStatus.OK, result.failures
    assert len(made) == 1 and len(made[0].calls) == 1


def test_inconsistent_native_chapters_warn_but_do_not_fail(server, tmp_path):
    from echo360_dl.models import ChapterProvenance

    native = ChapterSet([Chapter(0.0, "a"), Chapter(500.0, "beyond the end")],
                        ChapterProvenance.NATIVE)
    rec = make_recording(server.base_url, sources=(1,), chapters=native)
    result = run_download(_request(rec, tmp_path), FakeSession(server), progress=NullProgress())
    assert result.status is ResultStatus.OK, result.failures
    assert any("Native chapters were not embedded" in w for w in result.warnings)
