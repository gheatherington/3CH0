"""Side-by-side / top-bottom composite videos: real ffmpeg encodes, verified with ffprobe."""

from __future__ import annotations

from pathlib import Path

import pytest

from echo360_dl.media.ffmpeg import ffprobe_json
from echo360_dl.media.mux import (
    CompositeSpec,
    InputInfo,
    available_encoders,
    composite_filter,
    composite_tile_sizes,
    resolve_encoder,
)
from echo360_dl.media.pipeline import run_download
from echo360_dl.models import (
    Chapter,
    ChapterProvenance,
    ChapterSet,
    Container,
    DownloadRequest,
    Encoder,
    Layout,
    NullProgress,
    OutputKind,
    ResultStatus,
)
from echo360_dl.selection import composite_pair, parse_layout_order
from tests.media.conftest import FIXTURES, FakeSession, RecordingProgress, caption, make_recording

CHAPTERS = ChapterSet([Chapter(0.0, "Intro"), Chapter(3.0, "Second")],
                      ChapterProvenance.NATIVE, "metadata")
TRANSCRIPTS = {"en": (FIXTURES / "captions_en.vtt").read_bytes()}


def _job(layout: str, *sizes: tuple[int, int]) -> CompositeSpec:
    return CompositeSpec(videos=[InputInfo(Path(f"{i}.mp4"), width=w, height=h)
                                 for i, (w, h) in enumerate(sizes)],
                         layout=layout, encoder="libx264")


# -- pure layout maths -------------------------------------------------------------------


def test_side_by_side_scales_to_common_height_keeping_aspect():
    job = _job("side-by-side", (1280, 720), (640, 480))  # 16:9 camera + 4:3 screen
    assert composite_tile_sizes(job) == [(1280, 720), (960, 720)]
    assert job.size == (2240, 720)
    assert "hstack=inputs=2" in composite_filter(job)


def test_top_bottom_scales_to_common_width_keeping_aspect():
    job = _job("top-bottom", (1280, 720), (853, 480))
    assert composite_tile_sizes(job) == [(1280, 720), (1280, 720)]
    assert job.size == (1280, 1440)
    assert "vstack=inputs=2" in composite_filter(job)


def test_odd_sizes_become_even():
    job = _job("side-by-side", (641, 361), (999, 563))
    assert all(w % 2 == 0 and h % 2 == 0 for w, h in composite_tile_sizes(job))


def test_composite_pair_rules():
    from echo360_dl.errors import UsageError

    assert composite_pair((1, 2)) == (1, 2)
    assert composite_pair((1, 2), (2, 1)) == (2, 1)
    assert composite_pair((1, 2, 3), (3, 1)) == (3, 1)
    with pytest.raises(UsageError, match="exactly two"):
        composite_pair((1,))
    with pytest.raises(UsageError, match="exactly two"):
        composite_pair((1, 2, 3))
    with pytest.raises(UsageError, match="not selected"):
        composite_pair((1, 2), (3, 1))
    assert parse_layout_order("2,1") == (2, 1)
    with pytest.raises(UsageError):
        parse_layout_order("1")


def test_encoder_resolution():
    have = available_encoders()
    if "h264_videotoolbox" in have:
        assert resolve_encoder("auto") == "h264_videotoolbox"
    if "libx264" in have:
        assert resolve_encoder("x264") == "libx264"


# -- real encodes ------------------------------------------------------------------------


def _request(rec, tmp_path, **kw) -> DownloadRequest:
    kw.setdefault("output_dir", tmp_path / "out")
    kw.setdefault("source_indexes", (1, 2))
    return DownloadRequest(recording=rec, **kw)


def _streams(path: Path, kind: str) -> list[dict]:
    return [s for s in ffprobe_json(path)["streams"] if s["codec_type"] == kind]


@pytest.mark.ffmpeg
@pytest.mark.parametrize(("layout", "container", "encoder", "size"), [
    (Layout.SIDE_BY_SIDE, Container.MP4, Encoder.AUTO, (640, 180)),
    (Layout.TOP_BOTTOM, Container.MKV, Encoder.X264, (320, 360)),
])
def test_composite_download(server, tmp_path, layout, container, encoder, size):
    rec = make_recording(server.base_url, captions=[caption("en", "en", "vtt")],
                         chapters=CHAPTERS)
    progress = RecordingProgress()
    result = run_download(_request(rec, tmp_path, layout=layout, container=container,
                                   encoder=encoder),
                          FakeSession(server, TRANSCRIPTS), progress=progress)
    assert result.status is ResultStatus.OK, result.failures
    media = [o for o in result.outputs if o.kind in (OutputKind.VIDEO, OutputKind.COMPOSITE)]
    assert [o.kind for o in media] == [OutputKind.COMPOSITE]  # no separate files by default
    out = media[0].path
    assert ("side by side" if layout is Layout.SIDE_BY_SIDE else "top-bottom") in out.name
    (video,) = _streams(out, "video")
    assert video["codec_name"] == "h264" and (video["width"], video["height"]) == size
    (audio,) = _streams(out, "audio")
    assert audio["codec_name"] == "aac"  # audio stream-copied
    subs = _streams(out, "subtitle")
    assert len(subs) == 1 and subs[0]["tags"].get("language") == "eng"
    data = ffprobe_json(out)
    assert [c["tags"]["title"] for c in data["chapters"]] == ["Intro", "Second"]
    # sync: the composite keeps the video's start offset relative to the audio
    assert abs(float(video["start_time"]) - float(audio["start_time"])) < 0.1
    assert media[0].verify is not None and media[0].verify.ok
    # the shared audio was downloaded once, the encode reported real progress
    assert server.count("/lesson/s0q1.mp4") > 0
    encode_updates = [u for t, u in progress.updates if t.startswith("encode:")]
    assert encode_updates and all(u.get("total") for u in encode_updates)  # real totals
    assert encode_updates[-1]["completed"] == encode_updates[-1]["total"]  # ends at 100 %
    assert any(desc.startswith("Encoding ") for _t, desc in progress.starts)
    # chapters survive as a real QuickTime chapter track (step 2 is a stream-copy mux)
    if container is Container.MP4:
        assert any(s["codec_type"] == "data" for s in ffprobe_json(out)["streams"])


@pytest.mark.ffmpeg
def test_composite_order_and_keep_separate(server, tmp_path):
    rec = make_recording(server.base_url)
    result = run_download(_request(rec, tmp_path, layout=Layout.SIDE_BY_SIDE,
                                   layout_order=(2, 1), keep_separate=True,
                                   encoder=Encoder.X264),
                          FakeSession(server), progress=NullProgress())
    assert result.status is ResultStatus.OK, result.failures
    kinds = sorted(o.kind.value for o in result.outputs
                   if o.kind in (OutputKind.VIDEO, OutputKind.COMPOSITE))
    assert kinds == ["composite", "video", "video"]
    composite = next(o for o in result.outputs if o.kind is OutputKind.COMPOSITE)
    assert "Source 2 + Source 1" in composite.path.name


@pytest.mark.ffmpeg
def test_composite_with_one_source_fails_clearly(server, tmp_path):
    rec = make_recording(server.base_url, sources=(1,))
    result = run_download(_request(rec, tmp_path, source_indexes=(1,),
                                   layout=Layout.TOP_BOTTOM),
                          FakeSession(server), progress=NullProgress())
    assert result.status is ResultStatus.FAILED
    assert result.failures[0].kind == "usage" and "exactly two" in result.failures[0].message
    assert sum(server.state.requests.values()) == 0  # nothing downloaded


def test_videotoolbox_bitrate_follows_inputs(tmp_path):
    from echo360_dl.media.mux import VT_MAX_BITRATE, composite_bitrate

    files = []
    for i, size in enumerate((10_500_000, 19_800_000)):  # live 360p video-only sizes
        f = tmp_path / f"{i}.mp4"
        f.write_bytes(b"\0" * size)
        files.append(InputInfo(f, width=640, height=360, duration=3240.0))
    spec = CompositeSpec(videos=files, layout="side-by-side", encoder="h264_videotoolbox")
    rate = composite_bitrate(spec)
    inputs = (10_500_000 + 19_800_000) * 8 / 3240
    # 2x the inputs' video bitrate, but never below the 300 kb/s quality floor
    assert rate == max(int(2 * inputs), 300_000, int(0.012 * 1280 * 360 * 30))
    assert rate < 600_000  # still far below the ~650 kb/s constant-quality result seen live
    assert rate <= VT_MAX_BITRATE


def test_composite_is_reused_on_rerun(server, tmp_path):
    rec = make_recording(server.base_url)
    req = _request(rec, tmp_path, layout=Layout.SIDE_BY_SIDE, encoder=Encoder.X264,
                   keep_work_dir=True)
    first = run_download(req, FakeSession(server), progress=NullProgress())
    assert first.status is ResultStatus.OK, first.failures
    out = next(o.path for o in first.outputs if o.kind is OutputKind.COMPOSITE)
    out.unlink()
    progress = RecordingProgress()
    second = run_download(req, FakeSession(server), progress=progress)
    assert second.status is ResultStatus.OK, second.failures
    assert not any(t.startswith("encode:") for t, _d in progress.starts)  # not re-encoded
    assert any("Reusing the already encoded" in m for m in progress.messages)
