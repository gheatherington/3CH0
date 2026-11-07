"""Recording assembly and request-file mode with httpx.MockTransport (no network)."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from threech0.echo360.capture import CaptureState, ObservedResponse
from threech0.echo360.discovery import (
    build_recording,
    discover,
    discover_from_request_file,
    sibling_master_url,
)
from threech0.echo360.session import Echo360Session
from threech0.echo360.urls import parse_classroom_url
from threech0.errors import AuthExpiredError, UnauthorizedError, UnavailableError, UsageError
from threech0.models import (
    AccessState,
    AuthContext,
    Cookie,
    DiscoveryMethod,
    DiscoveryOptions,
    TranscriptProvenance,
    to_jsonable,
)

from .conftest import (
    CLASSROOM,
    FAKE_JWT,
    FAKE_SIGNATURE,
    FIXTURES,
    INSTITUTION,
    LESSON,
    MEDIA,
    OTHER_LESSON,
    cloudfront_policy,
    master_text,
    playlist_url,
)

REF = parse_classroom_url(CLASSROOM)


def _auth() -> AuthContext:
    return AuthContext(cookies=[
        Cookie("CloudFront-Signature", FAKE_SIGNATURE, ".echo360.ca", http_only=True),
        Cookie("ECHO_JWT", FAKE_JWT, "echo360.ca")], headers={"referer": "https://echo360.ca/"})


def _content_handler(status_by_source: dict[int, int], *,
                     transcript: bytes | None = None,
                     seen: list[httpx.Request] | None = None
                     ) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        path = request.url.path
        if path.endswith("/transcript"):
            if transcript is None:
                return httpx.Response(404)
            return httpx.Response(200, content=transcript,
                                  headers={"content-type": "application/json"})
        for idx, status in status_by_source.items():
            if path.endswith(f"/s{idx}_av.m3u8"):
                if status != 200:
                    return httpx.Response(status, text="<Error>AccessDenied</Error>")
                text = (master_text(idx) if idx in (1, 2)
                        else master_text(2).replace("s2q", f"s{idx}q"))
                return httpx.Response(200, text=text)
        return httpx.Response(403)  # CloudFront answers 403 for absent objects too
    return handler


def _session(handler) -> Echo360Session:
    return Echo360Session(_auth(), transport=httpx.MockTransport(handler), sleep=lambda s: None)


def _state(*events: ObservedResponse) -> CaptureState:
    state = CaptureState(LESSON)
    for ev in events:
        state.add(ev)
    return state


def _pl(index: int, status: int = 200, **kw) -> ObservedResponse:
    return ObservedResponse(url=playlist_url(index, **kw), status=status,
                            content_type="application/x-mpegurl")


def _meta() -> ObservedResponse:
    return ObservedResponse(url="https://echo360.ca/section/fictional/syllabus", status=200,
                            content_type="application/json",
                            body=(FIXTURES / "lesson_metadata.json").read_bytes())


def test_two_sources_shared_audio_labels_captions_chapters(progress):
    state = _state(_meta(), _pl(1), _pl(2))
    rec = build_recording(REF, state, _session(_content_handler({1: 200, 2: 200})), progress)
    assert [s.index for s in rec.sources] == [1, 2]
    assert all(s.access is AccessState.AVAILABLE for s in rec.sources)
    assert [s.display_name for s in rec.sources] == ["Lecturer camera", "Classroom screen"]
    assert all(s.label_method is DiscoveryMethod.METADATA for s in rec.sources)
    # Metadata lists both sources, so it is the recorded evidence even though the player
    # also requested them (metadata is the stronger source).
    assert all(s.method is DiscoveryMethod.METADATA for s in rec.sources)
    assert [r.id for r in rec.sources[1].renditions] == ["s2q0", "s2q1"]
    assert len(rec.audio_tracks) == 1 and rec.audio_tracks[0].shared
    assert rec.audio_tracks[0].used_by_sources == [1, 2]
    assert {s.audio_track_id for s in rec.sources} == {"s0"}
    assert rec.title.startswith("Fictional Lecture 7") and rec.duration == 3240.0
    assert rec.media_id == MEDIA and rec.institution_id == INSTITUTION
    assert sorted(rec.caption_languages) == ["en", "fr-CA"]
    assert [c.title for c in rec.chapters.chapters] == ["Introduction", "Worked example"]
    assert any(not n.observed and "hypothesis" in n.message for n in rec.notes)
    # Nothing secret reaches the JSON form of the recording.
    dumped = json.dumps(to_jsonable(rec))
    for secret in (FAKE_SIGNATURE, FAKE_JWT, "x-uid", "FAKESESSION000"):
        assert secret not in dumped


def test_source2_forbidden_is_not_absent_and_source1_still_usable(progress):
    state = _state(_pl(1), _pl(2))
    rec = build_recording(REF, state, _session(_content_handler({1: 200, 2: 403})), progress)
    s1, s2 = rec.source(1), rec.source(2)
    assert s1.access is AccessState.AVAILABLE and s1.renditions
    assert s2 is not None, "a 403 must never make a source disappear"
    assert s2.access is AccessState.FORBIDDEN and s2.access is not AccessState.NOT_FOUND
    assert "does not prove absence" in s2.access_detail
    assert s1.audio_track_id == "s0" and rec.audio_tracks[0].used_by_sources == [1]
    assert any("Source 2" in n.message and n.level == "warning" for n in rec.notes)


def test_naming_probe_fallback_bounded(progress):
    seen: list[httpx.Request] = []
    state = _state(_pl(2))  # the player only requested source 2
    rec = build_recording(REF, state,
                          _session(_content_handler({1: 200, 2: 200}, seen=seen)), progress)
    assert [s.index for s in rec.sources] == [1, 2]
    assert rec.source(1).method is DiscoveryMethod.NAMING_PROBE
    probed = sorted({r.url.path.rsplit("/", 1)[-1] for r in seen if "_av" in r.url.path})
    assert probed == ["s1_av.m3u8", "s2_av.m3u8", "s3_av.m3u8", "s4_av.m3u8"]
    msgs = [n.message for n in rec.notes]
    assert "Source 3: could not access (403 does not prove absence)." in msgs
    # Recorded so `--streams 3` reports "403, may exist" rather than "not found".
    assert rec.unreachable_sources == {3: AccessState.FORBIDDEN, 4: AccessState.FORBIDDEN}
    assert any("probed sibling" in m for m in msgs)
    assert all(n.observed is False for n in rec.notes if "probed sibling" in n.message)
    # probe query string preserved (x-lid etc.) for auth correlation
    assert all("x-lid=" in str(r.url) for r in seen if "_av" in r.url.path)


def test_probe_stops_after_two_misses_above_highest(progress):
    seen: list[httpx.Request] = []
    state = _state(_pl(1))
    build_recording(REF, state, _session(_content_handler({1: 200, 5: 200}, seen=seen)),
                    progress)
    names = [r.url.path.rsplit("/", 1)[-1] for r in seen if "_av" in r.url.path]
    assert names == ["s1_av.m3u8", "s2_av.m3u8", "s3_av.m3u8"]


def test_metadata_listed_sources_skip_probing(progress):
    seen: list[httpx.Request] = []
    state = _state(_meta(), _pl(1))
    rec = build_recording(REF, state, _session(_content_handler({1: 200, 2: 200}, seen=seen)),
                          progress, completion_reason="settled", missing_sources=(2,))
    assert rec.source(2).method is DiscoveryMethod.METADATA
    names = {r.url.path.rsplit("/", 1)[-1] for r in seen if "_av" in r.url.path}
    assert names == {"s1_av.m3u8", "s2_av.m3u8"}
    assert any("never requested" in n.message and not n.observed for n in rec.notes)


def test_media_playlist_only_derives_master(progress):
    media = ObservedResponse(url=playlist_url(1).replace("s1_av", "s1q1"), status=200,
                             content_type="application/x-mpegurl")
    rec = build_recording(REF, _state(media), _session(_content_handler({1: 200})), progress)
    assert rec.source(1).method is DiscoveryMethod.NAMING_PROBE
    assert rec.source(1).access is AccessState.AVAILABLE


def test_cross_lesson_prefetch_never_becomes_a_source(progress):
    state = _state(_pl(1), _pl(3, lesson=OTHER_LESSON))
    rec = build_recording(REF, state, _session(_content_handler({1: 200, 3: 200})), progress)
    assert 3 not in [s.index for s in rec.sources if s.method is
                     DiscoveryMethod.OBSERVED_REQUEST]
    assert any("other lesson" in n.message for n in rec.notes)


def test_no_media_raises(progress):
    with pytest.raises(UnavailableError, match="within 30 s"):
        build_recording(REF, _state(ObservedResponse(url="https://echo360.ca/x", status=200)),
                        _session(_content_handler({})), progress,
                        completion_reason="timeout", timeout=30)


def test_timeout_with_media_returns_with_warning(progress):
    rec = build_recording(REF, _state(_pl(1)), _session(_content_handler({1: 200})),
                          progress, completion_reason="timeout", timeout=30)
    assert any(n.level == "warning" and "time limit" in n.message for n in rec.notes)


def test_all_sources_expired_raises_auth_error(progress):
    auth = AuthContext(cookies=_auth().cookies, expires_at=1.0)
    session = Echo360Session(auth, transport=httpx.MockTransport(_content_handler({1: 403})),
                             sleep=lambda s: None)
    with pytest.raises(AuthExpiredError):
        build_recording(REF, _state(_pl(1)), session, progress)


def test_observed_transcript_and_candidate_endpoint(progress):
    body = (FIXTURES / "transcript_cues.json").read_bytes()
    observed = ObservedResponse(url=f"https://echo360.ca/api/ui/echoplayer/lessons/{LESSON}/x",
                                status=200, content_type="application/json", body=body)
    rec = build_recording(REF, _state(_pl(1), observed),
                          _session(_content_handler({1: 200})), progress)
    assert [c.method for c in rec.captions] == [DiscoveryMethod.OBSERVED_REQUEST]
    # Not observed -> the hypothesised endpoint is tried once and labelled.
    seen: list[httpx.Request] = []
    rec = build_recording(REF, _state(_pl(1)),
                          _session(_content_handler({1: 200}, transcript=body, seen=seen)),
                          progress)
    track = rec.captions[0]
    assert track.kind is TranscriptProvenance.PLATFORM_TRANSCRIPT
    assert track.method is DiscoveryMethod.NAMING_PROBE and track.timed is False
    assert sum(r.url.path.endswith("/transcript") for r in seen) == 1
    assert any("not verified" in n.message and not n.observed for n in rec.notes)


# -- request-file mode ----------------------------------------------------------------------


def _request_file(tmp_path, url: str, *, epoch: int = 4_102_000_000):
    path = tmp_path / "req.txt"
    path.write_text(f"curl '{url}' -H 'referer: https://echo360.ca/' "
                    f"-b 'CloudFront-Policy={cloudfront_policy(epoch)}; "
                    f"CloudFront-Signature={FAKE_SIGNATURE}'\n")
    return path


def _opts(path) -> DiscoveryOptions:
    return DiscoveryOptions(auth_mode="request-file", request_file=path, interactive=False)


def test_request_file_mode(tmp_path, progress):
    path = _request_file(tmp_path, playlist_url(2))
    seen: list[httpx.Request] = []
    rec, session = discover_from_request_file(
        REF, _opts(path), progress,
        transport=httpx.MockTransport(_content_handler({1: 200, 2: 200}, seen=seen)),
        sleep=lambda s: None)
    assert [s.index for s in rec.sources] == [1, 2]
    assert rec.source(2).method is DiscoveryMethod.USER_INPUT
    assert rec.source(1).method is DiscoveryMethod.NAMING_PROBE
    assert rec.audio_tracks[0].shared and rec.media_id == MEDIA
    assert rec.captions == [] and rec.chapters is None and rec.title is None
    assert any("metadata, transcripts and chapters are unavailable" in n.message
               for n in rec.notes)
    assert all(r.headers["cookie"].count(FAKE_SIGNATURE) == 1 for r in seen)
    assert session.auth.source == "request-file"
    with pytest.raises(AuthExpiredError, match="saved request file"):
        session.refresh()


def test_request_file_wrong_lesson_is_usage_error(tmp_path, progress):
    path = _request_file(tmp_path, playlist_url(1, lesson=OTHER_LESSON))
    with pytest.raises(UsageError, match="different lesson"):
        discover_from_request_file(REF, _opts(path), progress,
                                   transport=httpx.MockTransport(_content_handler({1: 200})))


def test_request_file_missing_lid_and_bad_host(tmp_path, progress):
    no_lid = playlist_url(1).split("?")[0]
    with pytest.raises(UsageError, match="x-lid"):
        discover_from_request_file(REF, _opts(_request_file(tmp_path, no_lid)), progress)
    with pytest.raises(UsageError, match="not for an Echo360 host"):
        discover_from_request_file(
            REF, _opts(_request_file(tmp_path, "https://evil.example/s1_av.m3u8?x-lid=" + LESSON)),
            progress)


def test_request_file_expired_or_refused(tmp_path, progress):
    expired = _request_file(tmp_path, playlist_url(1), epoch=1_000_000_000)
    with pytest.raises(AuthExpiredError):
        discover_from_request_file(REF, _opts(expired), progress)
    fresh = _request_file(tmp_path, playlist_url(1))
    with pytest.raises(UnauthorizedError):
        discover_from_request_file(REF, _opts(fresh), progress,
                                   transport=httpx.MockTransport(_content_handler({1: 403})),
                                   sleep=lambda s: None)


def test_discover_dispatch_writes_trace_in_request_file_mode(tmp_path, progress, monkeypatch):
    import threech0.echo360.discovery as disc

    path = _request_file(tmp_path, playlist_url(1))
    transport = httpx.MockTransport(_content_handler({1: 200}))
    original = disc.discover_from_request_file

    def patched(ref, options, progress, *, trace=None):
        return original(ref, options, progress, trace=trace, transport=transport,
                        sleep=lambda s: None)

    monkeypatch.setattr(disc, "discover_from_request_file", patched)
    trace_path = tmp_path / "trace.json"
    options = DiscoveryOptions(auth_mode="request-file", request_file=path, interactive=False,
                               save_trace=trace_path)
    rec, _session = discover(REF, options, progress)
    assert rec.sources
    text = trace_path.read_text()
    assert "s1_av.m3u8" in text
    for secret in (FAKE_SIGNATURE, "x-uid", "FAKESESSION000", "?"):
        assert secret not in text


def test_sibling_master_url():
    assert sibling_master_url(playlist_url(2), 1) == playlist_url(1)
    assert sibling_master_url(playlist_url(2).replace("s2_av", "s2q1"), 2) == playlist_url(2)
    assert sibling_master_url("https://content.echo360.ca/x/video.m3u8", 1) is None


def _named(index_name: str) -> ObservedResponse:
    return ObservedResponse(url=playlist_url(1).replace("/s1_av.m3u8", f"/{index_name}.m3u8"),
                            status=200, content_type="application/x-mpegurl")


def test_observed_echo360_traffic_pattern_2026_11_07(progress):
    """Replays the real (sanitised) trace: the player asked for s1_v (video-only master of
    source 1), s2_av, per-source media playlists and the shared s0q0 audio. Expected: two
    sources from the audio-paired s<N>_av masters, no "Source 0" (audio) and no phantom
    source for s1_v."""
    state = _state(_named("s1_v"), _named("s2_av"), _named("s1q0"), _named("s2q0"),
                   _named("s0q0"))
    rec = build_recording(REF, state, _session(_content_handler({1: 200, 2: 200})), progress)
    assert [s.index for s in rec.sources] == [1, 2]
    assert rec.source(1).playlist_url.split("?")[0].endswith("/s1_av.m3u8")
    assert all(s.access is AccessState.AVAILABLE for s in rec.sources)
    assert rec.audio_tracks and rec.audio_tracks[0].used_by_sources == [1, 2]
    assert 0 not in rec.unreachable_sources


def test_video_only_master_used_when_audio_paired_master_is_refused(progress):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/s1_v.m3u8"):
            return httpx.Response(200, text=master_text(1))
        return _content_handler({2: 200})(request)

    state = _state(_named("s1_v"), _named("s2_av"))
    rec = build_recording(REF, state, _session(handler), progress)
    assert rec.source(1) is not None
    assert rec.source(1).playlist_url.split("?")[0].endswith("/s1_v.m3u8")
    assert rec.source(1).method is DiscoveryMethod.OBSERVED_REQUEST


def test_player_metadata_takes_precedence_over_observed_playlists(progress):
    """Observed 2026-11-07: the player requested s1_v and s2_av while the page's player
    object listed s1_av/s1_v/s2_av/s2_v. Both sources must come from the metadata."""

    meta_urls = [playlist_url(1), playlist_url(2),
                 playlist_url(1).replace("/s1_av.", "/s1_v."),
                 playlist_url(2).replace("/s2_av.", "/s2_v.")]
    state = _state(_named("s1_v"), _named("s2_av"))
    original = state.metadata

    def metadata_with_player(html=None):
        meta = original(html)
        meta.playlist_urls = list(meta_urls)
        return meta

    state.metadata = metadata_with_player  # type: ignore[method-assign]
    rec = build_recording(REF, state, _session(_content_handler({1: 200, 2: 200})), progress)
    assert [(s.index, s.method) for s in rec.sources] == [(1, DiscoveryMethod.METADATA),
                                                          (2, DiscoveryMethod.METADATA)]
    assert all(s.playlist_url.split("?")[0].endswith("_av.m3u8") for s in rec.sources)
    assert not any("probed sibling" in n.message for n in rec.notes)
