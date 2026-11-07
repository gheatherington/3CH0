"""Completion logic and correlation, driven by fake event streams (no browser)."""

from __future__ import annotations

import json

from echo360_dl.echo360.capture import (
    CaptureState,
    ObservedResponse,
    run_capture_loop,
)

from .conftest import FIXTURES, LESSON, MEDIA, OTHER_LESSON, OTHER_MEDIA, master_text, playlist_url


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _playlist(index: int, **kw) -> ObservedResponse:
    return ObservedResponse(url=playlist_url(index, **kw), status=200,
                            content_type="application/vnd.apple.mpegurl",
                            body=master_text(1).encode(),
                            request_headers={"Referer": "https://echo360.ca/",
                                             "Cookie": "SHOULD-NOT-BE-KEPT",
                                             "User-Agent": "UA/1"})


def _metadata_response() -> ObservedResponse:
    return ObservedResponse(url="https://echo360.ca/section/fictional/syllabus", status=200,
                            content_type="application/json; charset=utf-8",
                            body=(FIXTURES / "lesson_metadata.json").read_bytes())


def _run(events: dict[float, list[ObservedResponse]], *, timeout: float = 90.0,
         play=None):
    """Drive run_capture_loop with events scheduled at relative times."""
    clock = Clock()
    state = CaptureState(LESSON, clock=clock)
    start = clock.t
    pending = sorted(events.items())

    def pump(seconds: float) -> None:
        clock.t += seconds
        while pending and start + pending[0][0] <= clock.t:
            for ev in pending.pop(0)[1]:
                state.add(ev)

    completion = run_capture_loop(state, pump=pump, clock=clock, deadline=start + timeout,
                                  play=play)
    return state, completion, clock.t - start


def test_does_not_stop_at_first_video_request():
    state, completion, elapsed = _run({
        1.0: [_metadata_response()],
        2.0: [_playlist(1)],
        4.0: [_playlist(2)],
    })
    assert completion.reason == "complete"
    assert state.accounted_sources() == {1, 2}
    assert 7.0 <= elapsed <= 7.5  # quiet for 3 s after the last relevant response (t=4)
    assert state.request_headers == {"referer": "https://echo360.ca/", "user-agent": "UA/1"}


def test_waits_for_sources_listed_in_metadata_then_settles():
    state, completion, elapsed = _run({1.0: [_metadata_response()], 2.0: [_playlist(1)]})
    assert completion.reason == "settled"
    assert completion.missing_sources == (2,)
    assert elapsed >= 2.0 + state.settle_period


def test_without_metadata_completes_after_quiet_period():
    _state, completion, elapsed = _run({2.0: [_playlist(2)]})
    assert completion.reason == "complete" and 5.0 <= elapsed <= 5.5


def test_segment_traffic_does_not_prevent_quiet():
    seg = [ObservedResponse(url="https://content.echo360.ca/x/s1q1.mp4", status=206,
                            content_type="video/mp4")]
    events = {2.0: [_playlist(1)], **{2.0 + i * 0.5: seg for i in range(1, 30)}}
    _state, completion, elapsed = _run(events)
    assert completion.reason == "complete" and elapsed < 6


def test_polling_json_does_not_prevent_quiet():
    """Heartbeat/analytics JSON every second must not keep discovery running to the
    timeout: repeats are ignored and unrelated JSON does not extend the quiet window."""
    poll = [ObservedResponse(url="https://echo360.ca/api/heartbeat?t=1", status=200,
                             content_type="application/json", body=b'{"ok": true}')]
    analytics = [ObservedResponse(url="https://echo360.ca/api/analytics/view", status=200,
                                  content_type="application/json", body=b'{"seen": 1}')]
    events = {2.0: [_playlist(1)], 2.5: analytics,
              **{2.0 + i: poll for i in range(1, 60)}}
    state, completion, elapsed = _run(events)
    assert completion.reason == "complete" and elapsed < 6
    urls = [u for u, _ in state.json_docs]
    assert len([u for u in urls if "heartbeat" in u]) == 1  # repeats not stored again


def test_timeout_without_media():
    state, completion, elapsed = _run({1.0: [_metadata_response()]}, timeout=30)
    assert completion.reason == "timeout" and elapsed >= 30
    assert not state.successful_playlists()


def test_prefetch_of_other_lessons_rejected():
    state, completion, _ = _run({
        1.0: [_playlist(1, lesson=OTHER_LESSON, media=OTHER_MEDIA)],
        2.0: [_playlist(1)],
        2.5: [_playlist(2, lesson=OTHER_LESSON, media=OTHER_MEDIA)],
    })
    assert completion.reason == "complete"
    assert len(state.playlists) == 1 and state.rejected_playlists == 2
    assert state.other_lessons == {OTHER_LESSON}
    assert state.observed_media_ids == {MEDIA}


def test_playlist_without_lesson_param_correlated_by_media_id():
    clock = Clock()
    state = CaptureState(LESSON, clock=clock)
    bare = f"https://content.echo360.ca/0000.00000000-1111-4222-8333-444444444444/{MEDIA}/1/s1_av.m3u8"
    other = bare.replace(MEDIA, OTHER_MEDIA)
    state.add(ObservedResponse(url=bare, status=200, content_type="application/x-mpegurl"))
    state.add(ObservedResponse(url=other, status=200, content_type="application/x-mpegurl"))
    assert not state.playlists and len(state.uncorrelated_playlists) == 2
    state.add(_metadata_response())  # names MEDIA for this lesson
    state.finalize_uncorrelated()
    assert [p.source_index for p in state.playlists.values()] == [1]
    assert len(state.uncorrelated_playlists) == 1


def test_json_correlation_uses_lesson_or_media_id():
    clock = Clock()
    state = CaptureState(LESSON, clock=clock)
    state.add(ObservedResponse(url="https://echo360.ca/api/user", status=200,
                               content_type="application/json", body=b'{"name": "someone"}'))
    state.add(ObservedResponse(url="https://echo360.ca/api/m", status=200,
                               content_type="application/json",
                               body=json.dumps({"mediaId": MEDIA}).encode()))
    state.add(ObservedResponse(url="https://foreign.example/api", status=200,
                               content_type="application/json",
                               body=json.dumps({"id": LESSON}).encode()))
    state.add(_playlist(1))
    urls = [u for u, _ in state.correlated_json()]
    assert urls == ["https://echo360.ca/api/m"]


def test_play_heuristic_is_bounded_and_labelled():
    clicks = []

    def play():
        clicks.append(1)
        return "button labelled 'Play'"

    state, completion, _ = _run({12.0: [_playlist(1)]}, play=play)
    assert clicks == [1] and completion.reason == "complete"
    note = next(n for n in state.notes if "Heuristic" in n.message)
    assert not note.observed and "Play" in note.message


def test_play_heuristic_not_used_when_media_appears():
    clicks = []
    _run({1.0: [_playlist(1)]}, play=lambda: clicks.append(1))
    assert clicks == []


def test_forbidden_playlist_in_browser_does_not_count_as_found():
    clock = Clock()
    state = CaptureState(LESSON, clock=clock)
    state.add(ObservedResponse(url=playlist_url(2), status=403,
                               content_type="application/x-mpegurl"))
    assert state.first_playlist_at is None and state.accounted_sources() == set()
    assert len(state.playlists) == 1
