from __future__ import annotations

import json
import sys
import types

import httpx
import pytest

from threech0.echo360.session import MAX_REFRESHES, Echo360Session
from threech0.errors import (
    AuthExpiredError,
    ForbiddenError,
    LoginRequiredError,
    TranscriptUnavailableError,
    ValidationError,
)
from threech0.models import (
    AccessState,
    AuthContext,
    CaptionTrack,
    Cookie,
    Cue,
    TranscriptProvenance,
)

from .conftest import FAKE_JWT, FIXTURES


def _auth(tag: str = "a") -> AuthContext:
    return AuthContext(cookies=[Cookie("ECHO_JWT", f"{FAKE_JWT}-{tag}", ".echo360.ca")])


def _session(handler=None, **kwargs) -> Echo360Session:
    handler = handler or (lambda r: httpx.Response(200, text="ok"))
    return Echo360Session(_auth(), transport=httpx.MockTransport(handler),
                          sleep=lambda s: None, **kwargs)


def test_refresh_bound_once_per_call_and_twice_per_lifetime():
    calls = []

    def refresher():
        calls.append(1)
        return _auth(str(len(calls)))

    session = _session(refresher=refresher)
    first = session.refresh()
    assert len(calls) == 1 and session.auth is first
    session.refresh()
    assert len(calls) == MAX_REFRESHES == 2
    with pytest.raises(AuthExpiredError) as info:
        session.refresh()
    assert len(calls) == 2  # the refresher is not called a third time
    assert info.value.hint and "auth login" in info.value.hint


def test_refresh_failure_is_auth_expired_and_counts():
    calls = []

    def refresher():
        calls.append(1)
        raise LoginRequiredError("not signed in", hint="Run: 3ch0 auth login")

    session = _session(refresher=refresher)
    with pytest.raises(AuthExpiredError, match="not signed in"):
        session.refresh()
    assert len(calls) == 1


def test_refresh_with_third_party_error_is_scrubbed():
    def refresher():
        raise RuntimeError(f"boom Cookie: ECHO_JWT={FAKE_JWT}")

    with pytest.raises(AuthExpiredError) as info:
        _session(refresher=refresher).refresh()
    assert FAKE_JWT not in info.value.message


def test_request_file_session_cannot_refresh():
    session = _session(refresher=None, no_refresh_reason="came from a saved request file")
    with pytest.raises(AuthExpiredError, match="saved request file"):
        session.refresh()


def test_refreshed_auth_used_for_later_requests():
    cookies = []

    def handler(request):
        cookies.append(request.headers.get("cookie"))
        return httpx.Response(200)

    session = _session(handler, refresher=lambda: _auth("new"))
    session.probe("https://echo360.ca/x")
    session.refresh()
    session.probe("https://echo360.ca/x")
    assert cookies[0].endswith("-a") and cookies[1].endswith("-new")


def test_probe_states():
    def handler(request):
        return httpx.Response({"/ok": 206, "/no": 403, "/gone": 404}[request.url.path])

    session = _session(handler)
    assert session.probe("https://content.echo360.ca/ok") is AccessState.AVAILABLE
    assert session.probe("https://content.echo360.ca/no") is AccessState.FORBIDDEN
    assert session.probe("https://content.echo360.ca/gone") is AccessState.NOT_FOUND


def _track(**formats) -> CaptionTrack:
    return CaptionTrack(id="t1", language="en", kind=TranscriptProvenance.PLATFORM_TRANSCRIPT,
                        formats=formats, label="English")


def test_fetch_json_transcript_keeps_raw():
    body = (FIXTURES / "transcript_cues.json").read_bytes()
    session = _session(lambda r: httpx.Response(200, content=body,
                                                headers={"content-type": "application/json"}))
    t = session.fetch_transcript(_track(json="https://echo360.ca/api/t"))
    assert t.raw == body and t.raw_format == "json"
    assert t.provenance is TranscriptProvenance.PLATFORM_TRANSCRIPT
    assert t.cues[0] == Cue(text="Welcome to the fictional lecture.", start=0.0, end=4.5,
                            speaker="Speaker 1")
    assert t.cues[2].start == 9.0 and t.cues[2].end is None  # no invented end time
    assert not t.timed


def test_fetch_untimed_txt_never_invents_times():
    session = _session(lambda r: httpx.Response(200, text="First paragraph.\n\nSecond one.\n"))
    t = session.fetch_transcript(_track(txt="https://echo360.ca/t.txt"), "txt")
    assert [c.text for c in t.cues] == ["First paragraph.", "Second one."]
    assert all(c.start is None and c.end is None for c in t.cues)
    assert not t.timed


def test_fetch_vtt_delegates_to_media_parser(monkeypatch):
    calls = []

    def parse_subtitle_bytes(data, fmt):
        calls.append((data, fmt))
        return [Cue(text="hi", start=1.0, end=2.0)]

    fake = types.ModuleType("threech0.media.subtitles")
    fake.parse_subtitle_bytes = parse_subtitle_bytes
    monkeypatch.setitem(sys.modules, "threech0.media.subtitles", fake)
    vtt = b"WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhi\n"
    session = _session(lambda r: httpx.Response(200, content=vtt))
    track = _track(vtt="https://captions.echo360.ca/x.vtt", json="https://echo360.ca/j")
    t = session.fetch_transcript(track)  # vtt preferred
    assert calls == [(vtt, "vtt")] and t.raw == vtt and t.cues[0].text == "hi"


def test_fetch_transcript_errors():
    session = _session(lambda r: httpx.Response(403))
    with pytest.raises(ForbiddenError, match="does not prove absence"):
        session.fetch_transcript(_track(json="https://echo360.ca/j"))
    with pytest.raises(TranscriptUnavailableError, match="available: json"):
        session.fetch_transcript(_track(json="https://echo360.ca/j"), "vtt")
    garbage = _session(lambda r: httpx.Response(200, text="not json"))
    with pytest.raises(ValidationError):
        garbage.fetch_transcript(_track(json="https://echo360.ca/j"))
    nocues = _session(lambda r: httpx.Response(200, text=json.dumps({"a": 1})))
    with pytest.raises(ValidationError):
        nocues.fetch_transcript(_track(json="https://echo360.ca/j"))
