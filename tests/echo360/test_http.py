from __future__ import annotations

import time

import httpx
import pytest

from echo360_dl.echo360.http import MAX_REDIRECTS, HttpClient, classify_status
from echo360_dl.models import AccessState, AuthContext, Cookie

from .conftest import FAKE_JWT, FAKE_SIGNATURE, LESSON, playlist_url


def _auth(expires_at: float | None = None) -> AuthContext:
    return AuthContext(
        cookies=[Cookie("CloudFront-Signature", FAKE_SIGNATURE, "content.echo360.ca"),
                 Cookie("ECHO_JWT", FAKE_JWT, ".echo360.ca"),
                 Cookie("PATHED", "pathvalue123", "echo360.ca", path="/api")],
        headers={"User-Agent": "UA/1", "Referer": "https://echo360.ca/",
                 "Authorization": "Bearer nope"},
        expires_at=expires_at)


def _client(handler, **kwargs) -> HttpClient:
    kwargs.setdefault("sleep", lambda s: None)
    return HttpClient(kwargs.pop("auth", _auth()), transport=httpx.MockTransport(handler),
                      **kwargs)


@pytest.mark.parametrize(("status", "expired", "state"), [
    (200, False, AccessState.AVAILABLE), (206, False, AccessState.AVAILABLE),
    (401, False, AccessState.UNAUTHORIZED), (403, False, AccessState.FORBIDDEN),
    (403, True, AccessState.EXPIRED), (404, False, AccessState.NOT_FOUND),
    (410, False, AccessState.NOT_FOUND), (500, False, AccessState.TRANSIENT),
    (503, False, AccessState.TRANSIENT), (429, False, AccessState.TRANSIENT),
    (400, False, AccessState.UNKNOWN)])
def test_classify_status(status, expired, state):
    assert classify_status(status, auth_expired=expired) is state


def test_cookie_header_scoped_per_url():
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[request.url.host + request.url.path] = request.headers.get("cookie")
        assert "authorization" not in request.headers
        assert request.headers["user-agent"] == "UA/1"
        return httpx.Response(200, text="ok")

    client = _client(handler)
    assert client.get(playlist_url(1)).ok
    assert client.get("https://echo360.ca/api/x").ok
    assert client.get("https://echo360.ca/apix").ok
    content = seen["content.echo360.ca" + playlist_url(1).split("echo360.ca", 1)[1].split("?")[0]]
    assert FAKE_SIGNATURE in content and FAKE_JWT in content and "PATHED" not in content
    assert seen["echo360.ca/api/x"] == f"ECHO_JWT={FAKE_JWT}; PATHED=pathvalue123"
    assert seen["echo360.ca/apix"] == f"ECHO_JWT={FAKE_JWT}"


def test_redirect_recomputes_cookies_and_never_leaks_to_foreign_host():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "content.echo360.ca":
            return httpx.Response(302, headers={"location": "https://echo360.ca/media/x"})
        if request.url.path == "/media/x":
            return httpx.Response(302, headers={"location": "https://cdn.foreign.example/x"})
        return httpx.Response(200, text="should not be reached")

    result = _client(handler).get(playlist_url(1))
    hosts = [r.url.host for r in requests]
    assert hosts == ["content.echo360.ca", "echo360.ca"]  # foreign hop not followed
    assert FAKE_SIGNATURE in requests[0].headers["cookie"]
    assert requests[1].headers["cookie"] == f"ECHO_JWT={FAKE_JWT}"  # host-only cookie dropped
    assert result.state is AccessState.UNKNOWN
    assert "not followed" in result.detail and "?" not in result.detail


def test_foreign_hop_receives_no_cookie_when_followed_by_permissive_policy():
    """Even if a caller allows foreign hosts, the Cookie header is computed per hop."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "content.echo360.ca":
            return httpx.Response(307, headers={"location": "https://cdn.foreign.example/x"})
        return httpx.Response(200, text="ok", headers={"set-cookie": "tracker=abc123456"})

    client = _client(handler, allowed=lambda url: url.startswith("https://"))
    assert client.get(playlist_url(1)).ok
    assert requests[1].url.host == "cdn.foreign.example"
    assert "cookie" not in requests[1].headers
    # Set-Cookie from the response is not stored or replayed.
    client.get(playlist_url(1))
    assert "tracker" not in requests[2].headers.get("cookie", "")


def test_redirect_to_login_is_unauthorized():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "echo360.ca" and request.url.path == "/api/x":
            return httpx.Response(302, headers={"location": "/login?returnTo=%2Fapi%2Fx"})
        return httpx.Response(200)

    result = _client(handler).get("https://echo360.ca/api/x")
    assert result.state is AccessState.UNAUTHORIZED and "sign-in" in result.detail


def test_http_downgrade_not_followed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(301, headers={"location": "http://content.echo360.ca/x"})

    result = _client(handler).get(playlist_url(1))
    assert result.state is AccessState.UNKNOWN


def test_redirect_limit():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(302, headers={"location": f"https://echo360.ca/r{len(calls)}"})

    result = _client(handler).get("https://echo360.ca/start")
    assert len(calls) == MAX_REDIRECTS + 1
    assert "too many redirects" in result.detail


def test_403_forbidden_vs_expired():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="<Error>AccessDenied</Error>")

    fresh = _client(handler, auth=_auth(expires_at=time.time() + 3600)).get(playlist_url(3))
    assert fresh.state is AccessState.FORBIDDEN
    assert "does not prove absence" in fresh.detail
    assert "x-uid" not in fresh.detail and LESSON not in fresh.detail.split("/lesson")[0][:0]
    stale = _client(handler, auth=_auth(expires_at=time.time() - 10)).get(playlist_url(3))
    assert stale.state is AccessState.EXPIRED


def test_bounded_retries_with_backoff():
    sleeps: list[float] = []
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503)

    result = _client(handler, sleep=sleeps.append).get(playlist_url(1))
    assert result.state is AccessState.TRANSIENT
    assert len(calls) == 4  # first attempt + 3 retries
    assert sleeps == [0.5, 1.0, 2.0]


def test_retry_then_success_and_timeouts():
    calls = []

    def flaky(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectTimeout("timed out", request=request)
        if len(calls) == 2:
            raise httpx.ConnectError("reset", request=request)
        return httpx.Response(200, text="#EXTM3U")

    result = _client(flaky).get(playlist_url(1))
    assert result.ok and result.text == "#EXTM3U" and len(calls) == 3

    def always_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    result = _client(always_timeout).get(playlist_url(1))
    assert result.state is AccessState.TRANSIENT and "timed out" in result.detail
    assert "?" not in result.detail


def test_non_retryable_states_are_not_retried():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404)

    assert _client(handler).get(playlist_url(1)).state is AccessState.NOT_FOUND
    assert len(calls) == 1


def test_probe_uses_range_and_refuses_unexpected_urls():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("range"))
        return httpx.Response(206, content=b"x")

    client = _client(handler)
    assert client.probe(playlist_url(1)).state is AccessState.AVAILABLE
    assert seen == ["bytes=0-0"]
    assert client.get("https://evil.example/x").state is AccessState.UNKNOWN
    assert len(seen) == 1
