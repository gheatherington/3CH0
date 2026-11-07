"""Browser glue with a fake Playwright context (no real browser, no network)."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

from threech0.echo360 import browser
from threech0.echo360.urls import parse_classroom_url
from threech0.errors import LoginRequiredError

from .conftest import CLASSROOM, FAKE_JWT, LESSON, master_text, playlist_url

REF = parse_classroom_url(CLASSROOM)
LOGIN = "https://login.university.example/sso/start"


class FakeTime:
    def __init__(self) -> None:
        self.now = 100.0

    def monotonic(self) -> float:
        return self.now


class FakeResponse:
    def __init__(self, url: str, body: bytes) -> None:
        self.url = url
        self.status = 200
        self.headers = {"content-type": "application/x-mpegurl"}
        self.request = SimpleNamespace(method="GET", resource_type="fetch",
                                       headers={"referer": "https://echo360.ca/",
                                                "cookie": "MUST-NOT-BE-KEPT"})
        self._body = body

    def body(self) -> bytes:
        return self._body


class FakePage:
    def __init__(self, ctx: FakeContext) -> None:
        self.ctx = ctx
        self.url = "about:blank"

    def is_closed(self) -> bool:
        return False

    def goto(self, url: str, **kwargs) -> None:
        self.url = self.ctx.route(url)

    def wait_for_timeout(self, ms: float) -> None:
        self.ctx.clock.now += ms / 1000
        self.ctx.tick()

    def content(self) -> str:
        return "<html></html>"

    def locator(self, selector: str):
        return SimpleNamespace(first=SimpleNamespace(count=lambda: 0))


class FakeContext:
    def __init__(self, *, headless: bool, clock: FakeTime, signed_in_after: float | None,
                 cookies: list[dict]) -> None:
        self.headless = headless
        self.clock = clock
        self.signed_in_after = signed_in_after
        self.handlers: list = []
        self.pages = [FakePage(self)]
        self._cookies = cookies
        self.closed = False
        self.sent = False

    def signed_in(self) -> bool:
        return self.signed_in_after is not None and self.clock.now >= self.signed_in_after

    def route(self, url: str) -> str:
        return url if self.signed_in() else LOGIN

    def tick(self) -> None:
        page = self.pages[0]
        if page.url == LOGIN and self.signed_in():
            page.url = CLASSROOM
        if page.url == CLASSROOM and not self.sent:
            self.sent = True
            for handler in self.handlers:
                handler(FakeResponse(playlist_url(1), master_text(1).encode()))

    def on(self, event: str, handler) -> None:
        assert event == "response"
        self.handlers.append(handler)

    def new_page(self) -> FakePage:
        page = FakePage(self)
        self.pages.append(page)
        return page

    def cookies(self) -> list[dict]:
        return self._cookies

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_browser(monkeypatch, tmp_path):
    clock = FakeTime()
    launches: list[FakeContext] = []
    config = {"signed_in_after": 0.0}
    cookies = [
        {"name": "ECHO_JWT", "value": FAKE_JWT, "domain": ".echo360.ca", "path": "/",
         "secure": True, "httpOnly": True, "expires": -1},
        {"name": "CloudFront-Signature", "value": "FAKESIG", "domain": "content.echo360.ca",
         "path": "/", "secure": True, "httpOnly": False, "expires": 4_102_000_000},
        {"name": "IDP_SESSION", "value": "idp-secret", "domain": "login.university.example",
         "path": "/", "secure": True, "httpOnly": True, "expires": -1},
    ]

    def launch(pw, profile_dir, *, headless):
        ctx = FakeContext(headless=headless, clock=clock,
                          signed_in_after=config["signed_in_after"], cookies=cookies)
        launches.append(ctx)
        return ctx

    @contextlib.contextmanager
    def fake_playwright():
        yield None

    monkeypatch.setattr(browser, "_launch", launch)
    monkeypatch.setattr(browser, "_playwright", fake_playwright)
    monkeypatch.setattr(browser, "time", clock)
    return SimpleNamespace(clock=clock, launches=launches, config=config,
                           profile=tmp_path / "profile")


def _capture(fb, progress, *, interactive: bool, headless=None):
    return browser.capture_lesson(REF, profile_dir=fb.profile, interactive=interactive,
                                  headless=headless, timeout=60, login_timeout=120,
                                  progress=progress)


def test_signed_in_headless_capture(fake_browser, progress):
    out = _capture(fake_browser, progress, interactive=False)
    assert [c.headless for c in fake_browser.launches] == [True]
    assert out.completion.reason == "complete"
    assert out.state.accounted_sources() == {1}
    names = {c.name: c for c in out.auth.cookies}
    assert set(names) == {"ECHO_JWT", "CloudFront-Signature"}  # IdP cookie not carried over
    assert names["ECHO_JWT"].domain == ".echo360.ca" and names["ECHO_JWT"].expires is None
    assert names["CloudFront-Signature"].host_only
    assert names["CloudFront-Signature"].expires == 4_102_000_000
    assert out.auth.headers == {"referer": "https://echo360.ca/"}
    assert all(c.closed for c in fake_browser.launches)
    assert (fake_browser.profile.stat().st_mode & 0o777) == 0o700


def test_non_interactive_never_opens_a_window(fake_browser, progress):
    fake_browser.config["signed_in_after"] = None  # never signed in
    with pytest.raises(LoginRequiredError) as info:
        _capture(fake_browser, progress, interactive=False, headless=False)
    assert "auth login" in info.value.hint
    assert [c.headless for c in fake_browser.launches] == [True]


def test_interactive_login_reopens_visible_window(fake_browser, progress):
    fake_browser.config["signed_in_after"] = 130.0  # the user finishes SSO after ~30 s
    out = _capture(fake_browser, progress, interactive=True)
    assert [c.headless for c in fake_browser.launches] == [True, False]
    assert out.state.accounted_sources() == {1}
    assert any("Sign-in needed" in m for _k, m in progress.messages)
    assert any("never types credentials" in m for _k, m in progress.messages)


def test_interactive_login_is_bounded(fake_browser, progress):
    fake_browser.config["signed_in_after"] = None
    with pytest.raises(LoginRequiredError, match="not completed within 120 s"):
        _capture(fake_browser, progress, interactive=True)


def test_auth_context_expiry_from_policy_cookie():
    from .conftest import cloudfront_policy

    cookies = [browser.Cookie("CloudFront-Policy", cloudfront_policy(4_000_000_000),
                              ".echo360.ca")]
    auth = browser.auth_from_cookies(cookies, {})
    assert auth.expires_at == 4_000_000_000 and auth.source == "browser"
    assert LESSON not in repr(auth)


def test_session_evidence_requires_current_echo360_session_cookie():
    from threech0.echo360.browser import session_evidence

    now = 1_000_000.0
    jwt = {"name": "ECHO_JWT", "value": "never-read", "domain": ".echo360.ca",
           "expires": now + 3600}
    assert session_evidence([jwt], now) == "ECHO_JWT"
    assert session_evidence([{**jwt, "expires": now - 1}], now) is None  # expired
    assert session_evidence([{**jwt, "expires": -1}], now) == "ECHO_JWT"  # session cookie
    assert session_evidence([{**jwt, "domain": "evil.example"}], now) is None
    assert session_evidence([{**jwt, "name": "_ga"}], now) is None  # analytics only
    assert session_evidence([], now) is None


def test_session_cookies_round_trip_and_stay_private(tmp_path):
    import stat

    from threech0.echo360.browser import (
        SESSION_FILE,
        load_session_cookies,
        save_session_cookies,
    )

    now = 1_000_000.0
    cookies = [
        {"name": "PLAY_SESSION", "value": "v1", "domain": "echo360.ca", "path": "/",
         "expires": -1, "httpOnly": True, "secure": True, "sameSite": "Lax"},  # session
        {"name": "ECHO_JWT", "value": "v2", "domain": ".echo360.ca", "path": "/",
         "expires": now + 3600, "httpOnly": False, "secure": True, "sameSite": "None"},
        {"name": "old", "value": "v3", "domain": "echo360.ca", "path": "/",
         "expires": now - 1},  # expired
        {"name": "idp", "value": "v4", "domain": "login.university.example", "path": "/",
         "expires": -1},  # not an Echo360 host: never written
    ]
    assert save_session_cookies(cookies, tmp_path, now) == 2
    path = tmp_path / SESSION_FILE
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert "v4" not in path.read_text()
    restored = load_session_cookies(tmp_path, now)
    assert sorted(c["name"] for c in restored) == ["ECHO_JWT", "PLAY_SESSION"]
    later = load_session_cookies(tmp_path, now + 7200)  # ECHO_JWT has expired by then
    assert [c["name"] for c in later] == ["PLAY_SESSION"]


def test_load_session_cookies_tolerates_missing_or_corrupt_file(tmp_path):
    from threech0.echo360.browser import SESSION_FILE, load_session_cookies

    assert load_session_cookies(tmp_path) == []
    (tmp_path / SESSION_FILE).write_text("{not json")
    assert load_session_cookies(tmp_path) == []
