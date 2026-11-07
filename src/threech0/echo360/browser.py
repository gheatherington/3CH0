"""Playwright (sync API) browser handling: login, status checks and traffic capture.

The browser uses a dedicated persistent profile (default ``paths.browser_profile_dir()``,
mode 0700), never the user's own browser profile. Credentials are never typed by this
code: when sign-in is needed a visible window is opened (interactive runs only) and the
user completes SSO/MFA themselves. Non-interactive runs never open a window.

The capture logic itself lives in :mod:`threech0.echo360.capture` (browser-free and unit
tested); this module only connects it to Playwright.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from threech0 import paths
from threech0.echo360.capture import (
    MAX_BODY,
    CaptureState,
    Completion,
    ObservedResponse,
    is_playlist_response,
    run_capture_loop,
)
from threech0.echo360.curl_import import cloudfront_policy_expiry
from threech0.echo360.trace import TraceRecorder
from threech0.echo360.urls import is_echo360_host, is_echo360_url, is_login_url
from threech0.errors import AuthError, DependencyError, LoginRequiredError, UsageError
from threech0.models import (
    REPLAYABLE_HEADERS,
    AuthContext,
    AuthStatus,
    ClassroomRef,
    Cookie,
    DiscoveryNote,
    ProgressSink,
)
from threech0.redact import public_url, scrub

if TYPE_CHECKING:
    from playwright.sync_api import BrowserContext, Page, Playwright

HOME_URL = "https://echo360.ca/"
#: Hypothesis: the signed-in home lists courses at /courses (not verified on echo360.ca;
#: a sign-in redirect from it is what check_status relies on).
COURSES_URL = "https://echo360.ca/courses"
LOGIN_HINT = "Run: 3ch0 auth login"
# The tool's own interpreter guarantees a Chromium build matching the installed Playwright.
INSTALL_HINT = ("Install the browser with: "
                f"{shlex.quote(sys.executable)} -m playwright install chromium")
_PLAY_SELECTORS = (
    ("role=button[name=/^play/i]", "button labelled 'Play'"),
    ("button[aria-label*='play' i]", "button with aria-label containing 'play'"),
    ("[class*='play-button' i], [class*='playButton' i]", "element with a play-button class"),
    ("video", "the video element"),
)


@dataclass
class CaptureOutcome:
    state: CaptureState
    completion: Completion
    auth: AuthContext = field(repr=False)
    html: str | None = field(default=None, repr=False)
    notes: list[DiscoveryNote] = field(default_factory=list)


# -- installation / launching ---------------------------------------------------------------


def browser_installed() -> tuple[bool, str]:
    """Is Playwright's Chromium installed? Starts the Playwright driver but no browser."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False, "Playwright is not installed (pip/uv package 'playwright')."
    try:
        with sync_playwright() as pw:
            executable = Path(pw.chromium.executable_path)
    except Exception as exc:
        return False, f"Playwright could not start: {scrub(str(exc))[:200]}. {INSTALL_HINT}"
    if executable.exists():
        return True, f"Chromium found at {executable}"
    return False, f"Chromium is not installed (expected {executable}). {INSTALL_HINT}"


def _profile(profile_dir: Path | None) -> Path:
    try:
        return paths.prepare_profile_dir(Path(profile_dir or paths.browser_profile_dir()))
    except ValueError:
        raise UsageError(
            f"{profile_dir} already contains files that 3ch0 did not create; refusing "
            "to use it as the browser profile (it may be your everyday browser's profile).",
            hint="Pass an empty or new directory to --profile-dir, or omit it.") from None


#: Echo360 cookies saved by 3ch0 itself. OBSERVED (2026-11-07): a successful
#: ``auth login`` was not visible to the next launch of the same profile, because Chromium
#: does not persist *session* cookies (no expiry) across restarts. Mode 0600 inside the
#: 0700 profile; only cookies for echo360.ca hosts are kept.
SESSION_FILE = "3ch0-session.json"

_PROFILES: dict[int, Path] = {}  # id(context) -> profile dir, for saving on close


def _session_path(profile_dir: Path) -> Path:
    return Path(profile_dir) / SESSION_FILE


def save_session_cookies(cookies: list[dict[str, Any]], profile_dir: Path,
                         now: float | None = None) -> int:
    """Persist current echo360.ca cookies (including session cookies). Returns count."""
    now = time.time() if now is None else now
    keep = []
    for c in cookies:
        domain = str(c.get("domain", "")).lstrip(".").lower()
        expires = c.get("expires")
        if not is_echo360_host(domain):
            continue
        if isinstance(expires, int | float) and 0 < expires <= now:
            continue
        keep.append({k: c[k] for k in ("name", "value", "domain", "path", "expires",
                                        "httpOnly", "secure", "sameSite") if k in c})
    path = _session_path(profile_dir)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"version": 1, "saved_at": now, "cookies": keep}, handle)
    os.replace(tmp, path)
    return len(keep)


def load_session_cookies(profile_dir: Path, now: float | None = None) -> list[dict[str, Any]]:
    """Saved echo360.ca cookies that have not expired (empty if none/unreadable)."""
    now = time.time() if now is None else now
    try:
        data = json.loads(_session_path(profile_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for c in data.get("cookies", []) if isinstance(data, dict) else []:
        if not isinstance(c, dict) or not {"name", "value", "domain", "path"} <= set(c):
            continue
        if not is_echo360_host(str(c["domain"]).lstrip(".").lower()):
            continue
        expires = c.get("expires")
        if isinstance(expires, int | float) and 0 < expires <= now:
            continue
        out.append(c)
    return out


def _restore_session(context: BrowserContext, profile_dir: Path) -> None:
    cookies = load_session_cookies(profile_dir)
    if cookies:
        with contextlib.suppress(Exception):
            context.add_cookies(cookies)


def _save_session(context: BrowserContext) -> None:
    profile = _PROFILES.pop(id(context), None)
    if profile is None:
        return
    with contextlib.suppress(Exception):
        save_session_cookies(context.cookies(), profile)


def _launch(pw: Playwright, profile_dir: Path, *, headless: bool) -> BrowserContext:
    try:
        context = pw.chromium.launch_persistent_context(
            str(profile_dir), headless=headless, accept_downloads=False,
            viewport=None if not headless else {"width": 1280, "height": 800})
        _restore_session(context, profile_dir)
        _PROFILES[id(context)] = Path(profile_dir)
        return context
    except Exception as exc:
        text = scrub(str(exc))
        if "Executable doesn't exist" in text or "playwright install" in text:
            raise DependencyError("Playwright's Chromium browser is not installed.",
                                  hint=INSTALL_HINT) from None
        if "SingletonLock" in text or "ProcessSingleton" in text or "already in use" in text:
            raise AuthError("The 3ch0 browser profile is in use by another window.",
                            hint="Close other 3ch0 browser windows and retry.") from None
        raise DependencyError(f"Could not start the browser: {text[:300]}",
                              hint=INSTALL_HINT) from None


@contextlib.contextmanager
def _playwright() -> Iterator[Playwright]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise DependencyError("Playwright is not installed.", hint=INSTALL_HINT) from None
    with sync_playwright() as pw:
        yield pw


def _close(context: BrowserContext | None) -> None:
    if context is not None:
        _save_session(context)  # before close: cookies are unreadable afterwards
        with contextlib.suppress(Exception):
            context.close()


def _main_page(context: BrowserContext) -> Page:
    pages = [p for p in context.pages if not p.is_closed()]
    return pages[-1] if pages else context.new_page()


def _open_page_urls(context: BrowserContext) -> list[str]:
    return [p.url for p in context.pages if not p.is_closed()]


# -- status / login --------------------------------------------------------------------------


#: Seconds to wait on a non-login page for a session cookie before giving up.
UNCONFIRMED_GRACE = 60.0

#: Cookie names that indicate an authenticated echo360.ca session.
#: ECHO_JWT: OBSERVED in a signed-in request (2026-11-06). PLAY_SESSION: HYPOTHESIS
#: (common Play-framework session cookie name); never sufficient on its own to fail.
SESSION_COOKIES = ("ECHO_JWT", "PLAY_SESSION")


def session_evidence(cookies: list[dict[str, Any]], now: float | None = None) -> str | None:
    """Name of a current echo360.ca session cookie, or None. Values are never read."""
    now = time.time() if now is None else now
    for cookie in cookies:
        domain = str(cookie.get("domain", "")).lstrip(".").lower()
        if cookie.get("name") not in SESSION_COOKIES or not is_echo360_host(domain):
            continue
        expires = cookie.get("expires")
        if isinstance(expires, int | float) and 0 < expires <= now:
            continue
        return str(cookie["name"])
    return None


def _context_session_evidence(context: BrowserContext) -> str | None:
    try:
        return session_evidence(context.cookies(["https://echo360.ca/"]))
    except Exception:
        return None


def check_status(profile_dir: Path, *, timeout: float) -> AuthStatus:
    """Headless check whether the saved profile is signed in to echo360.ca.

    ``logged_in`` is True when the courses page loads on echo360.ca without a sign-in
    redirect, False when it redirects to sign-in or no profile exists, None if unknown.
    """
    profile = Path(profile_dir or paths.browser_profile_dir())
    exists = profile.is_dir() and any(profile.iterdir())
    if not exists:
        return AuthStatus(profile_dir=profile, profile_exists=False, logged_in=False,
                          detail=f"No saved browser session. {LOGIN_HINT}")
    deadline = time.monotonic() + max(1.0, timeout)
    try:
        with _playwright() as pw:
            context = _launch(pw, _profile(profile), headless=True)
            try:
                page = _main_page(context)
                remaining = max(1.0, deadline - time.monotonic())
                try:
                    response = page.goto(COURSES_URL, wait_until="domcontentloaded",
                                         timeout=remaining * 1000)
                except Exception as exc:
                    if "Timeout" in type(exc).__name__ or "Timeout" in str(exc):
                        return AuthStatus(profile, True, None,
                                          f"Timed out after {timeout:.0f} s checking sign-in.")
                    return AuthStatus(profile, True, None,
                                      f"Could not load echo360.ca: {scrub(str(exc))[:200]}")
                # Give client-side redirects a moment (bounded by the overall timeout).
                settle_until = min(deadline, time.monotonic() + 3.0)
                while time.monotonic() < settle_until and not is_login_url(page.url):
                    page.wait_for_timeout(250)
                final = page.url
                status = response.status if response is not None else None
                if is_login_url(final):
                    return AuthStatus(profile, True, False,
                                      f"Not signed in (redirected to {public_url(final)}). "
                                      f"{LOGIN_HINT}")
                if status is not None and status >= 400:
                    return AuthStatus(profile, True, None,
                                      f"echo360.ca answered HTTP {status}; sign-in state "
                                      "unknown.")
                evidence = _context_session_evidence(context)
                if evidence is None:
                    return AuthStatus(profile, True, None,
                                      f"Loaded {public_url(final)} without a sign-in redirect, "
                                      "but no Echo360 session cookie was found; sign-in could "
                                      f"not be confirmed. {LOGIN_HINT}")
                return AuthStatus(profile, True, True,
                                  f"Signed in (loaded {public_url(final)}; session cookie "
                                  f"{evidence} present).")
            finally:
                _close(context)
    except (DependencyError, AuthError) as exc:
        return AuthStatus(profile, True, None, exc.message)


def login(profile_dir: Path, *, start_url: str | None, timeout: float,
          progress: ProgressSink) -> AuthStatus:
    """Open a visible window; wait (bounded) until the user lands on a non-login
    echo360.ca page, then close it. Credentials are entered only by the user."""
    profile = _profile(profile_dir)
    target = start_url or COURSES_URL
    if not is_echo360_url(target):
        target = COURSES_URL
    deadline = time.monotonic() + max(5.0, timeout)
    with _playwright() as pw:
        context = _launch(pw, profile, headless=False)
        try:
            page = _main_page(context)
            progress.info("A browser window has opened. Sign in to Echo360 there "
                          f"(waiting up to {timeout:.0f} s).")
            with contextlib.suppress(Exception):
                page.goto(target, wait_until="domcontentloaded", timeout=60_000)
            saw_login = False
            unconfirmed_since = 0.0
            while time.monotonic() < deadline:
                urls = _open_page_urls(context)
                if not urls:
                    return AuthStatus(profile, True, None,
                                      "The browser window was closed before sign-in finished.")
                if any(is_login_url(u) for u in urls):
                    saw_login = True
                done = [u for u in urls if is_echo360_url(u) and not is_login_url(u)]
                if done and not any(is_login_url(u) for u in urls):
                    # Let the post-login page finish setting cookies before deciding.
                    with contextlib.suppress(Exception):
                        _main_page(context).wait_for_load_state("networkidle", timeout=5_000)
                    evidence = _context_session_evidence(context)
                    if evidence is not None:
                        detail = ("Signed in." if saw_login else "Already signed in.")
                        progress.info(detail)
                        return AuthStatus(profile, True, True,
                                          f"{detail} Landed on {public_url(done[-1])}; "
                                          f"session cookie {evidence} present.")
                    if not unconfirmed_since:
                        unconfirmed_since = time.monotonic()
                        progress.info("Waiting for Echo360 to finish signing you in "
                                      "(sign in in the window if a form is shown).")
                    elif time.monotonic() - unconfirmed_since > UNCONFIRMED_GRACE:
                        # A page that is not a login URL but no session cookie: an in-page
                        # sign-in form, or an unknown cookie name. Do not claim success.
                        return AuthStatus(
                            profile, True, None,
                            f"Reached {public_url(done[-1])}, but no Echo360 session cookie "
                            "appeared, so sign-in could not be confirmed. If you completed "
                            "sign-in, run `3ch0 auth status`; otherwise run "
                            "`3ch0 auth login` again.")
                else:
                    unconfirmed_since = 0.0
                with contextlib.suppress(Exception):
                    _main_page(context).wait_for_timeout(500)
            return AuthStatus(profile, True, False,
                              f"Sign-in was not completed within {timeout:.0f} s.")
        finally:
            _close(context)


# -- capture ----------------------------------------------------------------------------------


def _wants_body(url: str, content_type: str | None, headers: dict[str, str]) -> bool:
    if not is_echo360_url(url):
        return False
    try:
        if int(headers.get("content-length", "0")) > MAX_BODY:
            return False
    except ValueError:
        pass
    ctype = (content_type or "").lower()
    return (is_playlist_response(url, content_type) or "json" in ctype or "vtt" in ctype
            or "subrip" in ctype or url.split("?", 1)[0].lower().endswith((".vtt", ".srt")))


def _handler(state: CaptureState, trace: TraceRecorder | None) -> Callable[[Any], None]:
    def on_response(response: Any) -> None:
        try:
            url = response.url
            headers = {k.lower(): v for k, v in (response.headers or {}).items()}
            ctype = headers.get("content-type")
            request = response.request
            body = None
            status = response.status
            if _wants_body(url, ctype, headers) and status < 300:
                with contextlib.suppress(Exception):
                    body = response.body()
            req_headers: dict[str, str] = {}
            if is_playlist_response(url, ctype):
                with contextlib.suppress(Exception):
                    req_headers = {k.lower(): v for k, v in request.headers.items()
                                   if k.lower() in REPLAYABLE_HEADERS}
            state.add(ObservedResponse(url=url, status=status, content_type=ctype, body=body,
                                       method=request.method,
                                       resource_type=request.resource_type,
                                       request_headers=req_headers))
            if trace is not None and is_echo360_url(url):
                trace.record(request.method, url, status, ctype, body,
                             resource_type=request.resource_type)
        except Exception:  # never let a handler failure break the page
            state.notes.append(DiscoveryNote("warning", "A network response could not be "
                                                        "inspected."))
    return on_response


def _cookies(context: BrowserContext) -> list[Cookie]:
    out: list[Cookie] = []
    for c in context.cookies():
        domain = str(c.get("domain") or "")
        if not domain or not is_echo360_host(domain.lstrip(".")):
            continue
        expires = c.get("expires")
        out.append(Cookie(name=str(c["name"]), value=str(c["value"]), domain=domain,
                          path=str(c.get("path") or "/"), secure=bool(c.get("secure")),
                          http_only=bool(c.get("httpOnly")),
                          expires=float(expires) if expires and expires > 0 else None))
    return out


def auth_from_cookies(cookies: list[Cookie], headers: dict[str, str]) -> AuthContext:
    policy = next((c for c in cookies if c.name == "CloudFront-Policy"), None)
    expires_at = cloudfront_policy_expiry(policy.value) if policy else None
    return AuthContext(cookies=cookies, headers=headers, source="browser",
                       expires_at=expires_at)


def _click_play(page: Page) -> str | None:
    for selector, description in _PLAY_SELECTORS:
        try:
            locator = page.locator(selector).first
            if locator.count() and locator.is_visible(timeout=500):
                locator.click(timeout=2_000)
                return description
        except Exception:  # noqa: S112 - each selector is optional
            continue
    return None


def _wait_for_login(context: BrowserContext, ref: ClassroomRef, *, timeout: float,
                    progress: ProgressSink) -> None:
    progress.info("Sign-in needed: complete it in the browser window "
                  f"(waiting up to {timeout:.0f} s). 3CH0 never types credentials.")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        urls = _open_page_urls(context)
        if not urls:
            raise LoginRequiredError("The browser window was closed before sign-in finished.",
                                     hint=LOGIN_HINT)
        if not any(is_login_url(u) for u in urls) and any(
                is_echo360_url(u) for u in urls):
            progress.info("Signed in; continuing discovery.")
            page = _main_page(context)
            if ref.lesson_id not in page.url:
                with contextlib.suppress(Exception):
                    page.goto(ref.url, wait_until="domcontentloaded", timeout=60_000)
            return
        with contextlib.suppress(Exception):
            _main_page(context).wait_for_timeout(500)
    raise LoginRequiredError(f"Sign-in was not completed within {timeout:.0f} s.",
                             hint=LOGIN_HINT)


def capture_lesson(ref: ClassroomRef, *, profile_dir: Path | None, interactive: bool,
                   headless: bool | None, timeout: float, login_timeout: float,
                   progress: ProgressSink, trace: TraceRecorder | None = None
                   ) -> CaptureOutcome:
    """Load the classroom page in the dedicated profile and observe its traffic."""
    profile = _profile(profile_dir)
    visible = bool(interactive and headless is False)
    if headless is False and not interactive:
        progress.debug("Non-interactive run: the browser stays headless.")
    state = CaptureState(ref.lesson_id, clock=time.monotonic)
    deadline = time.monotonic() + timeout
    with _playwright() as pw:
        context = _launch(pw, profile, headless=not visible)
        try:
            context.on("response", _handler(state, trace))
            page = _main_page(context)
            progress.info("Opening the classroom page" + (" (headless)." if not visible
                                                          else "."))
            _goto(page, ref.url, deadline - time.monotonic())
            if _needs_login(context):
                if not interactive:
                    raise LoginRequiredError(
                        "The saved browser session is not signed in to echo360.ca.",
                        hint=LOGIN_HINT)
                if not visible:
                    _close(context)
                    context = _launch(pw, profile, headless=False)
                    context.on("response", _handler(state, trace))
                    visible = True
                    _goto(_main_page(context), ref.url, timeout)
                if _needs_login(context):
                    _wait_for_login(context, ref, timeout=login_timeout, progress=progress)
                state.restart_clock()
                deadline = time.monotonic() + timeout  # the login wait has its own bound

            active = context
            progress.task_start("discover", "Watching player traffic", total=None,
                                unit="items")

            def pump(seconds: float) -> None:
                pages = [p for p in active.pages if not p.is_closed()]
                if not pages:
                    raise LoginRequiredError("The browser window was closed during discovery.",
                                             hint="Run the command again and keep the window "
                                                  "open.")
                if _on_login_pages([p.url for p in pages]):
                    raise LoginRequiredError("The classroom page redirected to sign-in.",
                                             hint=LOGIN_HINT)
                pages[-1].wait_for_timeout(seconds * 1000)

            def on_tick() -> None:
                progress.task_update("discover", completed=len(state.playlists))

            completion = run_capture_loop(
                state, pump=pump, clock=time.monotonic, deadline=deadline,
                play=lambda: _click_play(_main_page(active)), on_tick=on_tick)
            progress.task_done("discover", ok=bool(state.successful_playlists()))
            html = None
            with contextlib.suppress(Exception):
                html = _main_page(active).content()
            auth = auth_from_cookies(_cookies(active), state.request_headers or {
                "referer": "https://echo360.ca/", "origin": "https://echo360.ca"})
            return CaptureOutcome(state=state, completion=completion, auth=auth, html=html,
                                  notes=list(state.notes))
        finally:
            _close(context)


def _goto(page: Page, url: str, timeout: float) -> None:
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=max(5.0, timeout) * 1000)
    except Exception as exc:
        if is_login_url(page.url):
            return  # the redirect chain landed on a sign-in page; handled by the caller
        raise AuthError(f"Could not load the classroom page: {scrub(str(exc))[:300]}",
                        hint="Check the URL and your network connection.") from None
    # Client-side redirects to sign-in can happen just after DOMContentLoaded.
    end = time.monotonic() + 2.0
    while time.monotonic() < end and not is_login_url(page.url):
        page.wait_for_timeout(250)


def _needs_login(context: BrowserContext) -> bool:
    return _on_login_pages(_open_page_urls(context))


def _on_login_pages(urls: list[str]) -> bool:
    """True when every open web page (ignoring about:blank and similar) is a sign-in page."""
    web = [u for u in urls if u.startswith(("http://", "https://"))]
    return bool(web) and all(is_login_url(u) for u in web)
