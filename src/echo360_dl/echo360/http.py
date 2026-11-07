"""httpx access with per-request cookie scoping and response classification.

Rules (see BRIEFING):
* The ``Cookie`` header is computed for every request URL from
  :meth:`AuthContext.cookies_for`; httpx's own jar is disabled (a jar would store and
  forward ``Set-Cookie`` values on its own terms).
* Redirects are followed manually (at most :data:`MAX_REDIRECTS`), recomputing cookies per
  hop. A hop to a non-https URL or off the Echo360 hosts is not followed: login/SSO pages
  classify as UNAUTHORIZED, anything else as UNKNOWN.
* 200/206 AVAILABLE, 401 UNAUTHORIZED, 403 EXPIRED (if the auth is known to be past expiry)
  else FORBIDDEN (absent *or* refused - never "absent"), 404/410 NOT_FOUND, 408/429/5xx,
  timeouts and connection errors TRANSIENT with bounded exponential-backoff retries.
"""

from __future__ import annotations

import http.cookiejar
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from echo360_dl.echo360.urls import is_echo360_url, looks_like_login_page
from echo360_dl.models import REPLAYABLE_HEADERS, AccessState, AuthContext
from echo360_dl.redact import public_url, scrub

MAX_REDIRECTS = 5
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 0.5
DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
MAX_BODY_BYTES = 16 * 1024 * 1024  # playlists, JSON, subtitles - never media segments
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})

#: Called with (method, url, status, content_type, body) after every completed hop.
Observer = Callable[[str, str, int | None, str | None, bytes | None], None]


@dataclass
class FetchResult:
    """Outcome of a request. ``detail`` never contains secrets (public URLs only)."""

    url: str  # final URL (may carry a query - do not display directly)
    state: AccessState
    status: int | None = None
    content: bytes = field(default=b"", repr=False)
    content_type: str | None = None
    detail: str = ""
    redirects: int = 0

    @property
    def ok(self) -> bool:
        return self.state is AccessState.AVAILABLE

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


def classify_status(status: int, *, auth_expired: bool) -> AccessState:
    if status in (200, 206) or 200 <= status < 300:
        return AccessState.AVAILABLE
    if status == 401:
        return AccessState.UNAUTHORIZED
    if status == 403:
        return AccessState.EXPIRED if auth_expired else AccessState.FORBIDDEN
    if status in (404, 410):
        return AccessState.NOT_FOUND
    if status in (408, 425, 429) or 500 <= status < 600:
        return AccessState.TRANSIENT
    return AccessState.UNKNOWN


def _describe(state: AccessState, status: int | None, url: str) -> str:
    where = public_url(url)
    if status == 403 and state is AccessState.FORBIDDEN:
        return f"HTTP 403 from {where} (could not access; 403 does not prove absence)"
    if status == 403 and state is AccessState.EXPIRED:
        return f"HTTP 403 from {where} (authorization has expired)"
    return f"HTTP {status} from {where}" if status is not None else where


class _NoCookies(http.cookiejar.CookieJar):
    """A jar that never stores anything."""

    def __init__(self) -> None:
        super().__init__(policy=http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))

    def set_cookie(self, cookie: http.cookiejar.Cookie) -> None:
        return None

    def extract_cookies(self, response: object, request: object) -> None:
        return None


class HttpClient:
    """Scoped, bounded HTTP access for one :class:`AuthContext`."""

    def __init__(self, auth: AuthContext, *, transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep, retries: int = DEFAULT_RETRIES,
                 backoff: float = DEFAULT_BACKOFF,
                 timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
                 allowed: Callable[[str], bool] = is_echo360_url,
                 observer: Observer | None = None) -> None:
        self.auth = auth
        self._sleep = sleep
        self._retries = max(0, retries)
        self._backoff = backoff
        self._allowed = allowed
        self.observer = observer
        self._client = httpx.Client(transport=transport, follow_redirects=False,
                                    timeout=timeout, cookies=_NoCookies(), trust_env=False)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _headers(self, url: str, extra: dict[str, str] | None) -> dict[str, str]:
        headers = {k: v for k, v in self.auth.headers.items() if k in REPLAYABLE_HEADERS}
        if extra:
            headers.update(extra)
        headers.pop("cookie", None)
        headers.pop("Cookie", None)
        cookie = self.auth.cookie_header(url)
        if cookie:
            headers["Cookie"] = cookie
        return headers

    def get(self, url: str, *, headers: dict[str, str] | None = None,
            max_bytes: int = MAX_BODY_BYTES, read_body: bool = True) -> FetchResult:
        """GET with bounded retries for TRANSIENT outcomes."""
        attempt = 0
        while True:
            result = self._get_once(url, headers=headers, max_bytes=max_bytes,
                                    read_body=read_body)
            if result.state is not AccessState.TRANSIENT or attempt >= self._retries:
                if result.state is AccessState.TRANSIENT and attempt:
                    result.detail += f" (after {attempt + 1} attempts)"
                return result
            self._sleep(self._backoff * (2 ** attempt))
            attempt += 1

    def probe(self, url: str) -> FetchResult:
        """Check access cheaply: GET the first byte only (206 or 200 = available)."""
        return self.get(url, headers={"Range": "bytes=0-0"}, read_body=False)

    def _get_once(self, url: str, *, headers: dict[str, str] | None, max_bytes: int,
                  read_body: bool) -> FetchResult:
        current = url
        if not self._allowed(current):
            return FetchResult(url=current, state=AccessState.UNKNOWN,
                               detail=f"refusing to fetch unexpected URL {public_url(current)}")
        for hop in range(MAX_REDIRECTS + 1):
            try:
                with self._client.stream("GET", current,
                                         headers=self._headers(current, headers)) as resp:
                    status = resp.status_code
                    ctype = resp.headers.get("content-type")
                    location = resp.headers.get("location")
                    body = b""
                    if status in _REDIRECT_CODES and location:
                        self._notify(current, status, ctype, None)
                    else:
                        if read_body or status < 300:
                            body = self._read(resp, max_bytes if read_body else 1)
                        self._notify(current, status, ctype, body if read_body else None)
            except httpx.TimeoutException:
                return FetchResult(url=current, state=AccessState.TRANSIENT, redirects=hop,
                                   detail=f"timed out fetching {public_url(current)}")
            except httpx.TransportError as exc:
                return FetchResult(url=current, state=AccessState.TRANSIENT, redirects=hop,
                                   detail=f"network error fetching {public_url(current)}: "
                                          f"{scrub(type(exc).__name__)}")
            except _TooLarge:
                return FetchResult(url=current, state=AccessState.UNSUPPORTED, redirects=hop,
                                   detail=f"response from {public_url(current)} is too large")
            if status in _REDIRECT_CODES and location:
                nxt = httpx.URL(current).join(location)
                target = str(nxt)
                if looks_like_login_page(target):
                    return FetchResult(url=current, state=AccessState.UNAUTHORIZED,
                                       status=status, redirects=hop,
                                       detail=f"redirected to a sign-in page "
                                              f"({public_url(target)})")
                if not self._allowed(target):
                    return FetchResult(url=current, state=AccessState.UNKNOWN, status=status,
                                       redirects=hop,
                                       detail=f"redirect to unexpected location "
                                              f"{public_url(target)} was not followed")
                current = target
                continue
            state = classify_status(status, auth_expired=self.auth.is_expired())
            return FetchResult(url=current, state=state, status=status, content=body,
                               content_type=ctype, redirects=hop,
                               detail=_describe(state, status, current))
        return FetchResult(url=current, state=AccessState.UNKNOWN, redirects=MAX_REDIRECTS,
                           detail=f"too many redirects (> {MAX_REDIRECTS}) from {public_url(url)}")

    def _notify(self, url: str, status: int, ctype: str | None, body: bytes | None) -> None:
        if self.observer is not None:
            self.observer("GET", url, status, ctype, body)

    @staticmethod
    def _read(resp: httpx.Response, max_bytes: int) -> bytes:
        chunks: list[bytes] = []
        size = 0
        for chunk in resp.iter_bytes():
            chunks.append(chunk)
            size += len(chunk)
            if size >= max_bytes:
                if max_bytes <= 1:
                    break
                if size > max_bytes:
                    raise _TooLarge
        return b"".join(chunks)


class _TooLarge(Exception):
    pass
