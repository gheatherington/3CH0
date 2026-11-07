"""Helpers that keep credentials out of output, logs and errors.

Policy:
* Never print cookie values, ``Authorization``/``Cookie`` headers or signed query
  strings. Display URLs through :func:`public_url`.
* Free text from third-party tools (yt-dlp, ffmpeg, Playwright errors) passes
  through :func:`scrub` before it is shown.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

# Names of cookies/parameters whose *values* are redacted (not credentials themselves).
_REDACTED_KEYS = (
    r"CloudFront-[A-Za-z0-9-]+|ECHO_JWT|PLAY_SESSION|JSESSIONID|SESSION[A-Za-z0-9_]*|"
    r"Policy|Signature|Key-Pair-Id|X-Amz-[A-Za-z-]+|token|access_token|id_token|jwt|"
    r"refresh_token|auth[A-Za-z_]*|sig|signature|x-uid|x-sid|x-did|x-oid|x-act|x-instid"
)

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Header lines: "Cookie: ...", "Authorization: Bearer ..."
    (re.compile(r"(?im)\b(cookie|set-cookie|authorization|proxy-authorization)\s*[:=]\s*[^\r\n]+"),
     r"\1: [REDACTED]"),
    # curl style -b 'a=b; c=d' / --cookie "..."
    (re.compile(r"""(?i)(-b|--cookie)\s+(['"]).*?\2"""), r"\1 [REDACTED]"),
    # name=value pairs for known secret names (cookies or query params)
    (re.compile(rf"(?i)\b({_REDACTED_KEYS})=([^;&\s'\"]+)"), r"\1=[REDACTED]"),
    # JWT-looking tokens
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
     "[REDACTED-JWT]"),
    # Query strings on http(s) URLs (signed parameters live there)
    (re.compile(r"(https?://[^\s?#'\"]+)\?[^\s#'\"]*"), r"\1?[…]"),
]


def public_url(url: str) -> str:
    """Return ``url`` without query string, fragment or userinfo."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "[invalid-url]"
    netloc = parts.hostname or ""
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def scrub(text: str | None) -> str:
    """Best-effort removal of secrets from free text."""
    if not text:
        return ""
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


def contains_secret(text: str, secrets: list[str]) -> bool:
    """True if any non-trivial secret value appears in ``text`` (used by tests/guards)."""
    return any(s and len(s) >= 6 and s in text for s in secrets)
