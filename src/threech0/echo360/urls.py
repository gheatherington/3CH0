"""Classroom URL parsing and host/URL classification helpers.

Observed (2026-11-06): classroom pages are ``https://echo360.ca/lesson/<id>/classroom`` and
may carry a ``?timestamp=`` query parameter, which is ignored here (the default is always the
full recording). Content is served from ``content.echo360.ca`` with ``x-lid``/``x-mid`` query
parameters correlating a request with a lesson and a media item.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, unquote, urlsplit

from threech0.errors import InvalidURLError
from threech0.models import ClassroomRef

#: Hosts whose classroom URLs are supported. Only echo360.ca has been observed; other
#: regional Echo360 domains are deliberately not claimed.
SUPPORTED_HOSTS = frozenset({"echo360.ca"})

_CLASSROOM_PATH = re.compile(r"^/lesson/([^/]+)/classroom/?$")
# Observed form: G_<uuid>_<uuid>_<YYYY-MM-DDTHH:MM:SS.mmm>_<same> (123 chars, contains ":").
_LESSON_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.~:+-]{0,399}$")
_LOGIN_SEGMENT = re.compile(
    r"(?i)^(?:login|logon|log-in|signin|sign-in|sso|saml2?|auth|oauth2?|authorize|"
    r"authenticate|directlogin|cas|idp|adfs|shibboleth|wayf)(?:$|[._-])")
_HINT = "Use a classroom URL such as https://echo360.ca/lesson/<lesson-id>/classroom"


def parse_classroom_url(url: str) -> ClassroomRef:
    """Validate an Echo360 classroom URL and return its lesson reference.

    Query and fragment are dropped; a ``timestamp`` parameter never causes trimming.
    """
    text = (url or "").strip()
    if not text:
        raise InvalidURLError("No classroom URL was given.", hint=_HINT)
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError:
        raise InvalidURLError("The classroom URL could not be parsed.", hint=_HINT) from None
    if parts.scheme.lower() != "https":
        raise InvalidURLError("Classroom URLs must use https://.", hint=_HINT)
    host = (parts.hostname or "").lower().rstrip(".")
    if parts.username or parts.password or port not in (None, 443):
        raise InvalidURLError("Classroom URLs must not contain credentials or a port.",
                              hint=_HINT)
    if host not in SUPPORTED_HOSTS:
        raise InvalidURLError(
            f"Unsupported host {host or '(none)'!r}; only {', '.join(sorted(SUPPORTED_HOSTS))} "
            "has been verified.", hint=_HINT)
    match = _CLASSROOM_PATH.match(parts.path)
    if not match:
        raise InvalidURLError("This is not an Echo360 classroom URL "
                              "(expected /lesson/<id>/classroom).", hint=_HINT)
    lesson_id = unquote(match[1])
    if not _LESSON_ID.match(lesson_id):
        raise InvalidURLError("The lesson id in the URL is not valid.", hint=_HINT)
    return ClassroomRef(host=host, lesson_id=lesson_id,
                        url=f"https://{host}/lesson/{match[1]}/classroom")


def host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def is_echo360_host(host: str) -> bool:
    """True for a supported host or one of its subdomains (e.g. ``content.echo360.ca``)."""
    host = host.lower().rstrip(".")
    return any(host == h or host.endswith("." + h) for h in SUPPORTED_HOSTS)


def is_echo360_url(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return parts.scheme == "https" and is_echo360_host(parts.hostname or "")


def is_login_url(url: str) -> bool:
    """Heuristic: a URL that is part of a sign-in flow.

    Any page off the supported hosts (an institution IdP, Microsoft, Google, ...) or a path
    with a segment such as ``login``, ``sso``, ``saml`` or ``auth`` counts as a login page.
    Classroom pages never count.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return True
    if parts.scheme not in ("http", "https"):
        return False  # about:blank, chrome-error:// and similar are not login pages
    host = (parts.hostname or "").lower()
    if not is_echo360_host(host):
        return True
    if _CLASSROOM_PATH.match(parts.path):
        return False
    return any(_LOGIN_SEGMENT.match(seg) for seg in parts.path.split("/") if seg)


_LOGIN_HOST_LABEL = re.compile(
    r"(?i)^(?:login|logon|signin|sso|idp|auth|adfs|shibboleth|cas|accounts|wayf)\d*$")


def looks_like_login_page(url: str) -> bool:
    """Stricter than :func:`is_login_url` (used for HTTP redirects, where an off-site hop is
    not necessarily a sign-in): a login-like path segment or a login-like first host label
    (``login.microsoftonline.com``, ``sso.university.example``, ...)."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    host = (parts.hostname or "").lower()
    if _CLASSROOM_PATH.match(parts.path):
        return False
    if host and _LOGIN_HOST_LABEL.match(host.split(".", 1)[0]):
        return True
    return any(_LOGIN_SEGMENT.match(seg) for seg in parts.path.split("/") if seg)


def query_param(url: str, name: str) -> str | None:
    try:
        values = parse_qs(urlsplit(url).query).get(name)
    except ValueError:
        return None
    return values[0] if values else None


def lesson_param(url: str) -> str | None:
    """``x-lid`` (observed: the lesson id) of a content URL."""
    return query_param(url, "x-lid")


def media_param(url: str) -> str | None:
    """``x-mid`` (observed: the media UUID) of a content URL."""
    return query_param(url, "x-mid")


_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_INSTITUTION_SEGMENT = re.compile(r"^\d{4}\.([0-9a-fA-F-]{36})$")


def content_path_ids(url: str) -> tuple[str | None, str | None]:
    """(institution id, media id) from an observed content path
    ``/0000.<institution-uuid>/<media-uuid>/1/s2_av.m3u8``. Either may be None."""
    try:
        segments = [s for s in urlsplit(url).path.split("/") if s]
    except ValueError:
        return None, None
    institution = media = None
    for i, seg in enumerate(segments):
        match = _INSTITUTION_SEGMENT.match(seg)
        if match and _UUID.match(match[1]):
            institution = match[1].lower()
            if i + 1 < len(segments) and _UUID.match(segments[i + 1]):
                media = segments[i + 1].lower()
            break
    return institution, media
