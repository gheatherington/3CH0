"""Import a saved browser "Copy as cURL" request as *data*.

The text is tokenised with :mod:`shlex` and interpreted; it is never executed and no
subprocess is ever started. Supported: Chrome/Firefox bash and zsh captures (single, double
and ``$'...'`` quoting, ``\\`` line continuations), ``--url``, ``-H/--header``,
``-b/--cookie``, ``-A/--user-agent``, ``-e/--referer`` and ``--compressed``. Unknown flags are
ignored. Chrome's Windows "Copy as cURL (cmd)" format (``^"...^"`` quoting) is converted
cheaply; PowerShell captures are rejected.

Cookies from the request get host-only scope for the request host, ``secure=True`` and no
expiry: that is exactly the scope the browser proved by sending them to that URL. Only
:data:`~echo360_dl.models.REPLAYABLE_HEADERS` are kept. ``AuthContext.expires_at`` comes from
the CloudFront policy's ``DateLessThan`` (only the expiry is read).
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from echo360_dl.errors import UsageError
from echo360_dl.models import REPLAYABLE_HEADERS, AuthContext, Cookie

_MAX_FILE_BYTES = 1_000_000
_HINT = ("In the browser's developer tools (Network tab), right-click the .m3u8 request and "
         "choose Copy > Copy as cURL (bash), then save it to a file.")

# Flags whose next token is a value we do not need (skipped so it is not mistaken for a URL).
_SKIP_VALUE_FLAGS = frozenset({
    "-X", "--request", "-d", "--data", "--data-raw", "--data-binary", "--data-ascii",
    "--data-urlencode", "-F", "--form", "--form-string", "-o", "--output", "-u", "--user",
    "-x", "--proxy", "-m", "--max-time", "--connect-timeout", "-r", "--range", "-w",
    "--write-out", "-T", "--upload-file", "-c", "--cookie-jar", "-K", "--config",
    "--retry", "--resolve", "--cacert", "--cert", "--key", "-E", "--interface", "-U",
    "--proxy-user", "--limit-rate", "-z", "--time-cond", "--max-redirs",
})
_VALUE_FLAGS = frozenset({"-H", "--header", "-b", "--cookie", "--url", "-A", "--user-agent",
                          "-e", "--referer"})
_SHORT_ATTACHED = ("-H", "-b", "-A", "-e")
_CONTROL_OPERATORS = frozenset({";", "&", "&&", "|", "||", ";;", "|&"})


@dataclass(frozen=True)
class ImportedRequest:
    """Inert result of parsing a cURL command. ``auth`` is secret-bearing."""

    url: str
    auth: AuthContext = field(repr=False)
    warnings: tuple[str, ...] = ()

    def __repr__(self) -> str:
        from echo360_dl.redact import public_url

        return f"ImportedRequest(url={public_url(self.url)!r}, auth={self.auth!r})"


def load_request_file(path: Path) -> ImportedRequest:
    """Read and parse a saved cURL request file (never executed)."""
    try:
        size = path.stat().st_size
        if size > _MAX_FILE_BYTES:
            raise UsageError(f"Request file {path.name!r} is too large to be a cURL command.",
                             hint=_HINT)
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except FileNotFoundError:
        raise UsageError(f"Request file not found: {path}", hint=_HINT) from None
    except IsADirectoryError:
        raise UsageError(f"Request file is a directory: {path}", hint=_HINT) from None
    except OSError as exc:
        raise UsageError(f"Cannot read request file {path.name!r}: {exc.strerror}",
                         hint=_HINT) from None
    return parse_curl(text)


def parse_curl(text: str) -> ImportedRequest:
    """Parse a "Copy as cURL" command into an :class:`ImportedRequest`."""
    if not text or not text.strip():
        raise UsageError("The request file is empty.", hint=_HINT)
    stripped = text.lstrip()
    if re.match(r"(?i)(invoke-webrequest|invoke-restmethod|\$session\s*=|iwr\s)", stripped):
        raise UsageError("PowerShell captures are not supported.",
                         hint="Use Copy as cURL (bash) instead. " + _HINT)
    if _looks_like_cmd(text):
        text = _cmd_to_posix(text)
    tokens = _tokenize(_normalise_bash(text))
    if not tokens or tokens[0].rsplit("/", 1)[-1].lower() not in {"curl", "curl.exe"}:
        raise UsageError("The request file does not contain a curl command.", hint=_HINT)

    warnings: list[str] = []
    url: str | None = None
    headers: dict[str, str] = {}
    cookie_strings: list[str] = []
    extra_positionals = 0
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok in _CONTROL_OPERATORS:
            warnings.append("Ignored everything after a shell command separator; the file was "
                            "read as data and nothing was executed.")
            break
        name, value, consumed = _option(tokens, i)
        i += consumed
        if name is None:
            if url is None and re.match(r"(?i)^https?://", tok):
                url = tok
            else:
                extra_positionals += 1
            continue
        if value is None:
            continue
        if name == "--url":
            url = url or value
        elif name in ("-H", "--header"):
            hname, sep, hvalue = value.partition(":")
            if not sep:
                continue
            hname = hname.strip().lower()
            if hname == "cookie":
                cookie_strings.append(hvalue.strip())
            elif hname in REPLAYABLE_HEADERS:
                headers[hname] = hvalue.strip()
        elif name in ("-b", "--cookie"):
            if "=" in value:
                cookie_strings.append(value)
            else:
                warnings.append("A cookie-file reference (-b FILE) was ignored; cookie files "
                                "are not read.")
        elif name in ("-A", "--user-agent"):
            headers["user-agent"] = value
        elif name in ("-e", "--referer"):
            headers["referer"] = value
    if extra_positionals:
        warnings.append(f"Ignored {extra_positionals} extra argument(s) that are not part of "
                        "the request.")
    if not url:
        raise UsageError("No https URL was found in the curl command.", hint=_HINT)
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme.lower() != "https" or not host:
        raise UsageError("The request URL must be an https URL.", hint=_HINT)

    cookies: dict[str, Cookie] = {}
    for raw in cookie_strings:
        for pair in raw.split(";"):
            cname, sep, cvalue = pair.strip().partition("=")
            cname = cname.strip()
            if not sep or not cname or any(ch in cname for ch in " \t\r\n,"):
                continue
            cookies[cname] = Cookie(name=cname, value=cvalue.strip(), domain=host, path="/",
                                    secure=True, http_only=False, expires=None)
    expires_at = None
    policy = cookies.get("CloudFront-Policy")
    if policy is not None:
        expires_at = cloudfront_policy_expiry(policy.value)
    auth = AuthContext(cookies=list(cookies.values()), headers=headers, source="request-file",
                       expires_at=expires_at)
    return ImportedRequest(url=url, auth=auth, warnings=tuple(warnings))


def cloudfront_policy_expiry(value: str) -> float | None:
    """Earliest ``DateLessThan`` epoch from a CloudFront custom policy cookie value.

    CloudFront's URL-safe base64 replaces ``+ = /`` with ``- _ ~``. Only the expiry is read;
    the policy itself is never returned or logged. Returns None if it does not decode.
    """
    if not value or len(value) > 16_384:
        return None
    text = value.strip().replace("-", "+").replace("_", "=").replace("~", "/")
    text += "=" * (-len(text) % 4)
    try:
        decoded = json.loads(base64.b64decode(text, validate=False))
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
    statements = decoded.get("Statement") if isinstance(decoded, dict) else None
    if not isinstance(statements, list):
        return None
    epochs: list[float] = []
    for st in statements:
        try:
            raw = st["Condition"]["DateLessThan"]["AWS:EpochTime"]
        except (KeyError, TypeError):
            continue
        numeric = isinstance(raw, int | float) and not isinstance(raw, bool) and raw > 0
        if numeric or (isinstance(raw, str) and raw.isdigit()):
            epochs.append(float(raw))
    return min(epochs) if epochs else None


# -- tokenisation helpers ------------------------------------------------------------------


def _option(tokens: list[str], i: int) -> tuple[str | None, str | None, int]:
    """Return (flag name or None for a positional, value, tokens consumed)."""
    tok = tokens[i]
    if not tok.startswith("-") or tok == "-":
        return None, None, 1
    if tok.startswith("--") and "=" in tok:
        name, _, value = tok.partition("=")
        return (name, value, 1) if name in _VALUE_FLAGS else (name, None, 1)
    if tok in _VALUE_FLAGS:
        if i + 1 < len(tokens) and tokens[i + 1] not in _CONTROL_OPERATORS:
            return tok, tokens[i + 1], 2
        return tok, None, 1
    for short in _SHORT_ATTACHED:
        if tok.startswith(short) and len(tok) > 2:
            return short, tok[2:], 1
    if tok in _SKIP_VALUE_FLAGS:
        return tok, None, 2 if i + 1 < len(tokens) else 1
    return tok, None, 1  # unknown or valueless flag (--compressed, -L, -s, ...): ignored


def _tokenize(text: str) -> list[str]:
    lexer = shlex.shlex(text, posix=True, punctuation_chars=";&|")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError as exc:  # unbalanced quotes
        raise UsageError(f"The curl command could not be parsed ({exc}).", hint=_HINT) from None


def _normalise_bash(text: str) -> str:
    r"""Remove ``\`` line continuations and rewrite ``$'...'`` (ANSI-C) strings as ordinary
    single-quoted strings, tracking quote state so quoted text is left alone."""
    out: list[str] = []
    i, n = 0, len(text)
    quote: str | None = None
    while i < n:
        ch = text[i]
        if quote == "'":
            out.append(ch)
            if ch == "'":
                quote = None
            i += 1
            continue
        if quote == '"':
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                quote = None
            i += 1
            continue
        if ch == "\\" and text.startswith("\r\n", i + 1):
            out.append(" ")
            i += 3
            continue
        if ch == "\\" and i + 1 < n and text[i + 1] == "\n":
            out.append(" ")
            i += 2
            continue
        if ch == "\\" and i + 1 < n:
            out.append(text[i:i + 2])
            i += 2
            continue
        if ch == "$" and i + 1 < n and text[i + 1] == "'":
            decoded, i = _ansi_c(text, i + 2)
            out.append(shlex.quote(decoded))
            continue
        if ch in "'\"":
            quote = ch
        out.append(ch)
        i += 1
    return "".join(out)


_ANSI_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", "'": "'", '"': '"', "a": "\a",
                 "b": "\b", "e": "\x1b", "E": "\x1b", "f": "\f", "v": "\v", "?": "?"}


def _ansi_c(text: str, i: int) -> tuple[str, int]:
    """Decode a ``$'...'`` body starting at ``i``; return (value, index after the quote)."""
    out: list[str] = []
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "'":
            return "".join(out), i + 1
        if ch == "\\" and i + 1 < n:
            nxt = text[i + 1]
            if nxt in _ANSI_ESCAPES:
                out.append(_ANSI_ESCAPES[nxt])
                i += 2
                continue
            hex_len = {"x": 2, "u": 4, "U": 8}.get(nxt)
            if hex_len:
                match = re.match(rf"[0-9a-fA-F]{{1,{hex_len}}}", text[i + 2:])
                if match:
                    try:
                        out.append(chr(int(match[0], 16)))
                    except (ValueError, OverflowError):
                        out.append(match[0])
                    i += 2 + len(match[0])
                    continue
            out.append(ch)
            i += 1
            continue
        out.append(ch)
        i += 1
    raise UsageError("The curl command has an unterminated $'...' string.", hint=_HINT)


def _looks_like_cmd(text: str) -> bool:
    return '^"' in text or bool(re.search(r"\^\s*\r?\n", text))


def _cmd_to_posix(text: str) -> str:
    """Convert Chrome's "Copy as cURL (cmd)" escaping to POSIX quoting.

    cmd captures wrap arguments in ``^"...^"``, escape special characters with ``^`` and use
    ``\\"`` for a literal quote inside an argument. Removing the caret escapes and the
    ``^``-newline continuations yields double-quoted POSIX arguments.
    """
    text = re.sub(r"\^\s*\r?\n", " ", text)
    return re.sub(r"\^(.)", r"\1", text)
