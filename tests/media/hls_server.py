"""A tiny local HLS server for media tests (localhost only, no network).

Behaviour mirrors what was observed on the Echo360 content host:

* every request needs a valid ``CloudFront-Signature`` cookie, otherwise **403**
  (CloudFront answers 403 for absent objects as well);
* single-file fMP4 playlists are fetched with ``Range`` requests (``206``);
* query strings (signed ``x-*`` parameters) are accepted and ignored.

Test controls: :meth:`HLSServer.expire` (403 until the session refreshes),
``expire_after`` (expire after N media requests), ``forbid`` (path prefixes that
are always 403), ``fail_next`` (transient 503s), ``throttle`` (slow responses) and
a per-path request log.
"""

from __future__ import annotations

import collections
import re
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

COOKIE_NAME = "CloudFront-Signature"


class _State:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.lock = threading.Lock()
        self.token = "tok-initial-SECRETVALUE1"
        self.next_token = "tok-refreshed-SECRETVALUE2"
        self.expired = False
        self.expire_after: int | None = None  # media requests before expiring
        self.media_requests = 0
        self.forbid: set[str] = set()  # path prefixes ("/lesson/s2") that are always 403
        self.fail_next: dict[str, int] = {}  # path suffix -> remaining 503 answers
        self.throttle: float = 0.0  # seconds to sleep per media request
        self.requests: collections.Counter[str] = collections.Counter()
        self.statuses: list[tuple[str, int]] = []
        self.headers_seen: list[dict[str, str]] = []


class HLSServer:
    def __init__(self, root: Path) -> None:
        self.state = _State(root)
        state = self.state

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: object) -> None:
                pass

            def _send(self, status: int, body: bytes = b"", headers: dict | None = None,
                      head: bool = False) -> None:
                self.send_response(status)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if not head:
                    self.wfile.write(body)
                with state.lock:
                    state.statuses.append((urlsplit(self.path).path, status))

            def _authorised(self) -> bool:
                cookies = self.headers.get_all("Cookie") or []
                values = {}
                for header in cookies:
                    for part in header.split(";"):
                        name, _, value = part.strip().partition("=")
                        values[name] = value
                with state.lock:
                    return not state.expired and values.get(COOKIE_NAME) == state.token

            def do_HEAD(self) -> None:
                self.do_GET(head=True)

            def do_GET(self, head: bool = False) -> None:
                path = urlsplit(self.path).path
                with state.lock:
                    state.requests[path] += 1
                    state.headers_seen.append({k.lower(): v for k, v in self.headers.items()})
                    is_media = path.endswith(".mp4")
                    if is_media:
                        state.media_requests += 1
                        if (state.expire_after is not None
                                and state.media_requests > state.expire_after):
                            state.expired = True
                            state.expire_after = None
                    forbidden = any(path.startswith(p) for p in state.forbid)
                    transient = 0
                    for suffix, remaining in state.fail_next.items():
                        if remaining > 0 and path.endswith(suffix):
                            state.fail_next[suffix] = remaining - 1
                            transient = 1
                            break
                    throttle = state.throttle if is_media else 0.0
                if forbidden or not self._authorised():
                    self._send(HTTPStatus.FORBIDDEN, b"<Error>AccessDenied</Error>",
                               {"Content-Type": "application/xml"}, head)
                    return
                if transient:
                    self._send(HTTPStatus.SERVICE_UNAVAILABLE, b"busy", {}, head)
                    return
                target = (state.root / path.lstrip("/")).resolve()
                if state.root.resolve() not in target.parents or not target.is_file():
                    self._send(HTTPStatus.FORBIDDEN, b"<Error>AccessDenied</Error>", {}, head)
                    return
                if throttle:
                    time.sleep(throttle)
                data = target.read_bytes()
                ctype = ("application/vnd.apple.mpegurl" if path.endswith(".m3u8")
                         else "video/mp4")
                rng = self.headers.get("Range")
                if rng:
                    m = re.fullmatch(r"bytes=(\d+)-(\d*)", rng.strip())
                    if not m:
                        self._send(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, b"", {}, head)
                        return
                    start = int(m[1])
                    end = int(m[2]) if m[2] else len(data) - 1
                    end = min(end, len(data) - 1)
                    body = data[start:end + 1]
                    self._send(HTTPStatus.PARTIAL_CONTENT, body, {
                        "Content-Type": ctype, "Accept-Ranges": "bytes",
                        "Content-Range": f"bytes {start}-{end}/{len(data)}"}, head)
                    return
                self._send(HTTPStatus.OK, data, {"Content-Type": ctype,
                                                 "Accept-Ranges": "bytes"}, head)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    # -- lifecycle -------------------------------------------------------------------
    def start(self) -> HLSServer:
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    # -- controls ----------------------------------------------------------------------
    def expire(self) -> None:
        """Invalidate the current token; the next refresh obtains a new one."""
        with self.state.lock:
            self.state.expired = True

    def issue_refreshed_token(self) -> str:
        """Called by the fake session's ``refresh()``."""
        with self.state.lock:
            self.state.token = self.state.next_token
            self.state.next_token = self.state.next_token + "x"
            self.state.expired = False
            return self.state.token

    def count(self, suffix: str) -> int:
        with self.state.lock:
            return sum(n for p, n in self.state.requests.items() if p.endswith(suffix))

    def reset_log(self) -> None:
        with self.state.lock:
            self.state.requests.clear()
            self.state.statuses.clear()
            self.state.headers_seen.clear()
            self.state.media_requests = 0
