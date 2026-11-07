"""Sanitised discovery trace (``--save-trace``).

Each entry records only: method, public URL (no query, fragment or userinfo), status,
content type, origin of the observation, and - for JSON bodies - the *key structure* with
value *types*. No header, cookie, query or JSON value is ever written. Object keys that look
like data rather than field names (ids, e-mail addresses, long strings) are replaced by
``"<key>"``. The file is created with mode 0600.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from echo360_dl.redact import public_url, scrub

TRACE_VERSION = 1
MAX_ENTRIES = 2000
_MAX_DEPTH = 12
_FIELD_NAME = re.compile(r"^[A-Za-z_$@][A-Za-z0-9_$@.:-]{0,47}$")
_DATA_LIKE = re.compile(r"[0-9a-fA-F]{8,}|\d{5,}|@.*\.")


def _key(name: str) -> str:
    if _FIELD_NAME.match(name) and not _DATA_LIKE.search(name):
        return name
    return "<key>"


def json_shape(value: Any, depth: int = 0) -> Any:
    """Structure of a JSON value with types instead of values."""
    if depth > _MAX_DEPTH:
        return "…"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            name = _key(str(k))
            shape = json_shape(v, depth + 1)
            if name in out and name == "<key>":
                continue  # many data-keyed entries collapse into one example
            out[name] = shape
        return out
    if isinstance(value, list):
        if not value:
            return []
        shapes: list[Any] = []
        for item in value[:50]:
            shape = json_shape(item, depth + 1)
            if shape not in shapes:
                shapes.append(shape)
            if len(shapes) >= 3:
                break
        return shapes
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    return type(value).__name__


@dataclass
class TraceRecorder:
    entries: list[dict[str, Any]] = field(default_factory=list)
    dropped: int = 0

    def record(self, method: str, url: str, status: int | None, content_type: str | None,
               body: bytes | None = None, *, origin: str = "browser",
               resource_type: str | None = None) -> None:
        if len(self.entries) >= MAX_ENTRIES:
            self.dropped += 1
            return
        ctype = (content_type or "").split(";", 1)[0].strip().lower() or None
        entry: dict[str, Any] = {"origin": origin, "method": (method or "GET").upper(),
                                 "public_url": scrub(public_url(url)), "status": status,
                                 "content_type": ctype}
        if resource_type:
            entry["resource_type"] = re.sub(r"[^a-z_-]", "", resource_type.lower())[:20]
        if body is not None and ctype and "json" in ctype:
            try:
                entry["json_shape"] = json_shape(json.loads(body))
            except (ValueError, UnicodeDecodeError, RecursionError):
                entry["json_shape"] = "unparseable"
        if body is not None:
            entry["body_bytes"] = len(body)
        self.entries.append(entry)

    def record_html(self, url: str, html: str) -> None:
        """Structure of JSON embedded in an Echo360 page's ``<script>`` blocks.

        Records only the assignment target (a code identifier such as
        ``window.__INITIAL_STATE__``) and the JSON *shape*; text and values never leave
        the page. This is where lesson title, sources and duration are expected to live.
        """
        if len(self.entries) >= MAX_ENTRIES:
            self.dropped += 1
            return
        self.entries.append({"origin": "page", "method": "GET",
                             "public_url": scrub(public_url(url)), "status": None,
                             "content_type": "text/html",
                             "embedded_json": html_data_shapes(html),
                             "inline_scripts": inline_script_outline(html),
                             "body_bytes": len(html.encode("utf-8", "replace"))})

    def to_dict(self, *, lesson_id: str | None = None) -> dict[str, Any]:
        return {"version": TRACE_VERSION, "created_at": int(time.time()),
                "lesson_id": lesson_id, "dropped_entries": self.dropped,
                "note": "Sanitised: no headers, cookies, query strings or JSON values.",
                "entries": self.entries}

    def write(self, path: Path, *, lesson_id: str | None = None) -> Path:
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self.to_dict(lesson_id=lesson_id), indent=2, sort_keys=False)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data + "\n")
        os.chmod(path, 0o600)
        return path



def html_data_shapes(html: str) -> list[dict[str, Any]]:
    """Shapes (types, never values) of JSON embedded in a page's ``<script>`` blocks."""
    from echo360_dl.echo360.embedded import extract_embedded_json

    return [{"script": e.script_index, "kind": e.kind, "target": e.target,
             "shape": json_shape(e.data)} for e in extract_embedded_json(html)]


# All <script> blocks, numbered exactly like embedded.extract_embedded_json numbers them.
_ANY_SCRIPT = re.compile(r"<script\b([^>]*)>(.*?)</script>", re.IGNORECASE | re.DOTALL)
_HAS_SRC = re.compile(r"\bsrc\s*=", re.IGNORECASE)
_CALL_OR_ASSIGN = re.compile(
    r"""([A-Za-z_$][\w$]*(?:\.[\w$]+|\[\s*["'][\w$.-]{1,60}["']\s*\])*)\s*(?:\(|=(?!=))""")
_PROP_NAME = re.compile(r"""(?:["']|\b)([A-Za-z_][A-Za-z0-9_]{1,40})(?:["'])?\s*:""")


def inline_script_outline(html: str) -> list[dict[str, Any]]:
    """Per inline ``<script>``: size, identifiers it calls/assigns and property-like key
    names. Code structure only - string contents and values are never recorded."""
    outline: list[dict[str, Any]] = []
    for n, m in enumerate(_ANY_SCRIPT.finditer(html or "")):
        body = m[2]
        if _HAS_SRC.search(m[1]) or not body.strip():
            continue
        # Drop string literal contents first so nothing inside strings is recorded,
        # except key names of JSON documents encoded in strings (decoded, keys only).
        keys: list[str] = []
        for sm in re.finditer(r""""((?:\\.|[^"\\])*)"|'((?:\\.|[^'\\])*)'""", body):
            raw = sm[1] if sm[1] is not None else sm[2]
            if raw and len(raw) > 40 and ("\\\"" in raw or '":' in raw):
                keys.extend(k for k in re.findall(r'\\?"([A-Za-z_][A-Za-z0-9_]{1,40})\\?"\s*:',
                                                   raw))
        # Keep identifier-like bracket keys (Echo["classroomApp"]) - they are code, not data.
        protected = re.sub(r"""\[\s*(["'])([A-Za-z_$][\w$.-]{0,59})\1\s*\]""",
                           lambda km: f"[\x00{km[2]}\x00]", body)
        code = re.sub(r""""(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'""", '""', protected)
        code = re.sub(r"\[\x00([^\x00]*)\x00\]", lambda km: f'["{km[1]}"]', code)
        idents = []
        for cm in _CALL_OR_ASSIGN.finditer(code):
            name = cm[1]
            if name not in idents and not name.startswith(("function", "if", "for")):
                idents.append(name[:80])
            if len(idents) >= 15:
                break
        keys.extend(_PROP_NAME.findall(code))
        names = []
        for k in keys:
            key = _key(k)
            if key not in names:
                names.append(key)
            if len(names) >= 400:
                break
        outline.append({"script": n, "bytes": len(body), "identifiers": idents,
                        "property_names": names})
        if len(outline) >= 40:
            break
    return outline
