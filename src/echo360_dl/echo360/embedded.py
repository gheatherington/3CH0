"""JSON embedded in Echo360 HTML pages.

OBSERVED (2026-11-07 traces): the classroom page's lesson metadata (title, duration, source
list) is not served by any JSON API the player calls, so it must be embedded in the page.
HYPOTHESIS: as JSON inside ``<script>`` blocks, in one of these forms (all handled):

* ``<script type="application/json">{...}</script>``
* ``x = JSON.parse("{\\"...\\"}")``
* ``Echo["classroomApp"]("{\\"...\\"}")`` - a JSON *string* passed to a function
  (the pattern used by public Echo360 downloaders)
* ``window.x = {...};`` when the literal is strict JSON

Only the parsed data is returned; callers decide what to keep. The trace records shapes
(types) only - see :func:`echo360_dl.echo360.trace.html_data_shapes`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_SCRIPT = re.compile(r"<script\b([^>]*)>(.*?)</script>", re.IGNORECASE | re.DOTALL)
_SCRIPT_TYPE = re.compile(r"""type\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_SCRIPT_ID = re.compile(r"""\bid\s*=\s*["']([\w:.-]{1,60})["']""", re.IGNORECASE)
# A JavaScript string literal (double or single quoted), escapes allowed.
_STRING = re.compile(r""""((?:\\.|[^"\\])*)"|'((?:\\.|[^'\\])*)'""", re.DOTALL)
_TARGET = re.compile(
    r"""([A-Za-z_$][\w$]*(?:\.[\w$]+|\[\s*["'][\w$.-]{1,60}["']\s*\])*)\s*(=|\()\s*$""")
_MIN_STRING = 40  # shorter literals are never interesting JSON documents
MAX_ITEMS = 60


@dataclass(frozen=True)
class EmbeddedJSON:
    kind: str  # "application/json" | "JSON.parse" | "call-argument" | "assignment" | "string"
    target: str | None  # code identifier assigned to / called, e.g. 'Echo["classroomApp"]'
    data: Any
    script_index: int


def _target_before(text: str, start: int) -> tuple[str | None, str | None]:
    """Identifier and operator ('=' or '(') immediately before ``start``."""
    window = text[max(0, start - 160):start].rstrip()
    window = re.sub(r"JSON\.parse\s*\($", "", window).rstrip()
    m = _TARGET.search(window)
    if m is None:
        return None, None
    return m[1][:80], m[2]


def _js_unescape(body: str, quote: str) -> str | None:
    """Decode a JS string literal body via JSON (handles \\uXXXX, \\", \\n, \\/)."""
    if quote == "'":
        body = body.replace("\\'", "'").replace('"', '\\"')
    try:
        return json.loads(f'"{body}"')
    except ValueError:
        return None


def _object_end(text: str, start: int) -> int | None:
    depth, i, quote = 0, start, None
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _as_json_document(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return None
    try:
        value = json.loads(stripped)
    except ValueError:
        return None
    return value if isinstance(value, dict | list) and value else None


def extract_embedded_json(html: str) -> list[EmbeddedJSON]:
    found: list[EmbeddedJSON] = []
    for n, m in enumerate(_SCRIPT.finditer(html or "")):
        if len(found) >= MAX_ITEMS:
            break
        attrs, body = m[1], m[2]
        if not body.strip():
            continue
        type_match = _SCRIPT_TYPE.search(attrs)
        stype = type_match[1].lower() if type_match else ""
        if "json" in stype:
            data = _as_json_document(body)
            if data is not None:
                id_match = _SCRIPT_ID.search(attrs)
                found.append(EmbeddedJSON(stype, id_match[1] if id_match else None, data, n))
            continue
        if stype and "javascript" not in stype and "module" not in stype:
            continue  # templates etc.
        # 1) string literals that contain JSON documents
        for sm in _STRING.finditer(body):
            raw = sm[1] if sm[1] is not None else sm[2]
            if raw is None or len(raw) < _MIN_STRING:
                continue
            decoded = _js_unescape(raw, '"' if sm[1] is not None else "'")
            data = _as_json_document(decoded) if decoded else None
            if data is None:
                continue
            before = body[max(0, sm.start() - 160):sm.start()]
            target, op = _target_before(body, sm.start())
            if re.search(r"JSON\.parse\s*\(\s*$", before):
                kind = "JSON.parse"
            elif op == "(":
                kind = "call-argument"
            else:
                kind = "string"
            item = EmbeddedJSON(kind, target, data, n)
            if any(f.target == item.target and f.script_index == n and f.data == data
                   for f in found):
                continue  # the same document matched twice (e.g. quoted and re-quoted)
            found.append(item)
            if len(found) >= MAX_ITEMS:
                break
        # 2) strict-JSON object/array literals assigned to a name
        for am in re.finditer(r"=\s*([\[{])", body):
            target, op = _target_before(body, am.start() + 1)
            if target is None or op != "=":
                continue
            end = _object_end(body, am.start(1))
            if end is None or end - am.start(1) < 20:
                continue
            data = _as_json_document(body[am.start(1):end])
            if data is not None:
                found.append(EmbeddedJSON("assignment", target, data, n))
    return found
