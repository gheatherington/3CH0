from __future__ import annotations

import json
import stat

from threech0.echo360.trace import TraceRecorder, json_shape

from .conftest import FAKE_JWT, FIXTURES, LESSON, MEDIA, playlist_url


def test_json_shape_has_types_not_values():
    shape = json_shape({"name": "Secret Title", "n": 3, "ok": True, "x": None,
                        "list": [{"a": "v"}, {"a": "w"}], "empty": []})
    assert shape == {"name": "string", "n": "number", "ok": "bool", "x": "null",
                     "list": [{"a": "string"}], "empty": []}


def test_data_like_keys_are_replaced():
    shape = json_shape({MEDIA: {"t": 1}, "someone@example.edu": 1, "normalKey": "v"})
    assert "<key>" in shape and "normalKey" in shape
    assert MEDIA not in json.dumps(shape) and "example.edu" not in json.dumps(shape)


def test_trace_contains_no_cookie_header_query_or_values(tmp_path):
    rec = TraceRecorder()
    body = (FIXTURES / "lesson_metadata.json").read_bytes()
    rec.record("GET", "https://echo360.ca/section/fictional/syllabus?token=" + FAKE_JWT, 200,
               "application/json; charset=utf-8", body, resource_type="fetch")
    rec.record("GET", playlist_url(1), 200, "application/x-mpegurl", b"#EXTM3U")
    rec.record("get", "https://user:pw@content.echo360.ca/x#frag", 403, None, None,
               origin="httpx")
    path = rec.write(tmp_path / "sub" / "trace.json", lesson_id=LESSON)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    text = path.read_text()
    data = json.loads(text)
    assert data["entries"][0]["public_url"] == "https://echo360.ca/section/fictional/syllabus"
    assert data["entries"][0]["json_shape"]["data"][0]["lesson"]["lesson"]["name"] == "string"
    assert data["entries"][2]["method"] == "GET"
    for forbidden in (FAKE_JWT, "x-uid", "FAKESESSION000", "FAKEUSER000", "token=", "?", "#frag",
                      "user:pw", "Fictional Lecture", "FICT 101", "captions.echo360.ca/fictional",
                      "Cookie:", "3240000"):
        assert forbidden not in text, forbidden


def test_unparseable_json_and_cap():
    rec = TraceRecorder()
    rec.record("GET", "https://echo360.ca/x", 200, "application/json", b"{nope")
    assert rec.entries[0]["json_shape"] == "unparseable"
    from threech0.echo360 import trace as trace_mod
    rec.entries = [{}] * trace_mod.MAX_ENTRIES
    rec.record("GET", "https://echo360.ca/y", 200, None)
    assert rec.dropped == 1


def test_tokens_in_url_paths_are_redacted():


    token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzZWNyZXQtdXNlciJ9.c2lnbmF0dXJlLXZhbHVl"
    trace = TraceRecorder()
    trace.record("GET", f"https://echo360.ca/api/{token}/media?sig=abc", 200, "text/html")
    entry = trace.entries[0]
    assert token not in entry["public_url"] and "sig=abc" not in entry["public_url"]


def test_embedded_page_json_records_structure_only():
    from threech0.echo360.trace import html_data_shapes

    secret_title = "Lecture 7 SECRETTITLE"
    payload = json.dumps({"lesson": {"name": secret_title, "durationMs": 3240000,
                                     "video": [{"label": "Camera", "src": "s1_av.m3u8"}]}})
    js_string = json.dumps(payload)  # as embedded: JSON.parse("{\"lesson\": ...}")
    html = f"""<html><head>
      <script>Echo["classroomApp"] = JSON.parse({js_string});</script>
      <script>window.__STATE__ = {{"user": {{"email": "SECRETMAIL@x.io", "id": 5}}}};</script>
      <script type="application/json" id="boot">{{"token": "SECRETTOKEN123", "n": 1}}</script>
      <script>var cfg = {{noQuotes: "SECRETLITERAL", f: function() {{}}}};</script>
      <script src="/assets/app.js"></script>
      <script>Echo["mediaPlayerApp"]({json.dumps(json.dumps({"mediaId": "SECRETMEDIA",
          "sources": [{"id": "s1", "label": "Screen"}]}))});</script>
    </head></html>"""
    shapes = html_data_shapes(html)
    text = json.dumps(shapes)
    for secret in ("SECRETTITLE", "SECRETMAIL", "SECRETTOKEN123", "SECRETLITERAL", "Camera"):
        assert secret not in text
    by_kind = {s["kind"]: s for s in shapes}
    assert "SECRETMEDIA" not in text and "Screen" not in text
    assert by_kind["call-argument"]["target"] == 'Echo["mediaPlayerApp"]'
    assert by_kind["call-argument"]["shape"]["sources"] == [{"id": "string", "label": "string"}]
    assert by_kind["JSON.parse"]["target"] == 'Echo["classroomApp"]'
    assert by_kind["JSON.parse"]["shape"]["lesson"]["durationMs"] == "number"
    assert by_kind["JSON.parse"]["shape"]["lesson"]["video"] == [{"label": "string",
                                                                  "src": "string"}]
    assert by_kind["assignment"]["target"] == "window.__STATE__"
    assert by_kind["application/json"]["target"] == "boot"
    assert by_kind["application/json"]["shape"] == {"token": "string", "n": "number"}


def test_inline_script_outline_has_names_but_no_values():
    from threech0.echo360.trace import inline_script_outline

    encoded = json.dumps(json.dumps({"lessonName": "SECRETTITLE2", "durationMs": 1}))
    html = f"""<script src="/x.js"></script>
    <script>
      window.Echo = window.Echo || {{}};
      Echo.boot({{lesson: {{title: "SECRETTITLE", sections: [{{label: 'SECRETLABEL'}}]}}}});
      Echo["classroomApp"]({encoded});
    </script>"""
    outline = inline_script_outline(html)
    text = json.dumps(outline)
    for secret in ("SECRETTITLE", "SECRETLABEL", "SECRETTITLE2"):
        assert secret not in text
    (entry,) = outline
    assert "Echo.boot" in entry["identifiers"] and 'Echo["classroomApp"]' in entry["identifiers"]
    for name in ("lesson", "title", "sections", "label", "lessonName", "durationMs"):
        assert name in entry["property_names"], name
