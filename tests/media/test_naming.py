from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from threech0.media.naming import (
    BASE_BUDGET,
    MAX_NAME_BYTES,
    language_tag,
    lesson_basename,
    media_basename,
    request_basename,
    request_media_basename,
    safe_component,
    single_media_output,
    truncate_utf8,
    with_suffix,
)
from threech0.models import DownloadRequest, Layout, Recording


def _rec(**kw) -> Recording:
    base = dict(lesson_id="G_8f1c2a9e-1234", classroom_url="https://echo360.ca/lesson/x/classroom",
                host="echo360.ca")
    base.update(kw)
    return Recording(**base)


def test_full_name_layout():
    rec = _rec(title="Limits: Part 1", course="MATH 101", recorded_at="2026-11-05T14:00:00Z")
    SID = rec.short_id
    assert lesson_basename(rec) == f"MATH 101 - 2026-11-05 - Limits - Part 1 [{SID}]"
    assert media_basename(rec, "Source 1") == \
        f"MATH 101 - 2026-11-05 - Limits - Part 1 [{SID}] - Source 1"
    assert media_basename(rec, None).endswith(f"[{SID}] - audio")


def test_unknown_metadata_falls_back():
    rec = _rec(lesson_id="abcdef0123456789")
    SID = rec.short_id
    assert lesson_basename(rec) == f"Echo360 lesson [{SID}]"
    assert media_basename(rec, "Source 2") == f"Echo360 lesson [{SID}] - Source 2"


@pytest.mark.parametrize("raw,expected", [
    ("../../etc/passwd", "-..-etc-passwd"),
    ("a/b\\c", "a-b-c"),
    ("...hidden", "hidden"),
    ("trailing. . ", "trailing"),
    ('what?*"<x>|', "what'(x)-"),
    ("tab\tnew\nline\x00nul\x1b", "tab new line nul"),
    ("bidi‮evil", "bidi evil"),
    ("CON", "_CON"),
    ("nul.txt", "_nul.txt"),
    ("", "untitled"),
    ("\x00\x01", "untitled"),
])
def test_safe_component(raw, expected):
    assert safe_component(raw) == expected


def test_unicode_is_nfc_and_preserved():
    decomposed = unicodedata.normalize("NFD", "Café — 日本語 Ünïcödé")
    out = safe_component(decomposed)
    assert out == "Café — 日本語 Ünïcödé" and unicodedata.is_normalized("NFC", out)


def test_truncation_never_splits_code_points():
    text = "é" * 200
    cut = truncate_utf8(text, 101)
    assert cut == "é" * 50 and len(cut.encode()) == 100
    rec = _rec(title="講義" * 200, course="数学", recorded_at="2026-01-01")
    SID = rec.short_id
    for name in (media_basename(rec, "Source 1") + ".mp4",
                 with_suffix(lesson_basename(rec), "en-CA", "transcript", "platform", "json")):
        assert len(name.encode()) <= MAX_NAME_BYTES
        name.encode("utf-8").decode("utf-8")  # valid UTF-8
    assert f"[{SID}] - Source 1" in media_basename(rec, "Source 1")
    assert lesson_basename(rec).endswith(f"[{SID}]")


def test_language_tag_and_suffix():
    assert language_tag("en_CA") == "en-CA"
    assert language_tag(None) == "und"
    assert language_tag("../x") == "x"
    assert with_suffix("Base", "en", "vtt") == "Base.en.vtt"
    assert with_suffix("Base", "fr", "generated", "srt") == "Base.fr.generated.srt"


def test_stale_temp_sweep_only_touches_old_temps_of_this_lesson(tmp_path):
    import os

    from threech0.media.mux import STALE_TEMP_AGE, remove_stale_temps

    now = 10_000.0
    mine_old = tmp_path / ".Lecture [abc123] - Source 1.3ch0-tmp.mp4"
    mine_live = tmp_path / ".Lecture [abc123] - Source 2.3ch0-tmp.mp4"
    other_old = tmp_path / ".Other [zzz999] - Source 1.3ch0-tmp.mp4"
    sidecar_old = tmp_path / ".Lecture [abc123].en.3ch0-tmp.vtt"
    for path, age in ((mine_old, STALE_TEMP_AGE + 1), (mine_live, 5),
                      (other_old, STALE_TEMP_AGE + 1), (sidecar_old, STALE_TEMP_AGE + 1)):
        path.write_bytes(b"x")
        os.utime(path, (now - age, now - age))
    remove_stale_temps(tmp_path, "[abc123]", now=now)
    assert not mine_old.exists() and not sidecar_old.exists()
    assert mine_live.exists() and other_old.exists()


def test_long_labels_never_shorten_the_title():
    rec = _rec(title="Math 1ZA3 C01 (Dr. Childs) - Fall 2026",
               course="MATH 1A03:All Sections of Math 1A03 and 1ZA3",
               recorded_at="2026-09-14T10:28:00-04:00")
    base = lesson_basename(rec)
    for label in ("Source 1", "Source 1 + Source 2 (side by side)",
                  "Source 2 + Source 1 (top-bottom)", "audio"):
        name = media_basename(rec, label)
        assert name.startswith(base + " - ") and "Fall 2026" in name
        assert len((name + ".mp4").encode()) <= MAX_NAME_BYTES
    very_long = _rec(title="Ü" * 300, course="C" * 100, recorded_at="2026-01-01")
    name = media_basename(very_long, "Source 1 + Source 2 (side by side)")
    assert len((name + ".mkv").encode()) <= MAX_NAME_BYTES and name.endswith("(side by side)")


def _req(rec: Recording, **kw) -> DownloadRequest:
    return DownloadRequest(recording=rec, output_dir=Path("/out"), **kw)


@pytest.mark.parametrize("kw, single", [
    (dict(source_indexes=(1,)), True),
    (dict(audio_only=True), True),
    (dict(source_indexes=(1, 2)), False),
    (dict(source_indexes=(1, 2), layout=Layout.SIDE_BY_SIDE), True),
    (dict(source_indexes=(1, 2), layout=Layout.TOP_BOTTOM, keep_separate=True), False),
])
def test_custom_name(kw, single):
    rec = _rec(title="Limits", course="MATH 101")
    req = _req(rec, name="Week 3: Limits", **kw)
    assert request_basename(req) == "Week 3 - Limits"
    assert single_media_output(req) is single
    expected = "Week 3 - Limits" if single else "Week 3 - Limits - Source 1"
    assert request_media_basename(req, "Source 1") == expected


def test_no_custom_name_keeps_lesson_naming():
    rec = _rec(title="Limits", course="MATH 101")
    req = _req(rec, source_indexes=(1,))
    assert request_basename(req) == lesson_basename(rec)
    assert request_media_basename(req, "Source 1") == media_basename(rec, "Source 1")


def test_custom_name_is_made_safe_and_capped():
    rec = _rec()
    req = _req(rec, source_indexes=(1, 2), name="../" + "é" * 300)
    base = request_basename(req)
    assert "/" not in base and len(base.encode()) <= BASE_BUDGET
    assert len((request_media_basename(req, "Source 2") + ".mp4").encode()) <= MAX_NAME_BYTES
