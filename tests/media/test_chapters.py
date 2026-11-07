from __future__ import annotations

import json

import pytest

from tests.media.conftest import FIXTURES
from threech0.errors import UnavailableError, ValidationError
from threech0.media.chapters import (
    chapters_to_json,
    choose_chapters,
    ffmetadata_escape,
    interval_chapters,
    load_chapter_file,
    parse_chapter_data,
    resolve_chapters,
    to_ffmetadata,
    validate_chapters,
)
from threech0.models import Chapter, ChapterMode, ChapterModeKind, ChapterProvenance, ChapterSet


def test_load_valid_file_all_time_forms():
    cs = load_chapter_file(FIXTURES / "chapters_valid.json")
    assert cs.provenance is ChapterProvenance.USER_FILE
    assert cs.source_detail == "chapters_valid.json"
    assert [(c.start, c.end, c.title) for c in cs.chapters] == [
        (0.0, None, "Intro = ; # \\ é"), (2.5, None, "Second — 第二"), (4.0, 5.5, "Wrap-up")]


@pytest.mark.parametrize("obj,needle", [
    ({"version": 2, "chapters": [{"start": 0, "title": "a"}]}, "version must be 1"),
    ({"chapters": [{"start": 0, "title": "a"}]}, "version must be 1"),
    ({"version": 1, "chapters": []}, "'chapters' is empty"),
    ({"version": 1, "chapters": "x"}, "'chapters' must be a list"),
    ({"version": 1, "chapters": [{"title": "a"}]}, "missing 'start'"),
    ({"version": 1, "chapters": [{"start": "-00:01", "title": "a"}]}, "negative"),
    ({"version": 1, "chapters": [{"start": "1:75", "title": "a"}]}, "seconds field"),
    ({"version": 1, "chapters": [{"start": True, "title": "a"}]}, "must be a time"),
    ({"version": 1, "chapters": [{"start": 0, "title": "   "}]}, "title is empty"),
    ({"version": 1, "chapters": [{"start": 0, "title": 5}]}, "'title' must be a string"),
    ({"version": 1, "chapters": [{"start": 10, "title": "a"}, {"start": 10, "title": "b"}]},
     "strictly increase"),
    ({"version": 1, "chapters": [{"start": 10, "title": "a"}, {"start": 5, "title": "b"}]},
     "strictly increase"),
    ({"version": 1, "chapters": [{"start": 10, "end": 5, "title": "a"}]}, "end must be after"),
    ({"version": 1, "chapters": [{"start": 0, "end": 20, "title": "a"},
                                 {"start": 10, "title": "b"}]}, "overlaps the start"),
    ({"version": 1, "chapters": [{"start": 0, "title": "bell\x07"}]}, "control characters"),
    ([1, 2], "JSON object"),
])
def test_invalid_chapter_data(obj, needle):
    with pytest.raises(ValidationError) as exc:
        parse_chapter_data(obj, "c.json")
    assert needle in exc.value.message


def test_load_errors(tmp_path):
    with pytest.raises(ValidationError, match="not found"):
        load_chapter_file(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text('{"version": 1, "chapters": [', "utf-8")
    with pytest.raises(ValidationError, match="not valid JSON"):
        load_chapter_file(bad)
    bom = tmp_path / "bom.json"
    bom.write_bytes(b"\xef\xbb\xbf" + json.dumps(
        {"version": 1, "chapters": [{"start": 0, "title": "x"}]}).encode())
    assert load_chapter_file(bom).chapters == [Chapter(0.0, "x")]


def test_validate_against_duration():
    chapters = [Chapter(0, "a"), Chapter(50, "b", end=70)]
    problems = validate_chapters(chapters, 60.0)
    assert any("ends after the end of the media" in p for p in problems)
    assert any("at or after the end" in p for p in validate_chapters([Chapter(60, "x")], 60))
    assert validate_chapters([Chapter(0, "a"), Chapter(30, "b")], 60.0) == []


def test_resolve_fills_ends_from_next_start_and_duration():
    out = resolve_chapters([Chapter(0, "a"), Chapter(10, "b", end=15), Chapter(20, "c")], 30)
    assert out == [Chapter(0, "a", 10), Chapter(10, "b", 15), Chapter(20, "c", 30)]
    assert resolve_chapters([Chapter(0, "a")], None) == [Chapter(0, "a", None)]
    assert resolve_chapters([Chapter(0, "a", end=99)], 30)[0].end == 30


def test_interval_chapters():
    cs = interval_chapters(6.5, 2.0)
    assert cs.provenance is ChapterProvenance.GENERATED_INTERVAL
    assert [(c.start, c.end, c.title) for c in cs.chapters] == [
        (0.0, 2.0, "Part 1"), (2.0, 4.0, "Part 2"), (4.0, 6.5, "Part 3")]
    assert len(interval_chapters(600.5, 300).chapters) == 2  # 0.5 s tail merged
    with pytest.raises(UnavailableError):
        interval_chapters(0, 10)


def test_ffmetadata_escaping_and_timebase():
    assert ffmetadata_escape("a=b;c#d\\e\nf") == "a\\=b\\;c\\#d\\\\e\\\nf"
    text = to_ffmetadata([Chapter(0, "Intro = ; # \\ é", 2.5), Chapter(2.5, "B\nC", 6.0)],
                         title="Lesson; 1")
    assert text.startswith(";FFMETADATA1\ntitle=Lesson\\; 1\n")
    assert "[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=2500\ntitle=Intro \\= \\; \\# \\\\ é\n" \
        in text
    assert "START=2500\nEND=6000\ntitle=B\\\nC\n" in text
    with pytest.raises(ValueError):
        to_ffmetadata([Chapter(0, "x")])


def test_chapters_json_round_trip():
    cs = ChapterSet([Chapter(0, "a", 2.5), Chapter(2.5, "b", 6.0)], ChapterProvenance.NATIVE,
                    "lesson metadata")
    data = chapters_to_json(cs)
    assert data == {"version": 1, "provenance": "native", "source_detail": "lesson metadata",
                    "chapters": [{"start": "00:00:00.000", "end": "00:00:02.500", "title": "a"},
                                 {"start": "00:00:02.500", "end": "00:00:06.000", "title": "b"}]}
    assert parse_chapter_data(data, "x").chapters == cs.chapters


def test_choose_chapters_precedence(tmp_path):
    native = ChapterSet([Chapter(0, "Native")], ChapterProvenance.NATIVE)
    file_mode = ChapterMode(ChapterModeKind.FILE, file=FIXTURES / "chapters_valid.json")
    chosen, warnings = choose_chapters(file_mode, native, 6.0)
    assert chosen.provenance is ChapterProvenance.USER_FILE
    assert len(warnings) == 1 and "overridden" in warnings[0]
    assert choose_chapters(file_mode, None, 6.0)[1] == []

    chosen, warnings = choose_chapters(ChapterMode(ChapterModeKind.AUTO), native, 6.0)
    assert chosen is native and warnings == []
    chosen, warnings = choose_chapters(ChapterMode(ChapterModeKind.AUTO), None, 6.0)
    assert chosen is None and "No native chapters" in warnings[0]

    chosen, _ = choose_chapters(ChapterMode(ChapterModeKind.INTERVAL, interval=2.0), None, 6.0)
    assert chosen.provenance is ChapterProvenance.GENERATED_INTERVAL
    assert [c.title for c in chosen.chapters] == ["Part 1", "Part 2", "Part 3"]
    with pytest.raises(UnavailableError):
        choose_chapters(ChapterMode(ChapterModeKind.INTERVAL, interval=2.0), None, None)

    assert choose_chapters(ChapterMode(ChapterModeKind.NONE), native, 6.0) == (None, [])
