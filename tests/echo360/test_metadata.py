from __future__ import annotations

import json

import pytest

from echo360_dl.echo360 import metadata
from echo360_dl.errors import ValidationError
from echo360_dl.models import ChapterProvenance, DiscoveryMethod, TranscriptProvenance

from .conftest import FIXTURES, LESSON, MEDIA, OTHER_LESSON

SYLLABUS_URL = "https://echo360.ca/section/fictional-section/syllabus"


def _doc():
    return json.loads((FIXTURES / "lesson_metadata.json").read_text())


def test_extracts_lesson_fields_and_ignores_other_lessons():
    meta = metadata.extract_metadata([(SYLLABUS_URL, _doc())], LESSON)
    assert meta.title == "Fictional Lecture 7: Imaginary Numbers"
    assert meta.course == "FICT 101 - Example Course"
    assert meta.recorded_at == "2026-01-15T14:30:00Z"
    assert meta.duration == 3240.0
    assert meta.media_ids == [MEDIA]
    assert meta.source_labels == {1: "Lecturer camera", 2: "Classroom screen"}
    assert meta.listed_source_indexes == {1, 2}
    assert [c.title for c in meta.chapters.chapters] == ["Introduction", "Worked example"]
    assert meta.chapters.chapters[1].start == 510.0
    assert meta.chapters.provenance is ChapterProvenance.NATIVE
    langs = {c.language: c for c in meta.captions}
    assert set(langs) == {"en", "fr-CA"}
    assert langs["en"].formats == {"vtt": "https://captions.echo360.ca/fictional/en.vtt"}
    assert langs["fr-CA"].formats == {"srt": "https://captions.echo360.ca/fictional/fr.srt"}
    assert all(c.kind is TranscriptProvenance.PLATFORM_CAPTIONS and c.timed
               and c.method is DiscoveryMethod.METADATA for c in meta.captions)
    # Nothing from the other lesson leaks in.
    flat = json.dumps([meta.title, meta.course, [c.title for c in meta.chapters.chapters]])
    assert "WRONG" not in flat and "must not" not in flat


def test_every_used_hypothesis_is_labelled():
    meta = metadata.extract_metadata([(SYLLABUS_URL, _doc())], LESSON)
    hyp = [n.message for n in meta.notes if not n.observed]
    for word in ("Title", "Course", "Recording start", "Duration", "Media id",
                 "Source labels", "Caption", "chapters"):
        assert any(word in m for m in hyp), word
    assert all("hypothesis" in m for m in hyp)
    assert all(not n.observed for n in meta.notes)


def test_other_lesson_scope_selected_by_id():
    meta = metadata.extract_metadata([(SYLLABUS_URL, _doc())], OTHER_LESSON)
    assert meta.title.startswith("Another Fictional Lecture")
    assert meta.course == "WRONG COURSE"


def test_unrelated_documents_yield_nothing():
    meta = metadata.extract_metadata([("https://echo360.ca/api/user", {"user": {"name": "x"}})],
                                     LESSON)
    assert meta.title is None and meta.notes == [] and meta.chapters is None


def test_url_naming_the_lesson_scopes_whole_document():
    doc = {"data": {"name": "Fictional Title", "id": LESSON, "durationMs": "bad"}}
    url = f"https://echo360.ca/api/ui/echoplayer/lessons/{LESSON}/info"
    meta = metadata.extract_metadata([(url, doc)], LESSON)
    assert meta.title == "Fictional Title" and meta.duration is None


@pytest.mark.parametrize(("value", "expected"), [
    ({"duration": "PT54M"}, 3240.0), ({"duration": 3240}, 3240.0),
    ({"duration": "00:54:00"}, 3240.0), ({"duration": 10**9}, None), ({"duration": True}, None),
    ({"durationMs": 1500}, 1.5)])
def test_duration_variants(value, expected):
    meta = metadata.extract_metadata([("https://echo360.ca/x", {"id": LESSON, **value})], LESSON)
    assert meta.duration == expected


def test_generic_or_conflicting_labels_are_not_used():
    doc = {"id": LESSON, "sources": [
        {"uri": "https://content.echo360.ca/a/s1_av.m3u8", "label": "Source 1"},
        {"uri": "https://content.echo360.ca/a/s2_av.m3u8", "label": "Screen"},
        {"sourceIndex": 2, "label": "Camera"},
        {"sourceIndex": 3, "name": "Document camera"}]}
    meta = metadata.extract_metadata([("https://echo360.ca/x", doc)], LESSON)
    assert meta.source_labels == {3: "Document camera"}


def test_chapters_require_every_item_valid():
    doc = {"id": LESSON, "chapters": [{"title": "A", "start": "00:00:00"},
                                      {"title": "B"}]}
    meta = metadata.extract_metadata([("https://echo360.ca/x", doc)], LESSON)
    assert meta.chapters is None
    doc = {"id": LESSON, "navigation": [{"name": "A", "time": "00:00:00"},
                                        {"name": "B", "time": 90}]}
    meta = metadata.extract_metadata([("https://echo360.ca/x", doc)], LESSON)
    assert [(c.start, c.title) for c in meta.chapters.chapters] == [(0, "A"), (90, "B")]


def test_bookmarks_are_not_treated_as_chapters():
    doc = {"id": LESSON, "bookmarks": [{"title": "My private note", "startMs": 1000}]}
    assert metadata.extract_metadata([("https://echo360.ca/x", doc)], LESSON).chapters is None


def test_html_media_id_and_caption_patterns():
    html = ('<script>window.x = {\\"mediaId\\":\\"' + MEDIA + '\\", '
            '"u": "https:\\/\\/captions.echo360.ca\\/fictional\\/track.vtt"}</script>')
    meta = metadata.extract_metadata([], LESSON, html=html)
    assert meta.media_ids == [MEDIA]
    assert meta.captions[0].formats == {"vtt": "https://captions.echo360.ca/fictional/track.vtt"}
    assert all(not n.observed for n in meta.notes)


def test_caption_urls_off_echo360_are_ignored():
    doc = {"id": LESSON, "captions": [{"url": "https://evil.example/en.vtt", "lang": "en"}]}
    assert metadata.extract_metadata([("https://echo360.ca/x", doc)], LESSON).captions == []


def test_transcript_json_cues():
    cues = metadata.parse_transcript_json((FIXTURES / "transcript_cues.json").read_bytes())
    assert [(c.start, c.end) for c in cues] == [(0.0, 4.5), (4.5, 9.0), (9.0, None)]
    assert cues[0].speaker == "Speaker 1"


def test_transcript_untimed_and_inconsistent_times():
    data = json.dumps({"items": [{"text": "No timing at all."},
                                 {"text": "Backwards", "start": 5, "end": 2}]}).encode()
    cues = metadata.parse_transcript_json(data)
    assert all(c.start is None and c.end is None for c in cues)
    assert [c.text for c in cues] == ["No timing at all.", "Backwards"]


@pytest.mark.parametrize("data", [b"", b"<html>", b"[1, 2]", b'{"cues": "x"}'])
def test_transcript_garbage_rejected(data):
    with pytest.raises(ValidationError):
        metadata.parse_transcript_json(data)


def test_caption_track_from_observed_responses():
    vtt = metadata.caption_track_from_response("https://captions.echo360.ca/a/b.vtt",
                                               "text/vtt", b"WEBVTT")
    assert vtt.formats == {"vtt": "https://captions.echo360.ca/a/b.vtt"}
    assert vtt.method is DiscoveryMethod.OBSERVED_REQUEST
    body = (FIXTURES / "transcript_cues.json").read_bytes()
    js = metadata.caption_track_from_response("https://echo360.ca/api/t", "application/json",
                                              body)
    assert js.kind is TranscriptProvenance.PLATFORM_TRANSCRIPT and js.timed is False
    assert metadata.caption_track_from_response("https://echo360.ca/api/u",
                                                "application/json", b'{"a": [1]}') is None


def test_transcript_api_url_is_quoted():
    url = metadata.transcript_api_url("echo360.ca", LESSON, MEDIA)
    assert url == (f"https://echo360.ca/api/ui/echoplayer/lessons/{LESSON}/medias/{MEDIA}"
                   "/transcript")


def test_observed_caption_language_inferred_from_evidence():
    from echo360_dl.echo360.metadata import caption_language, caption_track_from_response

    vtt = b"WEBVTT\nKind: captions\nLanguage: en-CA\n\n00:00.000 --> 00:01.000\nhi\n"
    assert caption_language("https://x.example/c/abc.vtt", vtt) == "en-CA"
    assert caption_language("https://x.example/c.vtt?lang=fr", b"WEBVTT\n\n") == "fr"
    assert caption_language("https://x.example/captions/es.vtt", None) == "es"
    assert caption_language("https://x.example/m/s1_av.vtt", None) is None
    assert caption_language("https://x.example/c/abc.vtt", b"WEBVTT\n\n") is None
    track = caption_track_from_response("https://x.example/c/abc.vtt", "text/vtt", vtt)
    assert track is not None and track.language == "en-CA"


def test_observed_transcript_shape_and_automated_label():
    """Shape observed on echo360.ca (2026-11-07 trace); values fictional."""
    import json as _json

    from echo360_dl.echo360.metadata import parse_transcript_json, transcript_label

    body = {"status": "ok", "message": "", "data": {
        "id": "t1", "created": "2026-01-05T12:00:00Z", "status": "COMPLETE",
        "isAutomated": True, "institutionId": "i", "mediaId": "m",
        "contentJSON": {"cues": [
            {"startMs": 1000, "endMs": 2500, "speaker": "", "content": "Welcome back.",
             "notes": [], "confidence": {"raw": [0.9], "average": 0.9, "needsWork": False}},
            {"startMs": 2500, "endMs": 4000, "speaker": "Instructor", "content": "Today…",
             "notes": [], "confidence": {"raw": [0.8], "average": 0.8, "needsWork": False}}],
            "notes": []}}}
    cues = parse_transcript_json(_json.dumps(body).encode())
    assert [(c.start, c.end, c.text) for c in cues] == [(1.0, 2.5, "Welcome back."),
                                                        (2.5, 4.0, "Today…")]
    assert cues[1].speaker == "Instructor" and not cues[0].speaker
    assert transcript_label(body) == "Echo360 automatic transcript (speech recognition)"
    body["data"]["isAutomated"] = False
    assert transcript_label(body) == "Echo360 transcript"


def test_automatic_platform_captions_titled_as_such():
    from echo360_dl.media.subtitles import track_title
    from echo360_dl.models import Transcript, TranscriptProvenance

    t = Transcript(language="en", provenance=TranscriptProvenance.PLATFORM_TRANSCRIPT,
                   label="Echo360 automatic transcript")
    assert track_title(t) == "English — Echo360 automatic captions"


def test_metadata_read_from_json_string_embedded_in_page():
    """Hypothetical embedding form (JSON string passed to a function); fictional values."""
    import json as _json

    from echo360_dl.echo360.metadata import extract_metadata

    lesson = "G_fictional-lesson-0001"
    data = {"lesson": {"lesson": {"id": lesson, "name": "Fictional Lecture 3"}}}
    html = (f"<html><script>Echo[\"classroomApp\"]({_json.dumps(_json.dumps(data))});"
            "</script></html>")
    meta = extract_metadata([], lesson, html=html)
    assert meta.title == "Fictional Lecture 3"
    assert any(n.observed is False for n in meta.notes)  # still labelled a hypothesis


def _player_app(lesson_id: str) -> dict:
    """Mirrors the structure observed on echo360.ca on 2026-11-07; all values fictional."""
    base = "https://content.echo360.ca/0000.inst/media-1/1"
    return {
        "title": "Fictional Section - Fall", "chapters": None, "locale": "en",
        "captions": "https://captions.echo360.ca/inst/captions-media-1-20260105103000.vtt",
        "startTimeMillis": 2920000,  # playback resume position, not a trim
        "context": {"lessonId": lesson_id},
        "lesson": {"id": lesson_id, "name": "Fictional Lecture", "displayName": "Fictional Lecture",
                   "lessonHasName": True,
                   "timing": {"start": "2026-01-05T10:30:00.000", "end": "2026-01-05T11:20:00.000"},
                   "timeZone": {"id": "US/Eastern", "name": "(-05:00) US/Eastern",
                                "standardOffset": "PT-5H"}},
        "video": {"mediaId": "media-1", "duration": "PT3240.045S", "playableMedias": [
            {"sourceIndex": 0, "trackType": ["Audio"], "uri": f"{base}/s0_a.m3u8?x=1"},
            {"sourceIndex": 1, "trackType": ["Video"], "uri": f"{base}/s1_v.m3u8?x=1"},
            {"sourceIndex": 1, "trackType": ["Audio", "Video"], "uri": f"{base}/s1_av.m3u8?x=1"},
            {"sourceIndex": 2, "trackType": ["Video"], "uri": f"{base}/s2_v.m3u8?x=1"},
            {"sourceIndex": 2, "trackType": ["Audio", "Video"], "uri": f"{base}/s2_av.m3u8?x=1"}]},
        "sectionInfo": {"course": {"courseName": "FICT 101: Fictional Course"},
                        "term": {"name": "2026/2027",
                                 "session": {"startDate": "2026-01-01T00:00:00"}}},
    }


def test_observed_player_app_object_is_authoritative():
    import json as _json

    from echo360_dl.echo360.metadata import extract_metadata

    lesson = "G_fictional-lesson-0002"
    html = ('<script>Echo["echoPlayerV2FullApp"]('
            f"{_json.dumps(_json.dumps(_player_app(lesson)))});</script>")
    meta = extract_metadata([], lesson, html=html)
    assert meta.title == "Fictional Lecture"
    assert meta.course == "FICT 101: Fictional Course"
    assert meta.recorded_at == "2026-01-05T10:30:00-05:00"  # lesson timing, not term start
    assert meta.duration == pytest.approx(3240.045)
    assert meta.media_ids[0] == "media-1"
    assert [u.split("?")[0].rsplit("/", 1)[1] for u in meta.playlist_urls[:4]] == \
        ["s1_av.m3u8", "s2_av.m3u8", "s1_v.m3u8", "s2_v.m3u8"]
    assert meta.listed_source_indexes == {1, 2}  # s0_a is the shared audio, not a source
    messages = [n.message for n in meta.notes]
    assert any("echoPlayerV2FullApp" in n.message and n.observed for n in meta.notes)
    assert not any(m.startswith(("Title read from", "Recording start read from",
                                 "Duration read from", "Course read from")) for m in messages)


def test_player_app_for_another_lesson_is_ignored():
    import json as _json

    from echo360_dl.echo360.metadata import extract_metadata

    html = ('<script>Echo["echoPlayerV2FullApp"]('
            f"{_json.dumps(_json.dumps(_player_app('G_other-lesson')))});</script>")
    meta = extract_metadata([], "G_fictional-lesson-0002", html=html)
    assert meta.title != "Fictional Lecture"
    assert meta.recorded_at != "2026-01-05T10:30:00-05:00"


@pytest.mark.parametrize(("text", "seconds"), [
    ("PT3240.045S", 3240.045), ("PT1H2M3S", 3723.0), ("PT45M", 2700.0), ("P1DT1S", 86401.0)])
def test_parse_iso_duration(text, seconds):
    from echo360_dl.echo360.metadata import parse_iso_duration

    assert parse_iso_duration(text) == pytest.approx(seconds)
    assert parse_iso_duration("P") is None and parse_iso_duration("3240") is None
