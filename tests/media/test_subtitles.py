from __future__ import annotations

import json

import pytest

from tests.media.conftest import FIXTURES
from threech0.errors import TranscriptUnavailableError, UnavailableError, ValidationError
from threech0.media.subtitles import (
    detect_format,
    loses_information,
    normalize_cues,
    parse_subtitle_bytes,
    to_json,
    to_srt,
    to_txt,
    to_vtt,
    track_title,
    validate_cues,
)
from threech0.models import Cue, Transcript, TranscriptProvenance


def _t(cues, provenance=TranscriptProvenance.PLATFORM_CAPTIONS, language="en"):
    return Transcript(language=language, provenance=provenance, cues=cues)


def test_parse_vtt_with_header_style_note_settings_and_voices():
    cues = parse_subtitle_bytes((FIXTURES / "captions_en.vtt").read_bytes(), None)
    assert [(c.start, c.end) for c in cues] == [(1.0, 2.5), (3.0, 4.0), (4.5, 5.5)]
    assert cues[0].speaker == "Dr. Alice Smith"
    assert cues[0].text == "Welcome & hello — café"
    assert cues[1].speaker == "Bob" and cues[1].text == "Limits <of> functions"
    assert cues[2].speaker is None and cues[2].text == "Plain cue with italics"


def test_parse_srt_crlf_multiline():
    cues = parse_subtitle_bytes((FIXTURES / "captions_fr.srt").read_bytes(), "srt")
    assert cues == [Cue("Bonjour à tous", 1.2, 2.0),
                    Cue("Deuxième ligne\nsur deux lignes", 3.0, 4.25)]


@pytest.mark.parametrize("encoding,bom", [("utf-8", b"\xef\xbb\xbf"), ("utf-16-le", b"\xff\xfe"),
                                          ("utf-16-be", b"\xfe\xff"), ("utf-16-le", b""),
                                          ("utf-8", b"")])
def test_parse_encodings_and_bom(encoding, bom):
    text = "WEBVTT\r\n\r\n00:01.000 --> 00:02.000\r\nnaïve 日本語 ✓\r\n"
    cues = parse_subtitle_bytes(bom + text.encode(encoding), None)
    assert cues == [Cue("naïve 日本語 ✓", 1.0, 2.0)]


def test_detect_formats():
    assert detect_format("WEBVTT\n") == "vtt"
    assert detect_format("1\n00:00:01,000 --> 00:00:02,000\nx\n") == "srt"
    assert detect_format('{"cues": []}') == "json"
    assert detect_format("hello\nworld") == "txt"


def test_plain_text_is_untimed_and_never_gets_timestamps():
    cues = parse_subtitle_bytes((FIXTURES / "transcript_untimed.txt").read_bytes(), "txt")
    assert len(cues) == 2 and all(c.start is None and c.end is None for c in cues)
    t = _t(cues, TranscriptProvenance.PLATFORM_TRANSCRIPT)
    assert not t.timed
    with pytest.raises(TranscriptUnavailableError) as exc:
        to_vtt(t)
    assert isinstance(exc.value, UnavailableError) and "no timestamps" in exc.value.message
    with pytest.raises(UnavailableError):
        to_srt(t)
    assert to_txt(t) == "This is an untimed transcript line.\nSecond line of the transcript.\n"


@pytest.mark.parametrize("data", [
    b"",
    b"\x00\x01\x02\x03binary\xff\xfe\x00",
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
    b"WEBVTT\n\n00:00:01.000 --> nonsense\nhello\n",
    b"WEBVTT\n\n00:00:61.000 --> 00:00:62.000\nbad seconds\n",
    b"1\n00:00:01,000 -> 00:00:02,000\nmissing arrow\n",
    b"plain text with a bell \x07 inside",
    b"\xc3\x28 invalid utf8 \xa0\xa1",
])
def test_garbage_raises_validation_error(data):
    fmt = "srt" if data.startswith(b"1\n") else None
    with pytest.raises(ValidationError):
        parse_subtitle_bytes(data, fmt)


def test_unsupported_format_name():
    with pytest.raises(ValidationError):
        parse_subtitle_bytes(b"hello", "ass")


def test_vtt_round_trip_keeps_speaker_and_escapes():
    t = _t([Cue("a < b & c --> d", 1.0, 2.0, speaker="Ana <TA>"), Cue("x\n\ny", 2.0, 3.25)])
    vtt = to_vtt(t)
    assert "<v Ana &lt;TA&gt;>a &lt; b &amp; c --&gt; d" in vtt
    assert "00:00:02.000 --> 00:00:03.250\nx\ny\n" in vtt
    back = parse_subtitle_bytes(vtt.encode(), "vtt")
    assert back[0] == Cue("a < b & c --> d", 1.0, 2.0, speaker="Ana <TA>")
    assert back[1] == Cue("x\ny", 2.0, 3.25)
    assert "Ana <TA>: a < b" in to_vtt(t, voice_tags=False).replace("&lt;", "<") \
        .replace("&gt;", ">")


def test_srt_txt_json_exports():
    t = _t([Cue("Hello", 1.0, 2.0, speaker="Alice"), Cue("World", 3.5, 4.0)])
    assert to_srt(t) == ("1\n00:00:01,000 --> 00:00:02,000\nAlice: Hello\n\n"
                         "2\n00:00:03,500 --> 00:00:04,000\nWorld\n")
    assert to_txt(t) == "Alice: Hello\nWorld\n"
    data = json.loads(to_json(t))
    assert data["version"] == 1 and data["provenance"] == "platform-captions"
    assert data["timed"] is True and data["cues"][0]["speaker"] == "Alice"
    assert parse_subtitle_bytes(to_json(t).encode(), "json") == t.cues


def test_validate_cues_rules():
    cues = [Cue("ok", 0.0, 1.0), Cue("", 1.0, 2.0), Cue("back", 0.5, 0.4),
            Cue("ctl\x01", 3.0, 4.0), Cue("over", 3.5, 9.0), Cue("neg", -1.0, 0.5)]
    problems = validate_cues(cues, 6.0, allow_overlap=False)
    text = "\n".join(problems)
    assert "cue 2: empty text" in text
    assert "cue 3: end (0.400) must be after start" in text
    assert "cue 3: starts before the previous cue" in text
    assert "cue 4: control characters U+0001" in text
    assert "cue 5: overlaps the previous cue" in text
    assert "cue 5: ends at 9.000s, after the media end" in text
    assert "cue 6: negative start" in text
    assert validate_cues([Cue("a", 0, 2), Cue("b", 1, 3)], 6.0) == []  # overlap allowed
    assert "some cues have timestamps" in validate_cues([Cue("a", 0, 1), Cue("b")], None)[0]


def test_normalize_cues_never_invents_timestamps():
    cues = [Cue("b", 2.0, 5.0), Cue("a", 0.0, 3.0), Cue("  ", 1, 2), Cue("c", 4.0, 9.0)]
    out = normalize_cues(cues, 6.0, trim_overlaps=True)
    assert out == [Cue("a", 0.0, 2.0), Cue("b", 2.0, 4.0), Cue("c", 4.0, 6.0)]
    untimed = normalize_cues([Cue("x"), Cue(" ")])
    assert untimed == [Cue("x")]


def test_loses_information():
    styled = (FIXTURES / "captions_en.vtt").read_bytes()
    assert loses_information(styled, "vtt")
    assert not loses_information(b"WEBVTT\n\n00:01.000 --> 00:02.000\n<v A>hi\n", "vtt")
    assert loses_information(b'{"segments": []}', "json")
    assert loses_information(b"1\n00:00:01,000 --> 00:00:02,000\n<i>x</i>\n", "srt")
    assert not loses_information(b"plain", "txt")


def test_track_titles():
    assert track_title(_t([])) == "English — Echo360 captions"
    assert track_title(_t([], TranscriptProvenance.GENERATED_LOCAL, "fr")) == \
        "French — generated (local)"
    assert track_title(_t([], TranscriptProvenance.PLATFORM_TRANSCRIPT, None)) == \
        "Echo360 transcript"  # no "Unknown language — " prefix


def test_platform_json_sidecar_is_scrubbed_but_captions_kept_verbatim():
    from threech0.media.pipeline import _scrubbed_raw

    raw = b'{"cues": [], "src": "https://x.echo360.ca/a.vtt?Signature=SECRETSIG123"}'
    assert b"SECRETSIG123" not in _scrubbed_raw(raw, "json")
    vtt = b"WEBVTT\n\n00:00.000 --> 00:01.000\nsee https://example.com/?a=1\n"
    assert _scrubbed_raw(vtt, "vtt") == vtt


def test_duplicate_platform_transcript_is_embedded_once():
    from threech0.media.pipeline import dedupe_platform_tracks
    from threech0.models import Cue, Transcript, TranscriptProvenance

    cues = [Cue("Welcome back", 1.0, 2.0), Cue("Today: limits", 2.0, 4.0)]
    captions = Transcript(None, TranscriptProvenance.PLATFORM_CAPTIONS, cues)
    same = Transcript(None, TranscriptProvenance.PLATFORM_TRANSCRIPT,
                      [Cue("welcome  back", 1.1, 2.0), Cue("Today: limits", 2.2, 4.0)])
    different = Transcript(None, TranscriptProvenance.PLATFORM_TRANSCRIPT,
                           [Cue("Something else entirely", 1.0, 2.0), Cue("x", 9.0, 10.0)])
    keep, dropped = dedupe_platform_tracks([captions, same])
    assert keep == [captions] and dropped == [same]
    keep, dropped = dedupe_platform_tracks([captions, different])
    assert keep == [captions, different] and dropped == []
    other_lang = Transcript("fr", TranscriptProvenance.PLATFORM_TRANSCRIPT, cues)
    assert dedupe_platform_tracks([captions, other_lang])[1] == []


def test_merged_duplicate_marks_captions_automatic():
    from threech0.media.pipeline import dedupe_platform_tracks
    from threech0.media.subtitles import track_title
    from threech0.models import Cue, Transcript, TranscriptProvenance

    cues = [Cue("Welcome back", 1.0, 2.0), Cue("Today: limits", 2.0, 4.0)]
    captions = Transcript("en", TranscriptProvenance.PLATFORM_CAPTIONS, cues)
    auto = Transcript("en", TranscriptProvenance.PLATFORM_TRANSCRIPT, cues,
                      label="Echo360 automatic transcript (speech recognition)")
    (kept,), (dropped,) = dedupe_platform_tracks([captions, auto])
    assert dropped is auto and kept.provenance is TranscriptProvenance.PLATFORM_CAPTIONS
    assert track_title(kept) == "English — Echo360 automatic captions"
