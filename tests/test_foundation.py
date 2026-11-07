"""Tests for shared contracts: selection, timecodes, redaction, auth scoping, JSON."""

from __future__ import annotations

import json
import stat

import pytest

from threech0.errors import ExitCode, QualityUnavailableError, UsageError
from threech0.models import (
    AuthContext,
    CaptionTrack,
    ChapterModeKind,
    Cookie,
    DownloadResult,
    Failure,
    Recording,
    ResultStatus,
    SubtitleModeKind,
    TranscriptProvenance,
    VideoRendition,
    VideoSource,
    to_jsonable,
)
from threech0.redact import public_url, scrub
from threech0.selection import (
    iso639_2,
    language_matches,
    parse_chapters,
    parse_quality,
    parse_streams,
    parse_subtitles,
    resolve_streams,
    select_rendition,
)
from threech0.timecode import format_clock, parse_time


def _source(index: int, heights: list[int]) -> VideoSource:
    return VideoSource(index=index, playlist_url=f"https://content.example/s{index}_av.m3u8",
                       renditions=[VideoRendition(id=f"s{index}q{i}", url=f"u{i}", height=h,
                                                  width=h * 16 // 9, bandwidth=h * 1000)
                                   for i, h in enumerate(heights)])


def test_parse_streams_variants():
    assert parse_streams("all").all_sources
    assert parse_streams("audio").audio_only
    assert parse_streams("2").source_indexes == (2,)
    assert parse_streams("1, 2,2").source_indexes == (1, 2)
    assert parse_streams("source1,s2").source_indexes == (1, 2)
    with pytest.raises(UsageError):
        parse_streams("none")  # not an alias for audio-only
    with pytest.raises(UsageError):
        parse_streams("camera")


def test_resolve_streams_unknown_source():
    rec = Recording(lesson_id="L", classroom_url="u", host="echo360.ca",
                    sources=[_source(1, [360]), _source(2, [720])])
    assert resolve_streams(parse_streams("all"), rec) == (1, 2)
    with pytest.raises(Exception, match="Source 3 not found"):
        resolve_streams(parse_streams("3"), rec)
    from threech0.models import AccessState
    from threech0.selection import skipped_by_all
    rec.sources[1].access = AccessState.FORBIDDEN
    assert resolve_streams(parse_streams("all"), rec) == (1,)
    assert [s.index for s in skipped_by_all(rec)] == [2]
    assert resolve_streams(parse_streams("2"), rec) == (2,)  # explicit pick still allowed
    rec.sources[0].access = AccessState.FORBIDDEN
    with pytest.raises(Exception, match="No accessible video source"):
        resolve_streams(parse_streams("all"), rec)


@pytest.mark.parametrize(("spec", "expected"), [
    ("best", 1080), ("worst", 360), ("720p", 720), ("721", 720), ("480p", 360)])
def test_quality_selection(spec, expected):
    assert select_rendition(_source(1, [360, 720, 1080]), parse_quality(spec)).height == expected


def test_quality_below_all_renditions_is_clear_error():
    with pytest.raises(QualityUnavailableError, match="available: 360p, 720p"):
        select_rendition(_source(1, [360, 720]), parse_quality("240p"))
    with pytest.raises(UsageError):
        parse_quality("hd")


def test_subtitle_and_chapter_modes(tmp_path):
    assert parse_subtitles("auto").kind is SubtitleModeKind.AUTO
    assert parse_subtitles("none").kind is SubtitleModeKind.NONE
    assert parse_subtitles("en,fr-CA").languages == ("en", "fr-CA")
    with pytest.raises(UsageError):
        parse_subtitles("english!")
    assert parse_chapters("auto").kind is ChapterModeKind.AUTO
    assert parse_chapters("auto", file=str(tmp_path / "c.json")).kind is ChapterModeKind.FILE
    assert parse_chapters("interval", interval="5m").interval == 300
    with pytest.raises(UsageError):
        parse_chapters("none", file="x.json")
    with pytest.raises(UsageError):
        parse_chapters("interval", interval="10s")


def test_language_helpers():
    assert language_matches("en", "en-CA")
    assert not language_matches("en-GB", "en-CA")
    assert iso639_2("en-CA") == "eng" and iso639_2(None) == "und" and iso639_2("xx") == "und"


@pytest.mark.parametrize(("text", "seconds"), [
    ("00:08:30", 510), ("8:30", 510), ("90", 90), ("1h2m3s", 3723), ("10m", 600),
    ("00:00:01.250", 1.25), ("00:00:01,5", 1.5), (12, 12)])
def test_parse_time(text, seconds):
    assert parse_time(text) == pytest.approx(seconds)


@pytest.mark.parametrize("bad", ["", "-1", "00:61:00", "1:2:3:4", "abc", "00:00:75", "nan"])
def test_parse_time_rejects(bad):
    with pytest.raises(ValueError):
        parse_time(bad)


def test_format_clock():
    assert format_clock(3725.5, millis=True) == "01:02:05.500"
    assert format_clock(59.9996, millis=True, sep=",") == "00:01:00,000"


def test_cookie_scope_host_only_vs_domain():
    host_only = Cookie("a", "1", "content.echo360.ca")
    domain = Cookie("b", "2", ".echo360.ca", path="/")
    pathed = Cookie("c", "3", ".echo360.ca", path="/api")
    insecure_target = "http://content.echo360.ca/x"
    assert host_only.matches("https://content.echo360.ca/x")
    assert not host_only.matches("https://echo360.ca/x")
    assert domain.matches("https://echo360.ca/x") and domain.matches("https://a.echo360.ca/")
    assert not domain.matches("https://evil-echo360.ca/x")
    assert not domain.matches("https://echo360.ca.evil.com/x")
    assert not host_only.matches(insecure_target)
    assert pathed.matches("https://echo360.ca/api/x") and not pathed.matches(
        "https://echo360.ca/apix")


def test_auth_context_never_leaks(tmp_path):
    ctx = AuthContext(cookies=[Cookie("CloudFront-Signature", "SECRETVALUE123", ".echo360.ca",
                                      expires=4_000_000_000)],
                      headers={"Cookie": "x=SECRETVALUE123", "User-Agent": "UA",
                               "Authorization": "Bearer SECRETVALUE123"})
    assert "SECRETVALUE123" not in repr(ctx) and "SECRETVALUE123" not in str(ctx)
    assert set(ctx.headers) == {"user-agent"}
    with pytest.raises(TypeError):
        to_jsonable(ctx)
    jar = ctx.write_netscape_cookiejar(tmp_path / "jar.txt")
    assert stat.S_IMODE(jar.stat().st_mode) == 0o600
    line = next(ln for ln in jar.read_text().splitlines() if not ln.startswith("#")).split("\t")
    assert line[0] == ".echo360.ca" and line[1] == "TRUE"
    import http.cookiejar
    loaded = http.cookiejar.MozillaCookieJar(str(jar))
    loaded.load()
    assert [c.name for c in loaded] == ["CloudFront-Signature"]


def test_expired_cookie_excluded():
    ctx = AuthContext(cookies=[Cookie("x", "v", "content.echo360.ca", expires=1)])
    assert ctx.cookies_for("https://content.echo360.ca/") == []


def test_scrub_and_public_url():
    text = ("GET https://content.echo360.ca/a/s1_av.m3u8?x-uid=abc&Signature=zz Cookie: "
            "CloudFront-Policy=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijk")
    out = scrub(text)
    assert "abc" not in out.split("?")[1] and "eyJ" not in out
    assert public_url("https://u:p@content.echo360.ca/a?b=c#d") == "https://content.echo360.ca/a"


def test_jsonable_strips_query_and_lists_format_names():
    rec = Recording(lesson_id="L", classroom_url="https://echo360.ca/lesson/L/classroom",
                    host="echo360.ca", sources=[_source(1, [720])])
    rec.sources[0].playlist_url = "https://content.echo360.ca/s1_av.m3u8?x-uid=secret"
    rec.captions.append(CaptionTrack(id="c", language="en",
                                     kind=TranscriptProvenance.PLATFORM_CAPTIONS,
                                     formats={"vtt": "https://x/y?token=secret"}))
    data = json.dumps(to_jsonable(rec))
    assert "secret" not in data
    assert '"display_name": "Source 1"' in data and '"formats": ["vtt"]' in data


def test_result_exit_codes():
    assert DownloadResult(ResultStatus.OK).exit_code == ExitCode.OK
    assert DownloadResult(ResultStatus.PARTIAL).exit_code == ExitCode.PARTIAL
    failed = DownloadResult(ResultStatus.FAILED,
                            failures=[Failure("download", "Source 1", "auth-expired", "x")])
    assert failed.exit_code == ExitCode.AUTH


def test_probed_403_source_gives_forbidden_not_missing():
    from threech0.errors import ExitCode, ForbiddenError
    from threech0.models import AccessState

    rec = Recording(lesson_id="L", classroom_url="u", host="echo360.ca",
                    sources=[_source(1, [720])],
                    unreachable_sources={3: AccessState.FORBIDDEN})
    with pytest.raises(ForbiddenError, match="403") as info:
        resolve_streams(parse_streams("3"), rec)
    assert info.value.exit_code == ExitCode.AUTH
    with pytest.raises(Exception, match="Source 5 not found"):
        resolve_streams(parse_streams("5"), rec)
