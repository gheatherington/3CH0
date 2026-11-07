from __future__ import annotations

import pytest

from threech0.echo360.hls import (
    PlaylistError,
    build_audio_tracks,
    parse_attributes,
    parse_master,
    resolve_uri,
)

from .conftest import CONTENT_BASE, LESSON, SECRET_QUERY, master_text, playlist_url


def test_duplicated_codecs_keeps_fuller_value():
    attrs = parse_attributes('BANDWIDTH=1,CODECS="avc1.640029",AUDIO="q0",'
                             'CODECS="avc1.640029,mp4a.40.2",FRAME-RATE=30.0')
    assert attrs["CODECS"] == "avc1.640029,mp4a.40.2"
    attrs = parse_attributes('CODECS="avc1.640029,mp4a.40.2",CODECS="avc1.640029"')
    assert attrs["CODECS"] == "avc1.640029,mp4a.40.2"


def test_parse_observed_master_fixture():
    url = playlist_url(1)
    master = parse_master(master_text(1), url)
    assert master.is_master and not master.drm
    assert [v.id for v in master.variants] == ["s1q0", "s1q1"]
    low, high = master.variants
    assert (low.width, low.height, low.bandwidth, low.average_bandwidth) == (640, 360, 93350,
                                                                             25941)
    assert high.resolution == "1280x720" and high.frame_rate == 30.0
    assert high.codecs == "avc1.640029,mp4a.40.2"
    assert (low.audio_group, high.audio_group) == ("q0", "q1")
    assert high.url == f"{CONTENT_BASE}/s1q1.m3u8?{SECRET_QUERY}"
    assert [a.id for a in master.audio] == ["s0q0", "s0q1"]
    assert all(a.default and a.name == "Default" for a in master.audio)
    assert master.audio[1].url == f"{CONTENT_BASE}/s0q1.m3u8?{SECRET_QUERY}"
    assert f"x-lid={LESSON}" in master.audio[1].url


def test_resolve_uri_query_rules():
    base = "https://content.echo360.ca/a/b/s1_av.m3u8?x-lid=L&x-mid=M"
    assert resolve_uri("s1q0.m3u8", base) == "https://content.echo360.ca/a/b/s1q0.m3u8?x-lid=L&x-mid=M"
    assert resolve_uri("../c/s1q0.m3u8", base) == \
        "https://content.echo360.ca/a/c/s1q0.m3u8?x-lid=L&x-mid=M"
    assert resolve_uri("s1q0.m3u8?own=1", base) == "https://content.echo360.ca/a/b/s1q0.m3u8?own=1"
    # The signed query never travels to another host.
    assert resolve_uri("https://other.example/x.m3u8", base) == "https://other.example/x.m3u8"


def test_shared_audio_detected_across_sources():
    masters = [(1, parse_master(master_text(1), playlist_url(1))),
               (2, parse_master(master_text(2), playlist_url(2)))]
    tracks, mapping = build_audio_tracks(masters)
    assert len(tracks) == 1
    track = tracks[0]
    assert track.id == "s0" and track.shared and track.used_by_sources == [1, 2]
    assert [r.id for r in track.renditions] == ["s0q0", "s0q1"]
    assert mapping == {1: "s0", 2: "s0"}


def test_distinct_audio_not_shared():
    other = master_text(2).replace("s0q0", "s9q0").replace("s0q1", "s9q1")
    masters = [(1, parse_master(master_text(1), playlist_url(1))),
               (2, parse_master(other, playlist_url(2)))]
    tracks, mapping = build_audio_tracks(masters)
    assert [t.id for t in tracks] == ["s0", "s9"]
    assert not any(t.shared for t in tracks)
    assert mapping == {1: "s0", 2: "s9"}


def test_media_playlist_and_garbage():
    media = parse_master("#EXTM3U\n#EXTINF:10.0,\nseg.mp4\n", playlist_url(1))
    assert not media.is_master and media.variants == []
    with pytest.raises(PlaylistError) as info:
        parse_master("<html>Access Denied</html>", playlist_url(1))
    assert "x-uid" not in str(info.value) and "?" not in str(info.value)


def test_drm_and_subtitles_flagged():
    text = master_text(1) + (
        '\n#EXT-X-SESSION-KEY:METHOD=SAMPLE-AES,URI="skd://x",KEYFORMAT="com.apple.streamingkeydelivery"'
        '\n#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs",NAME="English",URI="subs.m3u8"\n')
    master = parse_master(text, playlist_url(1))
    assert master.drm and master.subtitle_groups == 1
