"""HLS master playlist parsing (observed Echo360 shape; see docs/dev/BRIEFING.md).

* Relative URIs resolve against the playlist URL and inherit its query string when they
  have none and stay on the same host (observed content URLs carry ``x-*`` parameters).
* A duplicated ``CODECS`` attribute (observed) is tolerated: the fuller value wins.
* Audio renditions referenced by several sources (observed: ``s1_av``/``s2_av`` both use
  ``s0q0``/``s0q1``) become one shared :class:`AudioTrack`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit, urlunsplit

from threech0.models import AudioRendition, AudioTrack, VideoRendition
from threech0.redact import public_url

_ATTR = re.compile(r'([A-Z0-9-]+)=("[^"]*"|[^,]*)')
_DRM_KEYFORMATS = ("widevine", "playready", "fairplay", "com.apple.streamingkeydelivery")


class PlaylistError(ValueError):
    """The text is not an HLS playlist we can use (message contains no secrets)."""


@dataclass
class MasterPlaylist:
    url: str
    variants: list[VideoRendition] = field(default_factory=list)
    audio: list[AudioRendition] = field(default_factory=list)
    is_master: bool = True
    drm: bool = False
    subtitle_groups: int = 0  # EXT-X-MEDIA TYPE=SUBTITLES entries (not consumed)


def parse_attributes(text: str) -> dict[str, str]:
    """Parse an HLS attribute list. Duplicate ``CODECS``: keep the fuller value;
    other duplicates: the last one wins."""
    attrs: dict[str, str] = {}
    for key, raw in _ATTR.findall(text):
        value = raw[1:-1] if raw.startswith('"') and raw.endswith('"') else raw.strip()
        if key == "CODECS" and key in attrs:
            old = attrs[key]
            if len([c for c in value.split(",") if c.strip()]) <= len(
                    [c for c in old.split(",") if c.strip()]):
                continue
        attrs[key] = value
    return attrs


def resolve_uri(uri: str, base_url: str) -> str:
    """Resolve ``uri`` against ``base_url``; carry the base query over when the result has
    none and is on the same scheme and host (needed for observed ``x-*`` parameters)."""
    resolved = urljoin(base_url, uri.strip())
    base = urlsplit(base_url)
    target = urlsplit(resolved)
    if (not target.query and base.query and target.scheme == base.scheme
            and target.netloc == base.netloc):
        resolved = urlunsplit(target._replace(query=base.query))
    return resolved


def rendition_id(url: str) -> str:
    """``.../s1q1.m3u8?x=..`` -> ``s1q1``."""
    name = urlsplit(url).path.rsplit("/", 1)[-1]
    return re.sub(r"\.m3u8$", "", name, flags=re.IGNORECASE) or name


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except ValueError:
        return None


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def parse_master(text: str, url: str) -> MasterPlaylist:
    """Parse a master playlist. A media playlist yields ``is_master=False`` with no variants."""
    if not text.lstrip("﻿").lstrip().startswith("#EXTM3U"):
        raise PlaylistError(f"Response from {public_url(url)} is not an HLS playlist.")
    result = MasterPlaylist(url=url)
    pending: dict[str, str] | None = None
    seen_ids: set[str] = set()
    for line in (ln.strip() for ln in text.splitlines()):
        if not line:
            continue
        if line.startswith("#EXT-X-STREAM-INF:"):
            pending = parse_attributes(line.split(":", 1)[1])
            continue
        if line.startswith("#EXT-X-MEDIA:"):
            attrs = parse_attributes(line.split(":", 1)[1])
            mtype = attrs.get("TYPE", "").upper()
            if mtype == "SUBTITLES":
                result.subtitle_groups += 1
            if mtype != "AUDIO" or not attrs.get("URI"):
                continue  # audio without URI is muxed into the variant
            audio_url = resolve_uri(attrs["URI"], url)
            rid = rendition_id(audio_url)
            if rid in {a.id for a in result.audio}:
                continue
            result.audio.append(AudioRendition(
                id=rid, url=audio_url, group_id=attrs.get("GROUP-ID"), name=attrs.get("NAME"),
                language=attrs.get("LANGUAGE") or None,
                default=attrs.get("DEFAULT", "").upper() == "YES"))
            continue
        if line.startswith(("#EXT-X-SESSION-KEY:", "#EXT-X-KEY:")):
            attrs = parse_attributes(line.split(":", 1)[1])
            method = attrs.get("METHOD", "NONE").upper()
            keyformat = attrs.get("KEYFORMAT", "").lower()
            if method == "SAMPLE-AES" or method.startswith("SAMPLE-AES") or any(
                    k in keyformat for k in _DRM_KEYFORMATS):
                result.drm = True
            continue
        if line.startswith("#"):
            continue
        if pending is not None:
            variant_url = resolve_uri(line, url)
            rid = rendition_id(variant_url)
            base_id, n = rid, 2
            while rid in seen_ids:
                rid, n = f"{base_id}-{n}", n + 1
            seen_ids.add(rid)
            width = height = None
            res = re.fullmatch(r"(\d+)x(\d+)", pending.get("RESOLUTION", ""))
            if res:
                width, height = int(res[1]), int(res[2])
            result.variants.append(VideoRendition(
                id=rid, url=variant_url, width=width, height=height,
                bandwidth=_int(pending.get("BANDWIDTH")),
                average_bandwidth=_int(pending.get("AVERAGE-BANDWIDTH")),
                codecs=pending.get("CODECS") or None,
                frame_rate=_float(pending.get("FRAME-RATE")),
                audio_group=pending.get("AUDIO") or None))
            pending = None
    if not result.variants:
        result.is_master = False
    return result


def build_audio_tracks(masters: Iterable[tuple[int, MasterPlaylist]]
                       ) -> tuple[list[AudioTrack], dict[int, str]]:
    """Group audio renditions into logical tracks.

    Sources whose masters reference any of the same audio playlists (compared without
    query string) share one track. Returns the tracks and a source-index -> track-id map.
    """
    items = [(idx, m) for idx, m in masters if m.audio]
    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner: dict[str, int] = {}
    for i, (_idx, master) in enumerate(items):
        for rend in master.audio:
            key = public_url(rend.url)
            if key in owner:
                parent[find(i)] = find(owner[key])
            else:
                owner[key] = i
    groups: dict[int, list[int]] = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(i)

    tracks: list[AudioTrack] = []
    mapping: dict[int, str] = {}
    used_ids: set[str] = set()
    for members in sorted(groups.values(), key=lambda m: min(items[i][0] for i in m)):
        renditions: dict[str, AudioRendition] = {}
        for i in members:
            for rend in items[i][1].audio:
                renditions.setdefault(public_url(rend.url), rend)
        rends = list(renditions.values())
        prefixes = {m[1] for r in rends if (m := re.fullmatch(r"(s\d+)q\d+", r.id))}
        track_id = prefixes.pop() if len(prefixes) == 1 and all(
            re.fullmatch(r"s\d+q\d+", r.id) for r in rends) else f"audio{len(tracks) + 1}"
        while track_id in used_ids:
            track_id = f"{track_id}-{len(tracks) + 1}"
        used_ids.add(track_id)
        languages = {r.language for r in rends}
        sources = sorted({items[i][0] for i in members})
        tracks.append(AudioTrack(id=track_id, renditions=rends,
                                 language=languages.pop() if len(languages) == 1 else None,
                                 shared=len(sources) > 1, used_by_sources=sources))
        for idx in sources:
            mapping[idx] = track_id
    return tracks, mapping
