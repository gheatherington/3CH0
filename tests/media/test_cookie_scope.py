"""yt-dlp cookie jars: per-URL selection, private location, strict host-only matching."""

from __future__ import annotations

import http.cookiejar
import os
import stat
import urllib.request

from threech0.media import download
from threech0.media.download import Component, WorkDir
from threech0.models import AuthContext, Cookie

CONTENT = "https://content.echo360.ca/0000.i/m/1/s1q1.m3u8"


def _auth() -> AuthContext:
    return AuthContext(cookies=[
        Cookie("CloudFront-Signature", "SIGVALUE123", "content.echo360.ca"),
        Cookie("ECHO_JWT", "JWTVALUE123", "echo360.ca"),  # host-only on the app host
        Cookie("lb", "LBVALUE123", ".echo360.ca"),  # domain cookie: browsers send it
    ])


def test_jar_contains_only_cookies_for_the_component_url(tmp_path):
    jar = _auth().write_netscape_cookiejar(tmp_path / "j.txt", for_url=CONTENT)
    text = jar.read_text()
    assert "SIGVALUE123" in text and "LBVALUE123" in text
    assert "JWTVALUE123" not in text


def test_jar_lives_outside_the_output_tree_and_is_removed(tmp_path):
    work = WorkDir(tmp_path / "out" / ".3ch0-x")
    comp = next(iter([c for c in [_component()]]))
    with work.cookie_jar(comp, _auth()) as jar:
        assert tmp_path not in jar.parents
        assert stat.S_IMODE(jar.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(jar.stat().st_mode) == 0o600
    assert not jar.exists()
    work.remove_cookie_files()
    assert not jar.parent.exists()


def test_stale_jar_dirs_from_dead_processes_are_swept(tmp_path, monkeypatch):
    monkeypatch.setattr(download.tempfile, "gettempdir", lambda: str(tmp_path))
    dead = tmp_path / f"{download._JAR_PREFIX}999999-abc"
    alive = tmp_path / f"{download._JAR_PREFIX}{os.getpid()}-abc"
    dead.mkdir()
    alive.mkdir()
    download.sweep_stale_jar_dirs()
    assert not dead.exists() and alive.exists()


def test_strict_policy_blocks_host_only_cookie_on_subdomain(tmp_path):
    jar_path = AuthContext(cookies=[Cookie("ECHO_JWT", "JWTVALUE123", "echo360.ca")]) \
        .write_netscape_cookiejar(tmp_path / "j.txt")
    jar = http.cookiejar.MozillaCookieJar(str(jar_path))
    jar.load(ignore_discard=True, ignore_expires=True)  # as yt-dlp loads session cookies
    jar.set_policy(http.cookiejar.DefaultCookiePolicy(
        strict_ns_domain=http.cookiejar.DefaultCookiePolicy.DomainStrictNonDomain))
    for url, expected in (("https://echo360.ca/x", True), (CONTENT, False)):
        req = urllib.request.Request(url)  # noqa: S310 - fixed https test URLs
        jar.add_cookie_header(req)
        assert ("JWTVALUE123" in (req.get_header("Cookie") or "")) is expected


def _component() -> Component:
    import dataclasses

    fields = {f.name for f in dataclasses.fields(Component)}
    values = {"key": "s1q1", "url": CONTENT, "label": "Source 1", "kind": "video",
              "rendition_id": "s1q1", "filename": "s1q1.mp4"}
    return Component(**{k: v for k, v in values.items() if k in fields})


def test_yt_dlp_jar_applies_strict_policy(tmp_path):
    """The real yt-dlp jar, configured as download.py does, keeps host-only cookies on
    their host."""
    import yt_dlp

    jar_path = _auth().write_netscape_cookiejar(tmp_path / "j.txt")
    with yt_dlp.YoutubeDL({"cookiefile": str(jar_path), "quiet": True}) as ydl:
        ydl.cookiejar.set_policy(http.cookiejar.DefaultCookiePolicy(
            strict_ns_domain=http.cookiejar.DefaultCookiePolicy.DomainStrictNonDomain))
        header = ydl.cookiejar.get_cookie_header(CONTENT)
        assert "SIGVALUE123" in header and "JWTVALUE123" not in header
        assert "JWTVALUE123" in ydl.cookiejar.get_cookie_header("https://echo360.ca/a")
