from __future__ import annotations

import os
import subprocess

import pytest

from echo360_dl.echo360.curl_import import (
    cloudfront_policy_expiry,
    load_request_file,
    parse_curl,
)
from echo360_dl.errors import UsageError
from echo360_dl.models import AuthContext

from .conftest import FAKE_JWT, FAKE_SIGNATURE, LESSON, cloudfront_policy, playlist_url

URL = playlist_url(2)
POLICY = cloudfront_policy(4_102_000_000)


def chrome_bash() -> str:
    return (
        f"curl '{URL}' \\\n"
        "  -H 'accept: */*' \\\n"
        "  -H 'accept-language: en-CA,en;q=0.9' \\\n"
        f"  -b 'CloudFront-Policy={POLICY}; CloudFront-Signature={FAKE_SIGNATURE}; "
        f"CloudFront-Key-Pair-Id=FAKEKEYPAIR; ECHO_JWT={FAKE_JWT}' \\\n"
        "  -H 'origin: https://echo360.ca' \\\n"
        "  -H 'referer: https://echo360.ca/' \\\n"
        "  -H 'sec-fetch-mode: cors' \\\n"
        "  -H 'authorization: Bearer FAKE-BEARER-TOKEN' \\\n"
        "  -H 'user-agent: Mozilla/5.0 (Macintosh) FakeBrowser/1.0'\n"
    )


def test_chrome_bash_capture():
    req = parse_curl(chrome_bash())
    assert req.url == URL
    auth = req.auth
    assert auth.source == "request-file"
    assert set(auth.headers) == {"accept-language", "origin", "referer", "user-agent"}
    names = {c.name for c in auth.cookies}
    assert names == {"CloudFront-Policy", "CloudFront-Signature", "CloudFront-Key-Pair-Id",
                     "ECHO_JWT"}
    for c in auth.cookies:
        assert c.domain == "content.echo360.ca" and c.host_only
        assert c.secure and c.expires is None and c.path == "/"
    assert auth.expires_at == 4_102_000_000
    assert FAKE_SIGNATURE not in repr(req) and FAKE_SIGNATURE not in repr(auth)
    assert "x-uid" not in repr(req)


def test_firefox_capture_cookie_header_and_url_flag():
    text = (f"curl --url \"{URL}\" --compressed -H \"User-Agent: FakeFox/2.0\" "
            f"-H \"Cookie: CloudFront-Signature={FAKE_SIGNATURE}; ECHO_JWT={FAKE_JWT}\" "
            "-H \"Referer: https://echo360.ca/\" -H \"Connection: keep-alive\" "
            "--some-unknown-flag -X GET")
    req = parse_curl(text)
    assert req.url == URL
    assert req.auth.headers == {"user-agent": "FakeFox/2.0", "referer": "https://echo360.ca/"}
    assert {c.name for c in req.auth.cookies} == {"CloudFront-Signature", "ECHO_JWT"}
    assert req.auth.expires_at is None


def test_zsh_ansi_c_quoting_and_attached_flags():
    text = (f"curl $'{URL}' -H$'referer: https://echo360.ca/' "
            f"--cookie=$'ECHO_JWT={FAKE_JWT}; note=it\\'s' --header='user-agent: Z/1'")
    req = parse_curl(text)
    assert req.url == URL
    assert req.auth.headers["referer"] == "https://echo360.ca/"
    assert req.auth.headers["user-agent"] == "Z/1"
    cookies = {c.name: c.value for c in req.auth.cookies}
    assert cookies["note"] == "it's"


def test_windows_cmd_capture_is_converted():
    text = (f'curl ^"{URL.replace("&", "^&")}^" ^\n'
            '  -H ^"referer: https://echo360.ca/^" ^\n'
            f'  -b ^"ECHO_JWT={FAKE_JWT}; CloudFront-Policy={POLICY}^" ^\n'
            '  --compressed')
    req = parse_curl(text)
    assert req.url == URL
    assert req.auth.headers == {"referer": "https://echo360.ca/"}
    assert req.auth.expires_at == 4_102_000_000


def test_powershell_capture_rejected():
    with pytest.raises(UsageError, match="PowerShell"):
        parse_curl("Invoke-WebRequest -UseBasicParsing -Uri \"https://x\"")


@pytest.mark.parametrize("text", ["", "   ", "wget https://echo360.ca/x",
                                  "curl -H 'a: b'", "curl 'http://content.echo360.ca/x'",
                                  "curl 'https://x"])
def test_invalid_commands(text):
    with pytest.raises(UsageError):
        parse_curl(text)


def test_cookie_file_reference_is_not_read(tmp_path):
    req = parse_curl(f"curl '{URL}' -b /etc/passwd")
    assert req.auth.cookies == []
    assert any("cookie files are not read" in w for w in req.warnings)


def test_shell_injection_is_never_executed(monkeypatch, tmp_path):
    marker = tmp_path / "pwned"
    spawned: list[object] = []

    def forbid(*args, **kwargs):
        spawned.append(args)
        raise AssertionError("a process was spawned")

    for name in ("Popen", "run", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, forbid)
    for name in ("system", "popen", "execv", "execvp", "spawnv", "posix_spawn",
                 "posix_spawnp", "fork"):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, forbid)

    commands = [
        f"curl '{URL}'; rm -rf ~",
        f"curl '{URL}' && touch {marker}",
        f"curl \"{URL}\" -H \"referer: $(touch {marker})\" | sh",
        f"curl `touch {marker}` '{URL}' -b 'ECHO_JWT=x$(touch {marker})'",
        f"curl $(touch {marker}) '{URL}'",
    ]
    for text in commands:
        req = parse_curl(text)
        assert req.url == URL
        assert isinstance(req.auth, AuthContext)
        for value in req.auth.headers.values():
            assert isinstance(value, str)
    assert spawned == []
    assert not marker.exists()
    # Command substitution text survives only as inert literal data.
    req = parse_curl(f"curl \"{URL}\" -H \"referer: $(touch {marker})\"")
    assert req.auth.headers["referer"] == f"$(touch {marker})"
    req = parse_curl(f"curl '{URL}'; rm -rf ~")
    assert any("nothing was executed" in w for w in req.warnings)


def test_policy_expiry_decoding():
    assert cloudfront_policy_expiry(cloudfront_policy(1_900_000_000)) == 1_900_000_000
    assert cloudfront_policy_expiry("not-base64!!") is None
    assert cloudfront_policy_expiry("") is None
    import base64
    assert cloudfront_policy_expiry(base64.b64encode(b'{"x": 1}').decode()) is None


def test_expired_policy_marks_auth_expired():
    text = f"curl '{URL}' -b 'CloudFront-Policy={cloudfront_policy(1_000_000_000)}'"
    assert parse_curl(text).auth.is_expired()


def test_load_request_file(tmp_path):
    path = tmp_path / "req.txt"
    path.write_text(chrome_bash())
    assert LESSON in load_request_file(path).url
    with pytest.raises(UsageError, match="not found"):
        load_request_file(tmp_path / "missing.txt")
    with pytest.raises(UsageError):
        load_request_file(tmp_path)
