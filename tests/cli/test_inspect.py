from __future__ import annotations

import json

import pytest

from echo360_dl.errors import UnauthorizedError

from .conftest import CLASSROOM, URL, assert_clean, invoke, make_recording


def test_inspect_json_is_clean_single_document(fake):
    result = invoke(["inspect", URL, "--json"])
    assert result.exit_code == 0, result.stderr
    doc = json.loads(result.stdout)
    assert doc["schema_version"] == 1
    assert doc["command"] == "inspect"
    assert doc["ok"] is True
    rec = doc["recording"]
    assert rec["lesson_id"] == "G_fake-lesson-0001"
    assert [s["access"] for s in rec["sources"]] == ["available", "forbidden"]
    assert rec["captions"][0]["formats"] == ["vtt"]
    assert_clean(result.stdout)
    assert_clean(result.stderr)
    # No query strings anywhere in URL values (signed parameters live there).
    assert "?" not in result.stdout
    assert "x-uid" not in result.stdout
    assert "cookies" not in result.stdout
    # JSON never prompts and never opens a login window.
    assert fake.discover_calls[0][1].interactive is False


def test_inspect_json_error_shape(fake):
    fake.discover_error = UnauthorizedError("Echo360 refused access (HTTP 401).",
                                            hint="Check you can play it in a browser.")
    result = invoke(["inspect", URL, "--json"])
    assert result.exit_code == 3
    doc = json.loads(result.stdout)
    assert doc["ok"] is False
    assert doc["error"]["kind"] == "unauthorized"
    assert doc["error"]["exit_code"] == 3
    assert "refused access" in result.stderr
    assert_clean(result.stdout + result.stderr)


@pytest.mark.parametrize("width", [40, 100])
def test_inspect_human_no_color_redirected(fake, width):
    result = invoke(["inspect", URL], env={"NO_COLOR": "1", "COLUMNS": str(width)})
    assert result.exit_code == 0, result.stderr
    assert result.stdout == ""  # human output goes to stderr only
    err = result.stderr
    assert_clean(err)
    for line in err.splitlines():
        assert len(line) <= width, line
    flat = " ".join(err.split())
    assert "Thermodynamics [draft]" in flat  # markup-like titles are printed literally
    assert "PHYS 201" in flat
    assert "54m 00s" in flat
    assert "Source 2" in err
    assert "forbidden" in err and "may exist" in flat
    assert "available" in err
    assert "shared" in err
    assert "platform-captions" in flat
    assert "3 (native)" in flat
    assert "(unverified)" in flat
    assert "?timestamp" not in err
    assert CLASSROOM.split("//")[1][:10] in err


def test_inspect_human_absent_captions_and_chapters(fake):
    fake.recording = make_recording(sources=1, captions=False, chapters=False)
    result = invoke(["inspect", URL])
    assert result.exit_code == 0
    assert "No platform transcript found" in result.stderr
    assert "No native chapters" in result.stderr
    assert "Source 2" not in result.stderr


def test_inspect_ascii_has_only_ascii(fake):
    result = invoke(["inspect", URL, "--ascii"])
    assert result.exit_code == 0
    assert result.stderr.isascii(), [c for c in result.stderr if not c.isascii()]
    assert "forbidden (403 - may exist)" in " ".join(result.stderr.split())


def test_inspect_verbose_debug_is_scrubbed(fake):
    result = invoke(["inspect", URL, "--verbose"])
    assert result.exit_code == 0
    assert "debug:" in result.stderr
    assert "[REDACTED" in result.stderr or "[…]" in result.stderr
    assert_clean(result.stderr)


def test_inspect_invalid_url_is_usage_error(fake):
    result = invoke(["inspect", "https://example.com/nothing", "--non-interactive"])
    assert result.exit_code == 2
    assert "Not an Echo360 classroom URL" in result.stderr


def test_request_file_options(fake, tmp_path):
    missing = invoke(["inspect", URL, "--auth", "request-file"])
    assert missing.exit_code == 2
    assert "--request-file" in missing.stderr
    req = tmp_path / "req.txt"
    req.write_text("curl 'https://content.echo360.ca/x'")
    ok = invoke(["inspect", URL, "--request-file", str(req), "--timeout", "15",
                 "--profile-dir", str(tmp_path / "prof")])
    assert ok.exit_code == 0, ok.stderr
    options = fake.discover_calls[-1][1]
    assert options.auth_mode == "request-file"
    assert options.request_file == req
    assert options.timeout == 15
    assert options.profile_dir == tmp_path / "prof"
