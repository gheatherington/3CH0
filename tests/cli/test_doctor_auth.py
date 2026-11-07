from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from echo360_dl import paths
from echo360_dl.models import AuthStatus

from .conftest import invoke


@dataclass
class Backend:
    name: str
    available: bool
    detail: str
    accelerated: bool = False


def test_doctor_all_ok(fake, tmp_path):
    fake.backends = [Backend("faster-whisper", True, "1.1.0"),
                     Backend("mlx-whisper", True, "0.4", accelerated=True)]
    (tmp_path / "home" / "data" / "browser-profile").mkdir(parents=True)
    result = invoke(["doctor", "-o", str(tmp_path)])
    assert result.exit_code == 0, result.stderr
    err = result.stderr
    for name in ("Python", "ffmpeg", "ffprobe", "yt-dlp", "Playwright Chromium",
                 "Browser profile", "Output directory", "faster-whisper"):
        assert name in err
    assert "[FAIL]" not in err.split("failed")[0]
    assert "GPU accelerated" in err


def test_doctor_missing_ffmpeg_exit_6(fake, tmp_path):
    fake.versions = {"ffmpeg": None, "ffprobe": "9.0", "yt-dlp": "2026.10.26"}
    result = invoke(["doctor", "-o", str(tmp_path)])
    assert result.exit_code == 6
    assert "[FAIL]" in result.stderr
    assert "brew install ffmpeg" in result.stderr
    doc = json.loads(invoke(["doctor", "-o", str(tmp_path), "--json"]).stdout)
    assert doc["ok"] is False
    ffmpeg = next(c for c in doc["checks"] if c["name"] == "ffmpeg")
    assert ffmpeg["status"] == "FAIL" and ffmpeg["hint"]


def test_doctor_handles_exceptions(fake, tmp_path):
    fake.versions_error = ImportError("No module named 'echo360_dl.media.ffmpeg'")
    fake.backends = RuntimeError("backend probe exploded")
    fake.browser = (False, "chromium not installed")
    result = invoke(["doctor", "-o", str(tmp_path), "--json"])
    assert result.exit_code == 6
    doc = json.loads(result.stdout)
    by_name = {c["name"]: c for c in doc["checks"]}
    assert by_name["ffmpeg"]["status"] == "FAIL"
    assert "could not check" in by_name["ffmpeg"]["detail"]
    assert by_name["Transcription"]["status"] == "WARN"
    assert by_name["Playwright Chromium"]["status"] == "FAIL"
    assert by_name["Browser profile"]["status"] == "WARN"


def test_doctor_no_backends_is_warning_only(fake, tmp_path):
    fake.backends = [Backend("faster-whisper", False, "not installed")]
    result = invoke(["doctor", "-o", str(tmp_path), "--ascii"])
    assert result.exit_code == 0
    assert "[WARN]" in result.stderr
    assert "[transcribe]" in result.stderr  # hint text (wrapped inside the table cell)
    assert result.stderr.isascii()


def test_auth_login_requires_interactive(fake):
    result = invoke(["auth", "login", "--non-interactive"])
    assert result.exit_code == 2
    assert fake.login_calls == []


def test_auth_login_interactive(fake, tty, tmp_path):
    result = invoke(["auth", "login", "--profile-dir", str(tmp_path / "p"), "--timeout", "60"])
    assert result.exit_code == 0, result.stderr
    assert fake.login_calls == [tmp_path / "p"]
    assert "Logged in" in result.stderr


def test_auth_status(fake, tmp_path):
    result = invoke(["auth", "status", "--json"])
    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["status"]["logged_in"] is True
    fake.auth = AuthStatus(tmp_path, True, False, "session expired")
    out = invoke(["auth", "status"])
    assert out.exit_code == 3
    assert "Not logged in" in out.stderr and "3ch0 auth login" in out.stderr


def _profile(path: Path) -> Path:
    """A profile echo360-dl created (marker file) that Chromium has used."""
    paths.prepare_profile_dir(path)
    (path / "Local State").write_text("{}")
    return path


def test_auth_logout_refuses_real_chrome_profile(fake, tmp_path):
    chrome = tmp_path / "Chrome" / "Default"
    chrome.mkdir(parents=True)
    (chrome.parent / "Local State").write_text("{}")
    (chrome / "Preferences").write_text("{}")
    result = invoke(["auth", "logout", "--yes", "--profile-dir", str(chrome.parent)])
    assert result.exit_code == 2
    assert (chrome / "Preferences").exists()


def test_browser_refuses_foreign_profile_dir(tmp_path):
    from echo360_dl.echo360.browser import _profile as browser_profile
    from echo360_dl.errors import UsageError

    foreign = tmp_path / "Chrome"
    foreign.mkdir()
    (foreign / "Local State").write_text("{}")
    with pytest.raises(UsageError, match="did not create"):
        browser_profile(foreign)
    fresh = browser_profile(tmp_path / "new-profile")
    assert (fresh / paths.PROFILE_MARKER).is_file()
    assert browser_profile(fresh) == fresh  # reusable once marked


def test_auth_logout_yes(fake, tmp_path):
    profile = _profile(tmp_path / "prof")
    result = invoke(["auth", "logout", "--yes", "--profile-dir", str(profile)])
    assert result.exit_code == 0, result.stderr
    assert not profile.exists()


def test_auth_logout_default_profile_needs_confirmation(fake):
    profile = paths.browser_profile_dir()
    profile.mkdir(parents=True)
    result = invoke(["auth", "logout"])
    assert result.exit_code == 2
    assert "--yes" in result.stderr
    assert profile.exists()


def test_auth_logout_interactive_confirm(fake, tty, tmp_path):
    profile = _profile(tmp_path / "prof")
    kept = invoke(["auth", "logout", "--profile-dir", str(profile)], input="n\n")
    assert kept.exit_code == 0 and profile.exists()
    removed = invoke(["auth", "logout", "--profile-dir", str(profile)], input="y\n")
    assert removed.exit_code == 0 and not profile.exists()


@pytest.mark.parametrize("make", ["plain", "home"])
def test_auth_logout_refuses_non_profile(fake, tmp_path, monkeypatch, make):
    target = tmp_path / "not-a-profile"
    target.mkdir()
    (target / "important.txt").write_text("keep me")
    if make == "home":
        monkeypatch.setenv("HOME", str(target))
    result = invoke(["auth", "logout", "--yes", "--profile-dir", str(target)])
    assert result.exit_code == 2
    assert (target / "important.txt").exists()


def test_auth_logout_missing_profile(fake, tmp_path):
    result = invoke(["auth", "logout", "--profile-dir", str(tmp_path / "none"), "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["removed"] is False
