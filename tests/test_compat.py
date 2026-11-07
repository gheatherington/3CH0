"""The legacy script translates its old flags into echo360-dl commands."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("legacy", ROOT / "download_echo360.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
URL = "https://echo360.ca/lesson/L1/classroom"


def test_default_downloads_all_sources():
    assert legacy.translate([URL]) == ["download", URL, "--streams", "all"]


def test_flags_are_mapped():
    args = legacy.translate([URL, "--request-file", "r.txt", "--output-dir", "out",
                             "--browser-profile", "prof", "--timeout", "60"])
    assert args == ["download", URL, "--streams", "all", "--auth", "request-file",
                    "--request-file", "r.txt", "--profile-dir", "prof", "--timeout", "60",
                    "--output", "out"]


def test_list_formats_is_inspect_and_fresh_ignores_request_file():
    assert legacy.translate([URL, "--list-formats"]) == ["inspect", URL]
    assert "--request-file" not in legacy.translate([URL, "--fresh", "--request-file", "r"])


def test_script_runs_and_rejects_bad_url():
    proc = subprocess.run([sys.executable, str(ROOT / "download_echo360.py"),
                           "https://example.com/x", "--list-formats"],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2 and "echo360" in proc.stderr.lower()
