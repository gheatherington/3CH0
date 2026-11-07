"""``3ch0 doctor``: environment checks with OK/WARN/FAIL and a fix hint each.

FAIL means a required piece is missing or broken (exit code 6). WARN is advisory
(e.g. no browser profile yet, no optional transcription backend).
"""

from __future__ import annotations

import os
import platform
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from threech0.cli.output import UI
from threech0.cli.render import NARROW, data_table, header_line
from threech0.cli.theme import boxed, label_grid, margin, section_label, status_label
from threech0.redact import scrub
from threech0.services import Services

Status = Literal["OK", "WARN", "FAIL"]

MIN_PYTHON = (3, 11)

REQUIRED_TOOLS = {
    "ffmpeg": "Install FFmpeg: brew install ffmpeg",
    "ffprobe": "ffprobe ships with FFmpeg: brew install ffmpeg",
    "yt-dlp": "Reinstall the project environment (uv sync) or: brew install yt-dlp",
}


@dataclass
class Check:
    name: str
    status: Status
    detail: str
    hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _one_line(exc: BaseException) -> str:
    text = scrub(f"{type(exc).__name__}: {exc}").strip().splitlines()
    return text[0] if text else type(exc).__name__


def check_python() -> Check:
    version = platform.python_version()
    if tuple(sys.version_info[:2]) >= MIN_PYTHON:
        return Check("Python", "OK", f"{version} ({sys.executable})")
    return Check("Python", "FAIL", f"{version} is too old", "Use Python 3.11 or newer (uv python "
                 "install 3.13).")


def check_tools(services: Services) -> list[Check]:
    try:
        versions = services.tool_versions()
        error = None
    except Exception as exc:
        versions, error = {}, _one_line(exc)
    checks: list[Check] = []
    for tool, hint in REQUIRED_TOOLS.items():
        version = versions.get(tool)
        if version:
            checks.append(Check(tool, "OK", scrub(str(version))))
        else:
            detail = f"could not check ({error})" if error else "not found"
            checks.append(Check(tool, "FAIL", detail, hint))
    return checks


def check_browser(services: Services) -> Check:
    from threech0.echo360.browser import INSTALL_HINT

    hint = f"{INSTALL_HINT} (or use --auth request-file as a fallback)."
    try:
        ok, detail = services.browser_installed()
    except Exception as exc:
        return Check("Playwright Chromium", "FAIL", f"could not check ({_one_line(exc)})", hint)
    if ok:
        return Check("Playwright Chromium", "OK", scrub(detail) or "installed")
    return Check("Playwright Chromium", "FAIL", scrub(detail) or "not installed", hint)


def check_profile(profile_dir: Path) -> Check:
    if profile_dir.is_dir():
        return Check("Browser profile", "OK", f"{profile_dir}",
                     "Check the saved login with: 3ch0 auth status")
    return Check("Browser profile", "WARN", f"{profile_dir} (not created yet)",
                 "Sign in once with: 3ch0 auth login")


def check_output_dir(path: Path) -> Check:
    target = path if path.exists() else next((p for p in path.parents if p.exists()), None)
    if target is None or not target.is_dir():
        return Check("Output directory", "FAIL", f"{path} is not a directory",
                     "Choose another directory with --output DIR.")
    try:
        with tempfile.NamedTemporaryFile(dir=target, prefix=".3ch0-", delete=True):
            pass
    except OSError as exc:
        return Check("Output directory", "FAIL", f"{path} is not writable ({exc.strerror})",
                     "Choose another directory with --output DIR.")
    suffix = "" if path.exists() else " (will be created)"
    return Check("Output directory", "OK", f"{path} is writable{suffix}")


def check_transcription(services: Services) -> list[Check]:
    hint = ("Optional. From the project folder: uv tool install --force '.[transcribe]' "
            "(or '.[transcribe-mlx]' on Apple Silicon).")
    try:
        backends = list(services.transcription_backends())
    except Exception as exc:
        return [Check("Transcription", "WARN", f"could not check ({_one_line(exc)})", hint)]
    if not backends:
        return [Check("Transcription", "WARN", "no local backend available", hint)]
    checks: list[Check] = []
    any_available = False
    for backend in backends:
        name = str(getattr(backend, "name", "backend"))
        available = bool(getattr(backend, "available", False))
        detail = scrub(str(getattr(backend, "detail", "") or ""))
        accelerated = bool(getattr(backend, "accelerated", False))
        any_available = any_available or available
        if available:
            extra = " (GPU accelerated)" if accelerated else ""
            checks.append(Check(f"Transcription: {name}", "OK", (detail or "available") + extra))
        else:
            checks.append(Check(f"Transcription: {name}", "WARN", detail or "not installed"))
    if not any_available:
        checks[-1].hint = hint
    return checks


def run_checks(services: Services, *, output_dir: Path, profile_dir: Path) -> list[Check]:
    checks = [check_python()]
    checks.extend(check_tools(services))
    checks.append(check_browser(services))
    checks.append(check_profile(profile_dir))
    checks.append(check_output_dir(output_dir))
    checks.extend(check_transcription(services))
    if os.environ.get("THREECH0_HOME"):
        checks.append(Check("THREECH0_HOME", "OK", os.environ["THREECH0_HOME"]))
    return checks


def doctor_view(checks: list[Check], ui: UI) -> RenderableType:
    table: Table
    if ui.err.width < NARROW:
        table = Table.grid(padding=(0, 1))
        table.add_column(no_wrap=True)
        table.add_column(overflow="fold")
    else:
        table = data_table(ui, "Status", "Check", "Detail")
    for check in checks:
        detail = Text(check.detail)
        if check.hint and check.status != "OK":
            detail.append(f"\n{check.hint}", style="muted")
        if ui.err.width < NARROW:
            body = Text(check.name, style="heading")
            body.append("\n")
            body.append_text(detail)
            table.add_row(status_label(check.status), body)
        else:
            table.add_row(status_label(check.status), Text(check.name), detail)
    fails = sum(c.status == "FAIL" for c in checks)
    warns = sum(c.status == "WARN" for c in checks)
    summary = status_label("FAIL" if fails else ("WARN" if warns else "OK"))
    summary.append(f" {fails} failed{ui.glyphs.sep}{warns} warning(s){ui.glyphs.sep}"
                   f"{len(checks) - fails - warns} ok")
    result = label_grid()
    result.add_row(section_label("Result"), summary)
    parts: list[RenderableType] = []
    head = header_line(ui, "doctor")
    if head is not None:
        parts.extend([head, Text("")])
    parts.extend([boxed("Checks", table, ui.glyphs), result])
    return margin(Group(*parts))
