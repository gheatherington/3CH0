"""Visual language: a restrained, technical console look.

Teal/cyan accents, amber warnings, red errors and muted secondary text. Every status
is spelled out as a word (``[OK]``, ``[WARN]``, ``[FAIL]``) so colour is never the only
signal. Styles are mid-tone colours that read on both light and dark backgrounds
(no pure white or black text).

ASCII mode (``--ascii`` or a non-UTF-8 stream encoding) swaps every non-ASCII glyph
and box character for a plain equivalent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import IO

from rich import box
from rich.box import Box
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

STYLES = {
    "accent": "dark_cyan",
    "heading": "bold dark_cyan",
    "label": "grey50",
    "muted": "grey50",
    "ok": "green4",
    "warn": "dark_orange3",
    "error": "bold red3",
    "path": "dark_cyan",
    "stage": "bold dark_cyan",
}

THEME = Theme(STYLES, inherit=False)

#: MINIMAL drawn with ASCII characters only.
ASCII_MINIMAL = Box(
    "    \n"
    "  | \n"
    "--+-\n"
    "  | \n"
    "--+-\n"
    "--+-\n"
    "  | \n"
    "    \n",
    ascii=True,
)


@dataclass(frozen=True)
class Glyphs:
    check: str
    cross: str
    bar: str
    dash: str
    dot: str
    arrow: str
    bullet: str
    ellipsis: str
    table_box: Box
    panel_box: Box

    @property
    def sep(self) -> str:
        """Separator between inline items (``360p · 720p``)."""
        return f" {self.dot} "


UNICODE = Glyphs(check="✓", cross="✗", bar="▌", dash="—", dot="·", arrow="→", bullet="•",
                 ellipsis="…", table_box=box.MINIMAL, panel_box=box.SQUARE)
ASCII = Glyphs(check="OK", cross="x", bar="|", dash="-", dot="-", arrow="->", bullet="*",
               ellipsis="...", table_box=ASCII_MINIMAL, panel_box=box.ASCII)


def encoding_is_utf8(stream: IO[str] | None) -> bool:
    encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "").replace("_", "")
    return encoding.startswith("utf8")


def glyphs_for(*, ascii_only: bool) -> Glyphs:
    return ASCII if ascii_only else UNICODE


def color_disabled(no_color_flag: bool) -> bool:
    """``--no-color`` or a non-empty ``NO_COLOR`` environment variable (no-color.org)."""
    return no_color_flag or bool(os.environ.get("NO_COLOR"))


def make_console(stream: IO[str], *, no_color: bool, ascii_only: bool,
                 width: int | None = None) -> Console:
    """A stderr-style console. Colour is dropped entirely when disabled; Rich already
    emits no escape codes when ``stream`` is not a terminal."""
    kwargs: dict[str, object] = {}
    if no_color:
        kwargs["color_system"] = None
    return Console(
        file=stream,
        theme=THEME,
        no_color=no_color,
        highlight=False,
        emoji=False,
        safe_box=True,
        width=width,
        legacy_windows=False,
        **kwargs,  # type: ignore[arg-type]
    )


def status_label(status: str) -> Text:
    """``[OK]`` / ``[WARN]`` / ``[FAIL]`` / ``[PARTIAL]`` ... as styled text."""
    key = status.upper()
    style = {"OK": "ok", "WARN": "warn", "FAIL": "error", "PARTIAL": "warn",
             "FAILED": "error", "CANCELLED": "warn", "SKIP": "muted",
             "INFO": "accent"}.get(key, "accent")
    return Text(f"[{key}]", style=style)


def status_block(status: str, message: str | Text, hint: str | None = None) -> Table:
    """``[FAIL] message`` with wrapped lines (and an optional hint) hanging-indented."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(no_wrap=True)
    grid.add_column(overflow="fold")
    grid.add_row(status_label(status), message if isinstance(message, Text) else Text(message))
    if hint:
        grid.add_row(Text(""), Text(f"hint: {hint}", style="muted"))
    return grid


def wordmark(glyphs: Glyphs, subtitle: str | None = None) -> Text:
    """One-line header: ``▌echo360-dl  inspect``."""
    text = Text()
    text.append(glyphs.bar, style="accent")
    text.append("echo360-dl", style="heading")
    if subtitle:
        text.append(f"  {subtitle}", style="muted")
    return text
