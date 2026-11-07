"""3CH0 visual system: a minimalist telemetry console.

One place for colour, spacing, labels and glyphs so command output, the live display
and prompts stay consistent.

* Terminal-default foreground/background; one cyan accent for active work, selection
  and important highlights; muted secondary text; amber warnings and red failures.
* Every status is also spelled out (``[OK]``, ``[WARN]``, ``[FAIL]``), so colour is never
  the only signal. Colours are mid-tones that read on light and dark backgrounds.
* Small uppercase section labels in a fixed-width column, normal-case values.
* Each section sits in a thin, muted box titled with its name, so it is clear what is what.
* ASCII mode (``--ascii`` or a non-UTF-8 stream encoding) swaps every non-ASCII glyph
  for a plain equivalent; ``--no-color``/``NO_COLOR`` drop colour entirely.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import IO

from rich import box
from rich.box import Box
from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.padding import Padding
from rich.panel import Panel
from rich.segment import Segment
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

#: Display name and command name.
BRAND = "3CH0"
COMMAND = "3ch0"

#: Left margin of every multi-section view, and the width of the label column.
MARGIN = 2
LABEL_WIDTH = 10

_ACCENT = "color(37)"  # mid cyan (#00afaf); maps to ANSI cyan on 16-colour terminals
_MUTED = "grey50"
_AMBER = "dark_orange3"
_RED = "red3"

STYLES = {
    "accent": _ACCENT,
    "heading": "bold",
    "label": _MUTED,
    "muted": _MUTED,
    "value": "none",
    "ok": _ACCENT,
    "warn": _AMBER,
    "error": f"bold {_RED}",
    "path": "none",
    "stage": f"bold {_ACCENT}",
    "selected": f"bold {_ACCENT}",
    "bar.done": _ACCENT,
    "bar.rest": _MUTED,
    "spark": _ACCENT,
    "border": _MUTED,
    "title": "bold",
}

THEME = Theme(STYLES, inherit=False)


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
    times: str  # 1280×720
    marker: str  # selected row
    fill: str  # progress bar, done part
    rest: str  # progress bar, remaining part
    rule: str  # thin separator
    underline: str  # current pipeline stage
    spark: str  # sparkline levels, lowest first
    corners: str  # box corners: top-left, top-right, bottom-left, bottom-right
    h: str  # box horizontal line
    v: str  # box vertical line
    panel_box: Box

    @property
    def sep(self) -> str:
        """Separator between inline items (``360p · 720p``)."""
        return f" {self.dot} "


UNICODE = Glyphs(check="✓", cross="✗", bar="▌", dash="—", dot="·", arrow="→", bullet="•",
                 ellipsis="…", times="×", marker="▸", fill="━", rest="─", rule="─",
                 underline="━", spark="▁▂▃▄▅▆▇█", corners="╭╮╰╯", h="─", v="│",
                 panel_box=box.ROUNDED)
ASCII = Glyphs(check="OK", cross="x", bar="|", dash="-", dot="-", arrow="->", bullet="*",
               ellipsis="...", times="x", marker=">", fill="#", rest="-", rule="-",
               underline="^", spark="_.:-=+*#", corners="++++", h="-", v="|",
               panel_box=box.ASCII)


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


# --------------------------------------------------------------------------------------
# Labels and layout primitives
# --------------------------------------------------------------------------------------

_STATUS_STYLES = {"OK": "ok", "WARN": "warn", "FAIL": "error", "PARTIAL": "warn",
                  "FAILED": "error", "CANCELLED": "warn", "SKIP": "muted", "INFO": "accent"}


def status_label(status: str) -> Text:
    """``[OK]`` / ``[WARN]`` / ``[FAIL]`` / ``[PARTIAL]`` ... as styled text."""
    key = status.upper()
    return Text(f"[{key}]", style=_STATUS_STYLES.get(key, "accent"))


def section_label(name: str, style: str = "label") -> Text:
    """A small uppercase label padded to the label column (``TRANSFER  ``)."""
    return Text(name.upper().ljust(LABEL_WIDTH - 1)[:LABEL_WIDTH - 1] + " ", style=style)


def label_grid() -> Table:
    """Two columns: fixed-width uppercase label, folding value."""
    grid = Table.grid(padding=0)
    grid.add_column(width=LABEL_WIDTH, no_wrap=True)
    grid.add_column(overflow="fold")
    return grid


class Trimmed:
    """Renders ``renderable`` without the trailing spaces Rich pads table cells with
    (they clutter logs and copied text)."""

    def __init__(self, renderable: RenderableType) -> None:
        self.renderable = renderable

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        for line in console.render_lines(self.renderable, options, pad=False):
            while line and not line[-1].control and not line[-1].text.strip():
                line.pop()
            if line and not line[-1].control:
                last = line[-1]
                line[-1] = Segment(last.text.rstrip(), last.style)
            yield from line
            yield Segment.line()


def boxed(title: str, body: RenderableType, glyphs: Glyphs, style: str = "border") -> Panel:
    """A section in a thin box titled with its (uppercase) name."""
    return Panel(body, title=Text(title.upper(), style="title"), title_align="left",
                 box=glyphs.panel_box, border_style=style, padding=(0, 1), expand=True)


def margin(renderable: RenderableType) -> Trimmed:
    """The standard left margin for multi-section views."""
    return Trimmed(Padding(renderable, (0, 0, 0, MARGIN)))


def trimmed(renderable: RenderableType) -> Trimmed:
    return Trimmed(renderable)


def status_block(status: str, message: str | Text, hint: str | None = None) -> Table:
    """``[FAIL] message`` with wrapped lines (and an optional hint) hanging-indented."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(no_wrap=True)
    grid.add_column(overflow="fold")
    grid.add_row(status_label(status), message if isinstance(message, Text) else Text(message))
    if hint:
        grid.add_row(Text(""), Text(f"next: {hint}", style="muted"))
    return grid


def wordmark(glyphs: Glyphs, subtitle: str | None = None) -> Text:
    """One-line header: ``3CH0 / download``."""
    text = Text()
    text.append(BRAND, style="heading")
    if subtitle:
        text.append(" / ", style="muted")
        text.append(subtitle, style="muted")
    return text
