"""Prompts (stderr only). Callers check :attr:`UI.interactive` before using them;
nothing here is ever reached with ``--json`` or ``--non-interactive``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.text import Text

from echo360_dl.cli.output import UI
from echo360_dl.cli.render import sources_table
from echo360_dl.errors import CancelledError, SelectionRequiredError, UnavailableError, UsageError
from echo360_dl.models import AccessState, Layout, Recording
from echo360_dl.redact import scrub
from echo360_dl.selection import parse_streams, resolve_streams

MAX_ATTEMPTS = 5

STREAMS_HINT = ("Add --streams all (every source), --streams 1 or --streams 1,2 "
                "(specific sources), or --streams audio (audio only).")


def available_sources(rec: Recording) -> tuple[int, ...]:
    """Sources a user can reasonably expect to download (default selection)."""
    usable = tuple(s.index for s in sorted(rec.sources, key=lambda s: s.index)
                   if s.access in (AccessState.AVAILABLE, AccessState.UNKNOWN))
    return usable or tuple(sorted(s.index for s in rec.sources))


def _ask(ui: UI, prompt: Text, **kwargs: Any) -> Any:
    try:
        return Prompt.ask(prompt, console=ui.err, **kwargs)
    except EOFError:
        raise CancelledError("No input received; nothing was downloaded.",
                             hint=STREAMS_HINT) from None


def _confirm(ui: UI, prompt: Text, *, default: bool) -> bool:
    try:
        return bool(Confirm.ask(prompt, console=ui.err, default=default))
    except EOFError:
        raise CancelledError("No input received.") from None


def choose_sources(ui: UI, rec: Recording) -> tuple[tuple[int, ...], bool]:
    """Show the source table and ask once. Returns ``(source_indexes, audio_only)``.

    ``all`` (the default) means every *available* source; numbers pick specific
    sources (including ones whose access could not be confirmed).
    """
    default = available_sources(rec)
    ui.err.print(sources_table(rec, ui))
    numbers = ",".join(map(str, default))
    help_text = Text(f"all = sources {numbers}{ui.glyphs.sep}audio = audio only"
                     f"{ui.glyphs.sep}or numbers such as 1,2", style="muted")
    ui.err.print(help_text)
    for _ in range(MAX_ATTEMPTS):
        answer = str(_ask(ui, Text("Download which sources?", style="heading"),
                          default="all", show_default=True)).strip()
        try:
            selection = parse_streams(answer or "all")
            if selection.audio_only:
                return (), True
            if selection.all_sources:
                return default, False
            return resolve_streams(selection, rec), False
        except (UsageError, UnavailableError) as err:
            ui.err.print(Text(scrub(err.message) + (f" {err.hint}" if err.hint else ""),
                              style="warn"))
    raise SelectionRequiredError("No valid source selection was made.", hint=STREAMS_HINT)


def choose_layout(ui: UI, rec: Recording, indexes: tuple[int, ...]) -> Layout:
    """Asked once when exactly two sources were picked interactively."""
    names = [rec.source(i).display_name if rec.source(i) else f"Source {i}" for i in indexes]
    ui.err.print(Text(f"separate = one file per source (original quality){ui.glyphs.sep}"
                      f"side-by-side = {names[0]} left, {names[1]} right{ui.glyphs.sep}"
                      f"top-bottom = {names[0]} on top (both re-encoded into one video)",
                      style="muted"))
    choices = [layout.value for layout in Layout]
    for _ in range(MAX_ATTEMPTS):
        answer = str(_ask(ui, Text("Layout?", style="heading"), default="separate",
                          show_default=True)).strip().lower()
        aliases = {"s": "separate", "sbs": "side-by-side", "lr": "side-by-side",
                   "tb": "top-bottom", "stacked": "top-bottom"}
        answer = aliases.get(answer, answer or "separate")
        if answer in choices:
            return Layout(answer)
        ui.err.print(Text(f"Choose one of: {', '.join(choices)}.", style="warn"))
    raise SelectionRequiredError("No valid layout was chosen.",
                                 hint="Pass --layout separate|side-by-side|top-bottom.")


def confirm_plan(ui: UI) -> bool:
    return _confirm(ui, Text("Start download?", style="heading"), default=True)


def confirm_logout(ui: UI, profile: Path) -> bool:
    ui.err.print(Text(f"This deletes the dedicated browser profile at {profile} "
                      "(saved Echo360 login).", style="warn"))
    return _confirm(ui, Text("Delete it?", style="heading"), default=False)


def model_download_confirmer(ui: UI, *, allow: bool,
                             pause: Callable[[], Any] | None = None) -> Callable[[str], bool]:
    """``confirm_download`` callback passed to services.

    Non-interactive: returns ``--allow-model-download``. Interactive: the flag
    approves directly; otherwise ask (default no), pausing live progress first.
    """
    if allow or not ui.interactive:
        return lambda _message: bool(allow)

    def confirm(message: str) -> bool:
        context = pause() if pause is not None else None
        if context is not None:
            context.__enter__()
        try:
            ui.err.print(Panel(Text(scrub(message)), title=Text("Speech model download",
                                                                  style="heading"),
                               title_align="left", box=ui.glyphs.panel_box,
                               border_style="warn", expand=False))
            return _confirm(ui, Text("Download it now?", style="heading"), default=False)
        finally:
            if context is not None:
                context.__exit__(None, None, None)

    return confirm
