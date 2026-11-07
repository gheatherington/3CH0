"""Prompts (stderr only). Callers check :attr:`UI.interactive` before using them;
nothing here is ever reached with ``--json`` or ``--non-interactive``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Group
from rich.prompt import Confirm, Prompt
from rich.text import Text

from threech0.cli.output import UI
from threech0.cli.render import header_line, sources_table
from threech0.cli.theme import MARGIN, boxed, label_grid, margin, section_label
from threech0.errors import CancelledError, SelectionRequiredError, UnavailableError, UsageError
from threech0.models import AccessState, Layout, Recording
from threech0.redact import scrub
from threech0.selection import parse_streams, resolve_streams

MAX_ATTEMPTS = 5

STREAMS_HINT = ("Add --streams all (every source), --streams 1 or --streams 1,2 "
                "(specific sources), or --streams audio (audio only).")


def available_sources(rec: Recording) -> tuple[int, ...]:
    """Sources a user can reasonably expect to download (default selection)."""
    usable = tuple(s.index for s in sorted(rec.sources, key=lambda s: s.index)
                   if s.access in (AccessState.AVAILABLE, AccessState.UNKNOWN))
    return usable or tuple(sorted(s.index for s in rec.sources))


def _prompt(text: str) -> Text:
    """Prompts sit on the view's left margin, value-styled with the accent marker."""
    return Text.assemble(" " * MARGIN, (text, "heading"))


def _note(ui: UI, text: str, style: str = "muted") -> None:
    ui.err.print(margin(Text(text, style=style)))


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
    head = header_line(ui, "download")
    title = label_grid()
    title.add_row(section_label("Lesson"), Text(rec.title or rec.lesson_id, style="heading"))
    ui.err.print(margin(Group(*([head] if head is not None else []),
                              boxed("Sources", Group(title, Text(""), sources_table(rec, ui)),
                                    ui.glyphs))))
    numbers = ",".join(map(str, default))
    _note(ui, f"all = sources {numbers}{ui.glyphs.sep}audio = audio only"
              f"{ui.glyphs.sep}or numbers such as 1,2")
    for _ in range(MAX_ATTEMPTS):
        answer = str(_ask(ui, _prompt("Download which sources?"),
                          default="all", show_default=True)).strip()
        try:
            selection = parse_streams(answer or "all")
            if selection.audio_only:
                return (), True
            if selection.all_sources:
                return default, False
            return resolve_streams(selection, rec), False
        except (UsageError, UnavailableError) as err:
            _note(ui, scrub(err.message) + (f" {err.hint}" if err.hint else ""), "warn")
    raise SelectionRequiredError("No valid source selection was made.", hint=STREAMS_HINT)


def choose_layout(ui: UI, rec: Recording, indexes: tuple[int, ...]) -> Layout:
    """Asked once when exactly two sources were picked interactively."""
    names = [rec.source(i).display_name if rec.source(i) else f"Source {i}" for i in indexes]
    _note(ui, f"separate = one file per source (original quality){ui.glyphs.sep}"
              f"side-by-side = {names[0]} left, {names[1]} right{ui.glyphs.sep}"
              f"top-bottom = {names[0]} on top (both re-encoded into one video)")
    choices = [layout.value for layout in Layout]
    for _ in range(MAX_ATTEMPTS):
        answer = str(_ask(ui, _prompt("Layout?"), default="separate",
                          show_default=True)).strip().lower()
        aliases = {"s": "separate", "sbs": "side-by-side", "lr": "side-by-side",
                   "tb": "top-bottom", "stacked": "top-bottom"}
        answer = aliases.get(answer, answer or "separate")
        if answer in choices:
            return Layout(answer)
        _note(ui, f"Choose one of: {', '.join(choices)}.", "warn")
    raise SelectionRequiredError("No valid layout was chosen.",
                                 hint="Pass --layout separate|side-by-side|top-bottom.")


def confirm_plan(ui: UI) -> bool:
    ui.err.print()
    return _confirm(ui, _prompt("Start download?"), default=True)


def confirm_logout(ui: UI, profile: Path) -> bool:
    _note(ui, f"This deletes the dedicated browser profile at {profile} "
              "(saved Echo360 login).", "warn")
    return _confirm(ui, _prompt("Delete it?"), default=False)


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
            ui.err.print(margin(boxed("Speech model", Text(scrub(message)), ui.glyphs,
                                      style="warn")))
            return _confirm(ui, _prompt("Download the speech model now?"), default=False)
        finally:
            if context is not None:
                context.__exit__(None, None, None)

    return confirm
