"""A flicker-free live region at the bottom of the terminal.

Rich's ``Live`` erases every line of the region and draws it again on each refresh. Some
terminals (especially through tmux) show that blank moment as a flash several times a
second. This region never blanks anything it redraws:

* only lines whose content changed are written, in place, each followed by
  "erase to end of line" (``ESC[K``) for the leftover tail;
* every update is wrapped in synchronized output (``ESC[?2026h`` … ``ESC[?2026l``), which
  supporting terminals (Ghostty, iTerm2, kitty, WezTerm, recent tmux) apply in one step;
  others ignore it;
* it repaints at most :data:`REFRESH_PER_SECOND` times a second, and only if something changed;
* lines stop one column short of the edge (writing the last column would leave a pending
  wrap that ``ESC[K`` erases on some terminals).

Durable output (warnings, verbose lines) goes through :meth:`LiveRegion.print_above`,
which clears the region, prints, and draws the region again below it.
"""

from __future__ import annotations

import contextlib
import io
import threading
from collections.abc import Callable

from rich.console import Console
from rich.text import Text

from threech0.cli.theme import THEME

REFRESH_PER_SECOND = 2.0

SYNC_ON, SYNC_OFF = "\x1b[?2026h", "\x1b[?2026l"
HIDE_CURSOR, SHOW_CURSOR = "\x1b[?25l", "\x1b[?25h"
ERASE_EOL, ERASE_DOWN = "\x1b[K", "\x1b[J"

Frame = Callable[[int, int], list[Text]]


class LiveRegion:
    def __init__(self, console: Console, frame: Frame, *,
                 refresh_per_second: float = REFRESH_PER_SECOND) -> None:
        self._console = console
        self._frame = frame
        self._interval = 1.0 / refresh_per_second
        self._lock = threading.RLock()
        self._prev: list[str] = []
        self._size: tuple[int, int] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        # Off-screen console that turns styled Text into escape sequences with the same
        # colour settings (no colour at all under NO_COLOR / --no-color).
        self._painter = Console(file=io.StringIO(), force_terminal=True, theme=THEME,
                                color_system=console.color_system,  # type: ignore[arg-type]
                                no_color=console.no_color, width=10_000, highlight=False,
                                emoji=False, legacy_windows=False)

    @property
    def running(self) -> bool:
        return self._running

    # -- lifecycle ---------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._stop.clear()
            self._write(HIDE_CURSOR)
            self._paint()
        self._thread = threading.Thread(target=self._loop, name="3ch0-live", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Remove the region and show the cursor again (safe to call repeatedly)."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._write(self._clear_sequence() + SHOW_CURSOR)
            self._prev = []

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            with contextlib.suppress(Exception):
                self.refresh()

    # -- drawing -----------------------------------------------------------------------
    def refresh(self) -> None:
        with self._lock:
            if self._running:
                self._paint()

    def print_above(self, emit: Callable[[], None]) -> None:
        """Run ``emit`` (which prints durable output) with the region out of the way."""
        with self._lock:
            if not self._running:
                emit()
                return
            self._write(SYNC_ON + self._clear_sequence())
            self._prev = []
            try:
                emit()
            finally:
                self._paint(sync_open=True)

    def _write(self, data: str) -> None:
        stream = self._console.file
        with contextlib.suppress(OSError, ValueError):
            stream.write(data)
            stream.flush()

    def _clear_sequence(self) -> str:
        """Cursor to the region's first line, then erase everything below."""
        if not self._prev:
            return "\r" + ERASE_DOWN
        up = f"\x1b[{len(self._prev) - 1}A" if len(self._prev) > 1 else ""
        return "\r" + up + ERASE_DOWN

    def _ansi(self, line: Text) -> str:
        with self._painter.capture() as capture:
            self._painter.print(line, end="", no_wrap=True, overflow="crop", soft_wrap=True)
        return capture.get()

    def _paint(self, sync_open: bool = False) -> None:
        width, height = self._console.size
        lines = self._frame(max(10, width - 1), max(3, height - 1))
        rendered = [self._ansi(line) for line in lines]
        resized = self._size is not None and self._size != (width, height)
        if rendered == self._prev and not resized and not sync_open:
            return  # nothing changed: write nothing at all
        out = [] if sync_open else [SYNC_ON]
        prev = self._prev
        if resized and prev:
            out.append(self._clear_sequence())
            prev = []
        elif prev:
            out.append("\r")
            if len(prev) > 1:
                out.append(f"\x1b[{len(prev) - 1}A")
        if not rendered:
            out.append(ERASE_DOWN)
        for i, line in enumerate(rendered):
            if i:
                out.append("\n")
            if i >= len(prev) or prev[i] != line:
                out.append("\r" + line + ERASE_EOL)
        if len(rendered) < len(prev):
            # Erase the old frame's extra lines, then return to the new last line.
            out.append("\x1b[1B\r" + ERASE_DOWN + "\x1b[1A")
        out.append(SYNC_OFF)
        self._write("".join(out))
        self._prev = rendered
        self._size = (width, height)
