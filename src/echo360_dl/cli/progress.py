"""ProgressSink implementations for the CLI (all output on stderr).

Every sink records what it is told in a :class:`~echo360_dl.cli.telemetry.Telemetry`
(used by the live view and by the final summary) and renders it differently:

* :class:`LiveProgress` - stderr is a terminal: the compact 3CH0 live console
  (:mod:`echo360_dl.cli.dashboard`) drawn by :class:`~echo360_dl.cli.live.LiveRegion`
  (2 Hz, only changed lines, never blanked first) and cleared when done (the command
  prints a durable summary). With an interactive stdin, arrow keys select a row and
  ``d`` toggles details. ``--plain`` turns it off.
* :class:`LineProgress` - pipes/logs: sequential plain lines; no carriage returns, cursor
  movement or animation; updates throttled per task.
* :class:`QuietProgress` - ``--quiet``: prints nothing (errors and the final outputs list
  are printed by the command itself).

Every message is passed through :func:`echo360_dl.redact.scrub`. Unknown totals never
show a percentage.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

from rich.text import Text

from echo360_dl.cli import dashboard
from echo360_dl.cli.keys import DETAILS, DOWN, ENTER, UP, KeyReader
from echo360_dl.cli.live import LiveRegion
from echo360_dl.cli.output import UI, fmt_amount, fmt_clock, fmt_pair, fmt_rate, fmt_span
from echo360_dl.cli.telemetry import STAGES, TaskState, Telemetry
from echo360_dl.cli.theme import label_grid, margin, section_label, status_label, trimmed
from echo360_dl.models import Stage
from echo360_dl.redact import scrub


def stage_position(stage: Stage) -> str:
    return f"{STAGES.index(stage) + 1}/{len(STAGES)}"


def _done_detail(task: TaskState | None, telemetry: Telemetry) -> str:
    if task is None:
        return ""
    parts: list[str] = []
    if task.completed:
        parts.append(fmt_amount(task.completed, task.unit))
    if task.started is not None and task.finished is not None:
        parts.append(fmt_span(task.finished - task.started))
    return "  ".join(parts)


class _Base:
    """Shared bookkeeping: telemetry plus the optional extras the CLI and pipeline use
    (``set_context``, ``set_streams``, ``task_queued``, ``notice``)."""

    telemetry: Telemetry

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        pass

    @contextlib.contextmanager
    def paused(self) -> Iterator[None]:
        yield

    def set_context(self, **values: str | None) -> None:
        self.telemetry.set_context(**values)

    def set_streams(self, details: dict[str, tuple[str, str]]) -> None:
        self.telemetry.set_streams(details)

    def task_queued(self, task_id: str, description: str, unit: str = "bytes",
                    total: float | None = None) -> None:
        self.telemetry.task_queued(task_id, description, unit, total)


# --------------------------------------------------------------------------------------
# Quiet
# --------------------------------------------------------------------------------------


class QuietProgress(_Base):
    """Prints nothing; keeps the same lifecycle API (and telemetry) as the other sinks."""

    def __init__(self) -> None:
        self.telemetry = Telemetry()

    def stage(self, stage: Stage, message: str | None = None) -> None:
        self.telemetry.stage(stage)

    def task_start(self, task_id: str, description: str, total: float | None = None,
                   unit: str = "bytes") -> None:
        self.telemetry.task_start(task_id, description, total, unit)

    def task_update(self, task_id: str, **kwargs: Any) -> None:
        self.telemetry.task_update(task_id, **kwargs)

    def task_done(self, task_id: str, *, ok: bool = True, message: str | None = None) -> None:
        self.telemetry.task_done(task_id, ok=ok, message=message)

    def info(self, message: str) -> None:
        pass

    def warn(self, message: str) -> None:
        pass

    def debug(self, message: str) -> None:
        pass

    def notice(self, message: str) -> None:
        pass


# --------------------------------------------------------------------------------------
# Plain lines
# --------------------------------------------------------------------------------------


class LineProgress(_Base):
    """Line-based progress for non-terminal stderr (logs, CI, pipes)::

        STAGE     3/6 download
                  start  Source 1 video (720p)
                  ...    Source 1 video (720p)  13.5 / 27.0 MB  50%  2.0 MB/s  remaining 00:07
        [OK]      Source 1 video (720p)  27.0 MB  12.3 s
    """

    def __init__(self, ui: UI, *, interval: float = 10.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._ui = ui
        self._interval = interval
        self._clock = clock
        self.telemetry = Telemetry(clock=clock)
        self._last_print: dict[str, float] = {}
        self._lock = threading.RLock()

    def _line(self, label: Text | str, value: Text | str) -> None:
        grid = label_grid()
        grid.add_row(label if isinstance(label, Text) else section_label(label),
                     value if isinstance(value, Text) else Text(value))
        with self._lock:
            self._ui.err.print(trimmed(grid))

    def stage(self, stage: Stage, message: str | None = None) -> None:
        if self.telemetry.stage(stage):
            value = Text.assemble((stage_position(stage) + " ", "muted"), (stage.value, "stage"))
            if message:
                value.append(f"  {scrub(message)}", style="muted")
            self._line("STAGE", value)
        elif message:
            self.info(message)

    def task_start(self, task_id: str, description: str, total: float | None = None,
                   unit: str = "bytes") -> None:
        task = self.telemetry.task_start(task_id, description, total, unit)
        with self._lock:
            self._last_print[task_id] = self._clock()
        size = f"  {fmt_amount(task.total, unit)}" if task.total is not None else ""
        self._line("", Text(f"start  {task.description}{size}", style="muted"))

    def task_update(self, task_id: str, *, completed: float | None = None,
                    total: float | None = None, rate: float | None = None,
                    eta: float | None = None) -> None:
        task = self.telemetry.task_update(task_id, completed=completed, total=total, rate=rate,
                                          eta=eta)
        if task is None:
            return
        with self._lock:
            now = self._clock()
            if now - self._last_print.get(task_id, 0.0) < self._interval:
                return
            self._last_print[task_id] = now
        self._line("", Text(f"...    {task.description}  {self.describe(task)}", style="muted"))

    @staticmethod
    def describe(task: TaskState) -> str:
        text = fmt_pair(task.completed, task.total, task.unit)
        if task.total and task.completed is not None:
            text += f"  {min(100.0, 100.0 * task.completed / task.total):.0f}%"
        else:
            text += " so far"
        if task.rate is not None and task.unit == "bytes":
            text += f"  {fmt_rate(task.rate)}"
        if task.left is not None and task.total:
            text += f"  remaining {fmt_clock(task.left)}"
        elif task.unit == "seconds" and task.speed:
            text += f"  speed {task.speed:.1f}x"
        return text

    def task_done(self, task_id: str, *, ok: bool = True, message: str | None = None) -> None:
        task = self.telemetry.task_done(task_id, ok=ok, message=message)
        with self._lock:
            self._last_print.pop(task_id, None)
        line = Text(scrub(task.description if task else task_id))
        detail = _done_detail(task, self.telemetry)
        if detail:
            line.append(f"  {detail}", style="muted")
        if message:
            line.append(f"  {scrub(message)}", style="muted")
        self._line(section_label(status_label("OK" if ok else "FAIL").plain,
                                 "ok" if ok else "error"), line)

    def info(self, message: str) -> None:
        self.telemetry.event("info", message)
        self._line("", Text(scrub(message), style="muted"))

    def warn(self, message: str) -> None:
        self.telemetry.event("warn", message)
        self._line(section_label("[WARN]", "warn"), Text(scrub(message)))

    def debug(self, message: str) -> None:
        self.telemetry.debug(message)
        if self._ui.verbose:
            self._line("", Text("debug: " + scrub(message), style="muted"))

    def notice(self, message: str) -> None:
        self.telemetry.event("info", message)
        self._line("", Text(scrub(message), style="warn"))


# --------------------------------------------------------------------------------------
# Live console
# --------------------------------------------------------------------------------------


class LiveProgress(_Base):
    """The live 3CH0 console on a terminal. Started lazily on the first report."""

    def __init__(self, ui: UI, *, command: str | None = None,
                 pipeline: bool | None = None) -> None:
        self.ui = ui
        self.command = command or ui.command
        self.pipeline = (self.command == "download") if pipeline is None else pipeline
        self.telemetry = Telemetry()
        self.expanded = False
        self._lock = threading.RLock()
        self._region: LiveRegion | None = None
        self._paused = False
        self._chosen: str | None = None  # set once the user moves the selection
        self._warnings: list[str] = []  # shown in the live view; printed when it ends
        self._keys = KeyReader(ui.stdin, self._on_key) if ui.interactive else None

    # -- lifecycle ---------------------------------------------------------------------
    @property
    def keys_active(self) -> bool:
        return self._keys is not None and self._keys.active

    def _frame(self, width: int, height: int) -> list[Text]:
        snap = self.telemetry.snapshot()
        return dashboard.render(
            snap, self.ui.glyphs, width=width, height=height, command=self.command,
            selected=self.selected_id(snap), expanded=self.expanded, keys=self.keys_active,
            pipeline=self.pipeline)

    def _ensure_started(self) -> None:
        with self._lock:
            if self._region is not None or self._paused:
                return
            if self._keys is not None:
                self._keys.start()
            self._region = LiveRegion(self.ui.err, self._frame)
            self.ui.live_printer = self._region.print_above
            self._region.start()

    def _stop(self) -> bool:
        with self._lock:
            region, self._region = self._region, None
            self.ui.live_printer = None
        if self._keys is not None:
            self._keys.stop()
        if region is None:
            return False
        region.stop()
        return True

    def close(self) -> None:
        self._stop()
        self._flush_warnings()

    def take_warnings(self) -> list[str]:
        """Hand the not-yet-printed warnings to the caller's summary (and forget them)."""
        with self._lock:
            pending, self._warnings = self._warnings, []
        return pending

    def _flush_warnings(self) -> None:
        for message in self.take_warnings():
            grid = label_grid()
            grid.add_row(section_label("[WARN]", "warn"), Text(message))
            self.ui.err.print(margin(grid))

    @contextlib.contextmanager
    def paused(self) -> Iterator[None]:
        """Remove the live display and restore the terminal (e.g. while prompting)."""
        with self._lock:
            self._paused = True
        was_started = self._stop()
        self._flush_warnings()  # durable, above the prompt
        try:
            yield
        finally:
            with self._lock:
                self._paused = False
            if was_started:
                self._ensure_started()

    def _refresh(self) -> None:
        region = self._region
        if region is not None:
            with contextlib.suppress(Exception):
                region.refresh()

    def _print(self, renderable: Any) -> None:
        """Durable output above the live area."""
        self.ui._print(margin(renderable))

    # -- selection ---------------------------------------------------------------------
    def selected_id(self, snap: Any) -> str | None:
        tasks = dashboard.visible_tasks(snap)
        if not tasks:
            return None
        if self._chosen and any(t.id == self._chosen for t in tasks):
            return self._chosen
        active = next((t for t in tasks if t.state == "active"), None)
        return (active or tasks[-1]).id

    def _on_key(self, key: str) -> None:
        snap = self.telemetry.snapshot()
        tasks = dashboard.visible_tasks(snap)
        if key in (UP, DOWN) and tasks:
            current = self.selected_id(snap)
            ids = [t.id for t in tasks]
            index = ids.index(current) if current in ids else 0
            index = max(0, min(len(ids) - 1, index + (-1 if key == UP else 1)))
            self._chosen = ids[index]
        elif key in (ENTER, DETAILS):
            self.expanded = not self.expanded
        self._refresh()

    # -- ProgressSink ------------------------------------------------------------------
    def stage(self, stage: Stage, message: str | None = None) -> None:
        self.telemetry.stage(stage)
        if message:
            self.telemetry.event("info", message)
        self._ensure_started()

    def task_start(self, task_id: str, description: str, total: float | None = None,
                   unit: str = "bytes") -> None:
        task = self.telemetry.task_start(task_id, description, total, unit)
        self.telemetry.event("info", f"started {task.description}")
        self._ensure_started()

    def task_update(self, task_id: str, *, completed: float | None = None,
                    total: float | None = None, rate: float | None = None,
                    eta: float | None = None) -> None:
        self.telemetry.task_update(task_id, completed=completed, total=total, rate=rate,
                                   eta=eta)

    def task_done(self, task_id: str, *, ok: bool = True, message: str | None = None) -> None:
        task = self.telemetry.task_done(task_id, ok=ok, message=message)
        name = task.description if task else scrub(task_id)
        if ok:
            word = "reused" if task is not None and task.state == "reused" else "finished"
            self.telemetry.event("ok", f"{word} {name}", _done_detail(task, self.telemetry))
        else:
            level = "warn" if task is not None and task.state == "cancelled" else "fail"
            self.telemetry.event(level, f"{name}: {scrub(message or 'failed')}")
        self._ensure_started()

    def info(self, message: str) -> None:
        self.telemetry.event("info", message)
        self._ensure_started()

    def warn(self, message: str) -> None:
        message = scrub(message)
        self.telemetry.event("warn", message)
        with self._lock:
            if message not in self._warnings:
                self._warnings.append(message)
        if self._region is None and not self._paused:
            self._flush_warnings()
        else:
            self._ensure_started()

    def debug(self, message: str) -> None:
        self.telemetry.debug(message)
        if self.ui.verbose:
            self._print(Text("debug: " + scrub(message), style="muted"))

    def notice(self, message: str) -> None:
        """Something the user caused (e.g. Ctrl-C, called from a signal handler): shown in
        the header and the event list only - printing from a signal handler could
        interleave with a redraw. The final summary reports the outcome."""
        self.telemetry.event("warn", message)
        self.telemetry.set_context(status="stopping after the current step")
        self._refresh()


Sink = LiveProgress | LineProgress | QuietProgress


def make_progress(ui: UI, command: str | None = None) -> Sink:
    """Pick the sink: quiet → nothing; terminal → live console; otherwise plain lines."""
    if ui.quiet:
        return QuietProgress()
    if ui.live and not ui.json and not ui.options.plain:
        return LiveProgress(ui, command=command)
    return LineProgress(ui)
