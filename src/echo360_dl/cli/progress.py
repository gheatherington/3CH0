"""ProgressSink implementations for the CLI (all output on stderr).

* :class:`RichProgress` - live Rich display when stderr is a terminal. Bytes, rate and
  ETA appear only when known; an unknown total shows a spinner and the amount so far,
  never a percentage.
* :class:`LineProgress` - plain lines for pipes/logs: no carriage returns, no
  animation, updates throttled per task.
* :class:`QuietProgress` - ``--quiet``: discards progress (errors and the final
  outputs list are printed by the command itself).

Every message is passed through :func:`echo360_dl.redact.scrub`.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
)
from rich.table import Column
from rich.text import Text

from echo360_dl.cli.output import UI, fmt_amount, fmt_bytes, fmt_eta
from echo360_dl.cli.theme import status_label
from echo360_dl.models import Stage
from echo360_dl.redact import scrub

STAGES = list(Stage)


def _stage_prefix(stage: Stage) -> str:
    return f"[{STAGES.index(stage) + 1}/{len(STAGES)}] {stage.value}"


def _known(value: float | None) -> bool:
    return value is not None and value == value and value >= 0  # excludes NaN/negative


class _StageMixin:
    _ui: UI
    _stage: Stage | None = None

    def _stage_text(self, stage: Stage, message: str | None) -> Text | None:
        """Header text for a new stage; ``None`` when the stage repeats (message only)."""
        if stage == self._stage:
            return None
        self._stage = stage
        text = Text()
        text.append(self._ui.glyphs.bar + " ", style="accent")
        text.append(_stage_prefix(stage), style="stage")
        if message:
            text.append(f"  {scrub(message)}", style="muted")
        return text


# --------------------------------------------------------------------------------------
# Quiet
# --------------------------------------------------------------------------------------


class QuietProgress:
    """Discards everything; keeps the same lifecycle API as the other sinks."""

    def __enter__(self) -> QuietProgress:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        pass

    @contextlib.contextmanager
    def paused(self) -> Iterator[None]:
        yield

    def stage(self, stage: Stage, message: str | None = None) -> None:
        pass

    def task_start(self, task_id: str, description: str, total: float | None = None,
                   unit: str = "bytes") -> None:
        pass

    def task_update(self, task_id: str, **kwargs: Any) -> None:
        pass

    def task_done(self, task_id: str, *, ok: bool = True, message: str | None = None) -> None:
        pass

    def info(self, message: str) -> None:
        pass

    def warn(self, message: str) -> None:
        pass

    def debug(self, message: str) -> None:
        pass


# --------------------------------------------------------------------------------------
# Plain lines
# --------------------------------------------------------------------------------------


@dataclass
class _LineTask:
    description: str
    unit: str
    total: float | None = None
    completed: float | None = None
    rate: float | None = None
    eta: float | None = None
    last_print: float = 0.0


class LineProgress(_StageMixin):
    """Line-based progress for non-terminal stderr (logs, CI, pipes)."""

    def __init__(self, ui: UI, *, interval: float = 10.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._ui = ui
        self._interval = interval
        self._clock = clock
        self._tasks: dict[str, _LineTask] = {}
        self._lock = threading.RLock()

    def __enter__(self) -> LineProgress:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        pass

    @contextlib.contextmanager
    def paused(self) -> Iterator[None]:
        yield

    def _print(self, text: Text | str) -> None:
        with self._lock:
            self._ui.err.print(text, overflow="fold")

    def stage(self, stage: Stage, message: str | None = None) -> None:
        header = self._stage_text(stage, message)
        if header is not None:
            self._print(header)
        elif message:
            self.info(message)

    def task_start(self, task_id: str, description: str, total: float | None = None,
                   unit: str = "bytes") -> None:
        with self._lock:
            task = _LineTask(scrub(description), unit, total if _known(total) else None)
            task.last_print = self._clock()
            self._tasks[task_id] = task
        size = f" ({fmt_amount(task.total, unit)})" if task.total is not None else ""
        self._print(Text(f"  start  {task.description}{size}", style="muted"))

    def task_update(self, task_id: str, *, completed: float | None = None,
                    total: float | None = None, rate: float | None = None,
                    eta: float | None = None) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            if completed is not None:
                task.completed = completed
            if _known(total):
                task.total = total
            if _known(rate):
                task.rate = rate
            if _known(eta):
                task.eta = eta
            now = self._clock()
            if now - task.last_print < self._interval:
                return
            task.last_print = now
            line = f"  ...    {task.description}: {self._describe(task)}"
        self._print(Text(line, style="muted"))

    @staticmethod
    def _describe(task: _LineTask) -> str:
        done = fmt_amount(task.completed or 0, task.unit)
        if task.total:
            pct = min(100.0, 100.0 * (task.completed or 0) / task.total)
            text = f"{done} / {fmt_amount(task.total, task.unit)} ({pct:.0f}%)"
        else:
            text = f"{done} so far"
        if task.rate is not None and task.unit == "bytes":
            text += f" at {fmt_bytes(task.rate)}/s"
        if task.eta is not None:
            text += f", ETA {fmt_eta(task.eta)}"
        return text

    def task_done(self, task_id: str, *, ok: bool = True, message: str | None = None) -> None:
        with self._lock:
            task = self._tasks.pop(task_id, None)
        description = task.description if task else task_id
        line = status_label("OK" if ok else "FAIL")
        line.append(f" {scrub(description)}")
        if task and task.completed:
            line.append(f"  {fmt_amount(task.completed, task.unit)}", style="muted")
        if message:
            line.append(f"  {scrub(message)}", style="muted")
        self._print(line)

    def info(self, message: str) -> None:
        self._print(Text("  " + scrub(message), style="muted"))

    def warn(self, message: str) -> None:
        line = status_label("WARN")
        line.append(" " + scrub(message))
        self._print(line)

    def debug(self, message: str) -> None:
        if self._ui.verbose:
            self._print(Text("  debug: " + scrub(message), style="muted"))


# --------------------------------------------------------------------------------------
# Rich live display
# --------------------------------------------------------------------------------------


class _BarOrBlank(ProgressColumn):
    """A bar only when the total is known; ASCII bar in ASCII mode."""

    def __init__(self, ascii_only: bool, width: int = 24) -> None:
        super().__init__()
        self._ascii = ascii_only
        self._width = width
        self._bar = BarColumn(bar_width=width, style="muted", complete_style="accent",
                              finished_style="ok")

    def render(self, task: Task) -> Any:
        if not task.total:
            return Text("")
        if not self._ascii:
            return self._bar.render(task)
        filled = int(self._width * min(1.0, task.completed / task.total))
        return Text("[" + "#" * filled + "-" * (self._width - filled) + "]", style="accent")


class _AmountColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        unit = task.fields.get("unit", "bytes")
        done = fmt_amount(task.completed, unit)
        if task.total:
            pct = min(100.0, 100.0 * task.completed / task.total)
            return Text(f"{done}/{fmt_amount(task.total, unit)} {pct:3.0f}%", style="muted")
        return Text(f"{done}" if task.completed else "", style="muted")


class _RateEtaColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        parts: list[str] = []
        unit = task.fields.get("unit", "bytes")
        rate = task.fields.get("rate")
        if rate is None and unit == "bytes" and task.completed:
            rate = task.speed
        if _known(rate) and unit == "bytes":
            parts.append(f"{fmt_bytes(rate)}/s")
        eta = task.fields.get("eta")
        if eta is None and task.total:
            eta = task.time_remaining
        if _known(eta) and task.total:
            parts.append(f"ETA {fmt_eta(eta)}")
        return Text("  ".join(parts), style="muted")


class RichProgress(_StageMixin):
    """Live progress on a terminal. Started lazily on the first task."""

    def __init__(self, ui: UI) -> None:
        self._ui = ui
        self._lock = threading.RLock()
        spinner = "line" if ui.ascii else "dots"
        self._progress = Progress(
            SpinnerColumn(spinner_name=spinner, style="accent"),
            TextColumn("{task.description}",
                       table_column=Column(no_wrap=True, overflow="ellipsis", ratio=1)),
            _BarOrBlank(ui.ascii),
            _AmountColumn(),
            _RateEtaColumn(),
            console=ui.err,
            transient=True,
            expand=False,
            auto_refresh=True,
            refresh_per_second=8,
        )
        self._ids: dict[str, TaskID] = {}
        self._units: dict[str, str] = {}
        self._started = False
        self._paused = False

    def __enter__(self) -> RichProgress:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._started:
                self._progress.stop()
                self._started = False

    @contextlib.contextmanager
    def paused(self) -> Iterator[None]:
        """Stop the live display (e.g. while prompting), then resume it."""
        with self._lock:
            was_started = self._started
            if was_started:
                self._progress.stop()
                self._started = False
        try:
            yield
        finally:
            if was_started and self._ids:
                self._ensure_started()

    def _ensure_started(self) -> None:
        with self._lock:
            if not self._started:
                self._progress.start()
                self._started = True

    def _print(self, text: Text) -> None:
        self._ui.err.print(text)

    def stage(self, stage: Stage, message: str | None = None) -> None:
        header = self._stage_text(stage, message)
        if header is not None:
            self._print(header)
        elif message:
            self.info(message)

    def task_start(self, task_id: str, description: str, total: float | None = None,
                   unit: str = "bytes") -> None:
        with self._lock:
            self._ensure_started()
            self._units[task_id] = unit
            self._ids[task_id] = self._progress.add_task(
                scrub(description), total=total if _known(total) and total else None,
                unit=unit, rate=None, eta=None)

    def task_update(self, task_id: str, *, completed: float | None = None,
                    total: float | None = None, rate: float | None = None,
                    eta: float | None = None) -> None:
        tid = self._ids.get(task_id)
        if tid is None:
            return
        fields: dict[str, Any] = {}
        if _known(rate):
            fields["rate"] = rate
        if _known(eta):
            fields["eta"] = eta
        kwargs: dict[str, Any] = {}
        if completed is not None:
            kwargs["completed"] = completed
        if _known(total) and total:
            kwargs["total"] = total
        self._progress.update(tid, **kwargs, **fields)

    def task_done(self, task_id: str, *, ok: bool = True, message: str | None = None) -> None:
        with self._lock:
            tid = self._ids.pop(task_id, None)
            unit = self._units.pop(task_id, "bytes")
            description, completed = task_id, 0.0
            if tid is not None:
                task = self._progress.tasks[[t.id for t in self._progress.tasks].index(tid)]
                description, completed = task.description, task.completed
                self._progress.remove_task(tid)
            line = Text("  ")
            line.append(self._ui.glyphs.check if ok else self._ui.glyphs.cross,
                        style="ok" if ok else "error")
            line.append(f" {description}")
            if completed:
                line.append(f"  {fmt_amount(completed, unit)}", style="muted")
            if message:
                line.append(f"  {scrub(message)}", style="muted")
            self._print(line)
            if not self._ids and self._started:
                self._progress.stop()
                self._started = False

    def info(self, message: str) -> None:
        self._print(Text("  " + scrub(message), style="muted"))

    def warn(self, message: str) -> None:
        line = status_label("WARN")
        line.append(" " + scrub(message))
        self._print(line)

    def debug(self, message: str) -> None:
        if self._ui.verbose:
            self._print(Text("  debug: " + scrub(message), style="muted"))


Sink = RichProgress | LineProgress | QuietProgress


def make_progress(ui: UI) -> Sink:
    """Pick the sink: quiet → nothing; terminal → live Rich; otherwise plain lines."""
    if ui.quiet:
        return QuietProgress()
    if ui.live and not ui.json:
        return RichProgress(ui)
    return LineProgress(ui)
