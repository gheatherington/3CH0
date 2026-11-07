"""Telemetry for the 3CH0 console: what the pipeline reported, plus metrics derived from it.

The only inputs are the :class:`~echo360_dl.models.ProgressSink` calls the discovery and
media code already make (``stage``, ``task_*``, ``info``/``warn``/``debug``) and a few
context strings the CLI sets. Nothing here makes a request or touches a file.

Reported values (shown as reported):
    per-task completed/total amounts (byte totals come from the HLS playlist's byte
    ranges), yt-dlp's per-task speed and the ETA derived from it, stage changes,
    task start/finish, messages.

Derived values (and how):
    transferred  positive deltas of each byte task's ``completed`` after its first
                 report, so bytes resumed from an earlier run or a reused file are
                 never counted as network traffic.
    current      bytes transferred in the last :data:`RATE_WINDOW` seconds / window.
    average      transferred / time from the first byte to the last (or now while active).
    peak         highest ``current`` value seen once a full window had elapsed.
    history      bytes per one-second bucket, last :data:`HISTORY_SECONDS` seconds.
    overall      done / total over byte tasks, only when every byte task has reported a
                 total (queued ones included: the pipeline reads their playlist sizes
                 ahead); otherwise the total is unknown.
    remaining    (total - done) / current, only when both are known.
    task speed   per task, progress units per second over the last :data:`TASK_WINDOW`
                 seconds (bytes/s, or media seconds per second for encoding); a task's
                 ``left`` is yt-dlp's ETA for downloads, else (total - done) / speed.

Everything is bounded (events, debug lines, samples, history) and thread-safe: the
pipeline reports from worker threads while the renderer reads snapshots.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from echo360_dl.models import Stage
from echo360_dl.redact import scrub

RATE_WINDOW = 3.0
TASK_WINDOW = 5.0
HISTORY_SECONDS = 180
MAX_EVENTS = 100
MAX_DEBUG = 60
MAX_TASKS = 64  # finished tasks beyond this are forgotten (oldest first)

STAGES: tuple[Stage, ...] = tuple(Stage)

#: Verb shown while a task runs, by task-id prefix (ids are set by the pipeline:
#: ``download:<kind>-<rendition>``, ``encode:<file>``, ``mux:<file>``, ``transcribe``,
#: ``discover``).
_ACTIVITY = {"download": "fetching", "encode": "encoding", "mux": "writing",
             "transcribe": "transcribing", "discover": "observing"}


def activity_for(task_id: str) -> str:
    return _ACTIVITY.get(task_id.split(":", 1)[0], "running")


def _known(value: float | None) -> bool:
    return value is not None and value == value and value >= 0  # excludes NaN/negative


@dataclass
class TaskState:
    id: str
    description: str
    unit: str = "bytes"
    total: float | None = None
    completed: float | None = None
    rate: float | None = None  # as reported (yt-dlp speed)
    eta: float | None = None  # as reported
    state: str = "queued"  # queued | active | done | reused | failed | cancelled
    started: float | None = None
    finished: float | None = None
    message: str | None = None
    order: int = 0
    speed: float | None = None  # derived: units per second
    left: float | None = None  # seconds remaining (reported ETA, else derived)
    _baseline: float | None = None  # last ``completed`` seen, for transfer deltas
    _trail: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=64))

    @property
    def fraction(self) -> float | None:
        if self.total and self.completed is not None:
            return max(0.0, min(1.0, self.completed / self.total))
        return None

    @property
    def running(self) -> bool:
        return self.state == "active"

    @property
    def activity(self) -> str:
        return activity_for(self.id) if self.state == "active" else self.state


@dataclass(frozen=True)
class Event:
    wall: datetime
    level: str  # info | warn | fail | ok | stage
    text: str
    detail: str = ""
    count: int = 1


@dataclass(frozen=True)
class Aggregate:
    done: float
    total: float | None
    reused: float
    tasks: int
    active: int
    queued: int

    @property
    def fraction(self) -> float | None:
        if self.total:
            return max(0.0, min(1.0, self.done / self.total))
        return None


@dataclass(frozen=True)
class Rates:
    transferred: float
    current: float | None
    average: float | None
    peak: float | None
    history: tuple[float, ...]  # bytes per second, oldest first, last bucket = now
    measuring: bool  # bytes are flowing but the first window is not complete yet
    span: float | None = None  # seconds from the first transferred byte to the last/now


@dataclass(frozen=True)
class Snapshot:
    now: float
    wall: datetime
    elapsed: float
    context: dict[str, str]
    stage: Stage | None
    stages_seen: tuple[Stage, ...]
    tasks: tuple[TaskState, ...]
    aggregate: Aggregate
    rates: Rates
    remaining: float | None
    events: tuple[Event, ...]
    debug: tuple[tuple[datetime, str], ...]
    streams: dict[str, tuple[str, str]] = field(default_factory=dict)

    def task(self, task_id: str | None) -> TaskState | None:
        return next((t for t in self.tasks if t.id == task_id), None)

    def wall_at(self, moment: float) -> datetime:
        """Wall-clock time of a monotonic ``moment`` (for display)."""
        return self.wall - timedelta(seconds=self.now - moment)

    def duration(self, task: TaskState) -> float | None:
        """Seconds the task ran (so far, while active)."""
        if task.started is None:
            return None
        return (task.finished if task.finished is not None else self.now) - task.started


@dataclass
class Telemetry:
    clock: Callable[[], float] = time.monotonic
    wall: Callable[[], datetime] = datetime.now
    started: float = field(default=0.0)

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self.started = self.clock()
        self._tasks: dict[str, TaskState] = {}
        self._order = 0
        self._context: dict[str, str] = {}
        self._streams: dict[str, tuple[str, str]] = {}
        self._stage: Stage | None = None
        self._stages_seen: list[Stage] = []
        self._events: deque[Event] = deque(maxlen=MAX_EVENTS)
        self._debug: deque[tuple[datetime, str]] = deque(maxlen=MAX_DEBUG)
        # transfer accounting
        self._transferred = 0.0
        self._first_byte: float | None = None
        self._last_byte: float | None = None
        self._samples: deque[tuple[float, float]] = deque(maxlen=4096)
        self._buckets: deque[list[float]] = deque(maxlen=HISTORY_SECONDS)  # [second, bytes]
        self._peak: float | None = None
        self._peak_second: int | None = None

    # -- context and events ------------------------------------------------------------
    def set_context(self, **values: str | None) -> None:
        with self._lock:
            for key, value in values.items():
                if value:
                    self._context[key] = scrub(str(value))
                else:
                    self._context.pop(key, None)

    def set_streams(self, details: dict[str, tuple[str, str]]) -> None:
        """Static (format, codec) per task id, from the recording's playlists."""
        with self._lock:
            self._streams = {k: (scrub(f), scrub(c)) for k, (f, c) in details.items()}

    def event(self, level: str, text: str, detail: str = "") -> None:
        text = scrub(text)
        with self._lock:
            last = self._events[-1] if self._events else None
            if last is not None and last.text == text and last.level == level:
                # Repeated messages collapse into one line with a count.
                self._events[-1] = Event(last.wall, level, text, detail or last.detail,
                                         last.count + 1)
                return
            self._events.append(Event(self.wall(), level, text, scrub(detail)))

    def debug(self, message: str) -> None:
        with self._lock:
            self._debug.append((self.wall(), scrub(message)))

    def stage(self, stage: Stage) -> bool:
        """Record a stage change; False when ``stage`` is already current."""
        with self._lock:
            if stage == self._stage:
                return False
            self._stage = stage
            if stage not in self._stages_seen:
                self._stages_seen.append(stage)
            self.event("stage", f"stage {stage.value}")
            return True

    @property
    def current_stage(self) -> Stage | None:
        return self._stage

    # -- tasks -------------------------------------------------------------------------
    def _new(self, task_id: str, description: str, unit: str) -> TaskState:
        self._order += 1
        task = TaskState(task_id, scrub(description), unit, order=self._order)
        self._tasks[task_id] = task
        if len(self._tasks) > MAX_TASKS:
            for old in sorted(self._tasks.values(), key=lambda t: t.order):
                if old.state not in ("active", "queued"):
                    del self._tasks[old.id]
                    break
        return task

    def task_queued(self, task_id: str, description: str, unit: str = "bytes",
                    total: float | None = None) -> None:
        with self._lock:
            task = self._tasks.get(task_id) or self._new(task_id, description, unit)
            if _known(total) and total:
                task.total = total

    def task_start(self, task_id: str, description: str, total: float | None,
                   unit: str) -> TaskState:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.state not in ("queued",):
                task = self._new(task_id, description, unit)
            task.description = scrub(description)
            task.unit = unit
            task.state = "active"
            task.started = self.clock()
            task.finished = None
            task.message = None
            task.completed = None
            task._baseline = None
            task._trail.clear()
            task.speed = task.left = None
            if _known(total) and total:
                task.total = total  # else keep a size reported while it was queued
            return task

    def task_update(self, task_id: str, *, completed: float | None = None,
                    total: float | None = None, rate: float | None = None,
                    eta: float | None = None) -> TaskState | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if _known(total) and total:
                task.total = total
            if _known(rate):
                task.rate = rate
            if _known(eta):
                task.eta = eta
            if completed is not None and _known(completed):
                task.completed = completed
                if task.unit == "bytes":
                    self._account(task, completed)
                self._track(task, completed)
            if task.state == "active":
                task.left = task.eta if task.eta is not None else self._left(task)
            return task

    def _track(self, task: TaskState, completed: float) -> None:
        now = self.clock()
        trail = task._trail
        if trail and completed < trail[-1][1]:
            trail.clear()  # restarted
        trail.append((now, completed))
        while len(trail) > 2 and now - trail[1][0] >= TASK_WINDOW:
            trail.popleft()
        t0, c0 = trail[0]
        task.speed = (completed - c0) / (now - t0) if now - t0 >= 1.0 else None

    @staticmethod
    def _left(task: TaskState) -> float | None:
        if task.total is None or task.completed is None or not task.speed:
            return None
        return max(0.0, task.total - task.completed) / task.speed

    def task_done(self, task_id: str, *, ok: bool, message: str | None) -> TaskState | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.finished = self.clock()
            task.message = scrub(message) if message else None
            text = (message or "").lower()
            if ok:
                task.state = "reused" if "reused" in text else "done"
            else:
                task.state = "cancelled" if "cancel" in text else "failed"
            task.rate = None
            task.eta = None
            task.left = None
            return task

    # -- transfer accounting -----------------------------------------------------------
    def _account(self, task: TaskState, completed: float) -> None:
        now = self.clock()
        if task._baseline is None or completed < task._baseline:
            # First report (may include resumed bytes) or a restart: new baseline.
            task._baseline = completed
            if self._first_byte is None:
                self._first_byte = now
                self._samples.append((now, self._transferred))
            return
        delta = completed - task._baseline
        task._baseline = completed
        if delta <= 0:
            return
        self._transferred += delta
        self._last_byte = now
        self._samples.append((now, self._transferred))
        second = int(now - self.started)
        if self._buckets and self._buckets[-1][0] == second:
            self._buckets[-1][1] += delta
        else:
            self._buckets.append([second, delta])
        self._update_peak(now, second)

    def _rate_at(self, now: float) -> float | None:
        if self._first_byte is None:
            return None
        span = min(RATE_WINDOW, now - self._first_byte)
        if span < 1.0:
            return None
        cutoff = now - span
        while len(self._samples) > 2 and self._samples[1][0] <= cutoff:
            self._samples.popleft()
        if not self._samples:
            return 0.0
        t0, cum0 = self._samples[0]
        if t0 > cutoff:  # no sample before the window start: use what we have
            span = max(now - t0, 1.0)
        return max(0.0, (self._transferred - cum0) / span)

    def _update_peak(self, now: float, second: int) -> None:
        if self._first_byte is None or now - self._first_byte < RATE_WINDOW:
            return
        if self._peak_second == second:
            return
        self._peak_second = second
        rate = self._rate_at(now)
        if rate is not None and (self._peak is None or rate > self._peak):
            self._peak = rate

    # -- reading -----------------------------------------------------------------------
    def _aggregate(self) -> Aggregate:
        done = reused = 0.0
        total: float | None = 0.0
        count = active = queued = 0
        for task in self._tasks.values():
            if task.unit != "bytes" or task.id.split(":", 1)[0] != "download":
                continue
            if task.state in ("failed", "cancelled"):
                continue
            count += 1
            active += task.state == "active"
            queued += task.state == "queued"
            done += task.completed or 0.0
            if task.state == "reused":
                reused += task.completed or 0.0
            if task.total is None:
                total = None
            elif total is not None:
                total += task.total
        return Aggregate(done, total if count else None, reused, count, active, queued)

    def _history(self, now: float) -> tuple[float, ...]:
        if not self._buckets:
            return ()
        last = int(now - self.started)
        first = max(self._buckets[0][0], last - HISTORY_SECONDS + 1)
        values = {int(s): b for s, b in self._buckets}
        return tuple(values.get(s, 0.0) for s in range(first, last + 1))

    def snapshot(self) -> Snapshot:
        with self._lock:
            now = self.clock()
            agg = self._aggregate()
            active_bytes = any(t.state == "active" and t.unit == "bytes"
                               for t in self._tasks.values())
            current = self._rate_at(now) if active_bytes else None
            if current is not None and now - (self._first_byte or now) >= RATE_WINDOW and (
                    self._peak is None or current > self._peak):
                self._peak = current  # every full window counts, not just bucket edges
            average = span = None
            if self._first_byte is not None and self._transferred > 0:
                end = now if active_bytes else (self._last_byte or now)
                span = end - self._first_byte
                average = self._transferred / span if span >= 1.0 else None
            measuring = active_bytes and self._first_byte is not None and current is None
            remaining = None
            if agg.total is not None and current and agg.active:
                remaining = max(0.0, agg.total - agg.done) / current
            rates = Rates(self._transferred, current, average, self._peak,
                          self._history(now), measuring, span)
            tasks = tuple(TaskState(**{k: getattr(t, k) for k in (
                "id", "description", "unit", "total", "completed", "rate", "eta", "state",
                "started", "finished", "message", "order", "speed", "left")})
                for t in sorted(self._tasks.values(), key=lambda t: t.order))
            return Snapshot(
                now=now, wall=self.wall(), elapsed=now - self.started,
                context=dict(self._context), stage=self._stage,
                stages_seen=tuple(self._stages_seen), tasks=tasks, aggregate=agg,
                rates=rates, remaining=remaining, events=tuple(self._events),
                debug=tuple(self._debug), streams=dict(self._streams))
