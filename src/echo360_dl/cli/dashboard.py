"""The 3CH0 live console: a pure function from a telemetry :class:`Snapshot` to lines.

Layout (boxes appear only when they have real data)::

    3CH0 / download                                                  elapsed 00:24
    ╭─ LESSON ───────────────────────────────────────────────────────────────────╮
    │ URL     https://echo360.ca/lesson/…/classroom                              │
    │ TITLE   Lecture 7                                                          │
    │ TARGET  ~/Downloads                                                        │
    ╰────────────────────────────────────────────────────────────────────────────╯
    ╭─ TRANSFER ─────────────────────────────────────────────────────────────────╮
    │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━──────────────  64.8%           remaining 00:13 │
    │ 162.0 / 250.0 MB · 2 active · 1 queued                                     │
    │ current 6.8 MB/s   average 6.7 MB/s   peak 9.2 MB/s                        │
    │ ▁▂▃▅▆▅▄▆█▇▆▅▅▆▇▆▅▆▆▅▆▇▆▅                            throughput · last 24 s │
    ╰────────────────────────────────────────────────────────────────────────────╯
    ╭─ STREAMS ──────────────────────────────────────────────────────────────────╮
    │   STREAM          PROGRESS          SIZE              LEFT   STATE         │
    │ ▸ Source 1 video  ━━━━━━──────  50% 13.5 / 27.0 MB    00:07  fetching      │
    ╰────────────────────────────────────────────────────────────────────────────╯
    ╭─ ENCODE ─  (only while encoding: bar, media time, speed, remaining)
    ╭─ PIPELINE ─  discover ✓ → select ✓ → [download] → subtitles → chapters → verify
    ╭─ EVENTS ─  14:32:08  stage download
    ↑↓ select · d details · ctrl-c stop

Every line is cut to the given width. When the terminal is short, the least important
parts go first (event rows, then the lesson box, the graph, the pipeline...), so the
display never scrolls or wraps.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from rich.cells import cell_len
from rich.text import Text

from echo360_dl.cli.output import fmt_bytes, fmt_clock, fmt_pair, fmt_rate, fmt_span
from echo360_dl.cli.telemetry import STAGES, Event, Snapshot, TaskState
from echo360_dl.cli.theme import BRAND, MARGIN, Glyphs

#: Tasks of these kinds belong to the main work; discovery tasks show only until then.
_WORK_PREFIXES = ("download", "encode", "mux", "transcribe")
#: Long-running non-download work that gets its own box with a bar and a time estimate.
_FOCUS = {"encode": "ENCODE", "transcribe": "TRANSCRIBE"}
MAX_TASK_ROWS = 8
MAX_EVENT_ROWS = 5
KEY_WIDTH = 10  # label column inside boxes


def short_path(path: str | Path) -> str:
    """``~/Downloads/x`` for paths under the home directory."""
    text = str(path)
    try:
        home = str(Path.home())
    except RuntimeError:  # pragma: no cover - no home directory
        return text
    if text == home:
        return "~"
    if text.startswith(home + "/"):
        return "~" + text[len(home):]
    return text


def fit(text: Text, width: int, g: Glyphs) -> Text:
    """Cut ``text`` to ``width`` cells, ending with an ellipsis when cut."""
    if width <= 0:
        return Text("")
    if text.cell_len <= width:
        return text
    out = text.copy()
    out.truncate(max(0, width - cell_len(g.ellipsis)))
    out.append(g.ellipsis, style="muted")
    return out


def fit_left(value: str, width: int, g: Glyphs) -> str:
    """Cut from the start (paths: the file name matters most)."""
    if cell_len(value) <= width:
        return value
    keep = max(0, width - cell_len(g.ellipsis))
    return g.ellipsis + value[len(value) - keep:] if keep else g.ellipsis[:width]


def spread(left: Text, right: Text, width: int, g: Glyphs) -> Text:
    """``left`` and ``right`` on one line; ``right`` is dropped if they do not fit."""
    gap = width - left.cell_len - right.cell_len
    if gap >= 2:
        return Text.assemble(left, " " * gap, right)
    return fit(left, width, g)


def keyed(key: str, value: Text | str, width: int, g: Glyphs) -> Text:
    """``KEY     value`` inside a box."""
    value = value if isinstance(value, Text) else Text(value)
    return fit(Text.assemble((key.upper().ljust(KEY_WIDTH), "label"), value), width, g)


def bar(fraction: float, width: int, g: Glyphs) -> Text:
    filled = round(width * max(0.0, min(1.0, fraction)))
    return Text.assemble((g.fill * filled, "bar.done"), (g.rest * (width - filled), "bar.rest"))


def sparkline(values: Sequence[float], g: Glyphs) -> Text:
    """One character per value, scaled to the largest value shown."""
    peak = max(values, default=0.0)
    levels = len(g.spark)
    text = Text(style="spark")
    for value in values:
        if peak <= 0 or value <= 0:
            text.append(g.spark[0], style="muted")
        else:
            text.append(g.spark[min(levels - 1, max(1, round(value / peak * (levels - 1))))])
    return text


def pct(fraction: float | None, digits: int = 0) -> str:
    return "" if fraction is None else f"{fraction * 100:.{digits}f}%"


def box(title: str, body: list[Text], width: int, g: Glyphs,
        title_style: str = "title") -> list[Text]:
    """``body`` (lines at most ``width - 4`` cells) inside a thin titled box."""
    tl, tr, bl, br = g.corners
    inner = width - 4
    label = f" {title} "
    top = Text.assemble((tl + g.h, "border"), (label, title_style),
                        (g.h * max(0, width - 3 - cell_len(label)) + tr, "border"))
    lines = [top]
    for line in body:
        line = fit(line, inner, g)
        lines.append(Text.assemble((g.v + " ", "border"), line,
                                   " " * max(0, inner - line.cell_len), (" " + g.v, "border")))
    lines.append(Text(bl + g.h * (width - 2) + br, style="border"))
    return lines


# --------------------------------------------------------------------------------------
# Section bodies (each returns the lines that go inside its box)
# --------------------------------------------------------------------------------------


def header(snap: Snapshot, command: str, width: int, g: Glyphs, *, pipeline: bool) -> Text:
    left = Text.assemble((BRAND, "heading"), (" / ", "muted"), (command, "muted"))
    if not pipeline and snap.stage is not None:
        left.append(f"{g.sep}{snap.stage.value}", style="stage")
    status = snap.context.get("status")
    if status:
        left.append(f"   {status}", style="warn")
    right = Text.assemble(("elapsed ", "muted"), fmt_clock(snap.elapsed))
    return spread(left, right, width, g)


def lesson_body(snap: Snapshot, width: int, g: Glyphs) -> list[Text]:
    lines: list[Text] = []
    for key, name in (("source", "URL"), ("title", "TITLE"), ("target", "TARGET")):
        value = snap.context.get(key)
        if not value:
            continue
        if key == "target":
            value = fit_left(short_path(value), width - KEY_WIDTH, g)
        lines.append(keyed(name, value, width, g))
    return lines


def transfer_body(snap: Snapshot, width: int, g: Glyphs, *, graph: bool = True,
                  network: bool = True) -> list[Text]:
    agg = snap.aggregate
    if not agg.tasks:
        return []
    lines: list[Text] = []
    if snap.remaining is not None:
        right = Text.assemble(("remaining ", "muted"), fmt_clock(snap.remaining))
    elif agg.active:
        right = Text.assemble(("remaining ", "muted"), (g.dash, "muted"))
    else:
        right = Text("")
    if agg.fraction is not None:
        bar_width = max(8, min(48, width - 8 - right.cell_len - 2))
        lines.append(spread(Text.assemble(bar(agg.fraction, bar_width, g), "  ",
                                          (pct(agg.fraction, 1), "accent")), right, width, g))
        amount = Text(fmt_pair(agg.done, agg.total))
    else:
        lines.append(spread(Text("total not known yet", style="muted"), right, width, g))
        amount = Text(fmt_bytes(agg.done))
    counts: list[str] = []
    if agg.active:
        counts.append(f"{agg.active} active")
    if agg.queued:
        counts.append(f"{agg.queued} queued")
    if agg.reused:
        counts.append(f"{fmt_bytes(agg.reused)} reused")
    if counts:
        amount.append(g.sep + g.sep.join(counts), style="muted")
    lines.append(fit(amount, width, g))
    rates = snap.rates
    if network and (rates.transferred > 0 or rates.measuring):
        def item(name: str, value: float | None, waiting: str) -> Text:
            shown = fmt_rate(value) if value is not None else waiting
            accent = name == "current" and value is not None
            return Text.assemble((name + " ", "muted"), (shown, "accent" if accent else ""))

        current_waiting = "measuring" if rates.measuring else (g.dash if agg.active else "idle")
        items = [item("current", rates.current, current_waiting),
                 item("average", rates.average, g.dash), item("peak", rates.peak, g.dash)]
        joined = Text("   ").join(items)
        lines.append(fit(joined if joined.cell_len <= width else Text("  ").join(items),
                         width, g))
        if graph and rates.history:
            for template in ("throughput " + g.dot + " last {n} s", "last {n} s"):
                columns = width - cell_len(template.format(n=len(rates.history))) - 2
                if columns >= 8:
                    values = rates.history[-columns:]
                    right = Text(template.format(n=len(values)), style="muted")
                    lines.append(spread(sparkline(values, g), right, width, g))
                    break
    return lines


def visible_tasks(snap: Snapshot) -> list[TaskState]:
    work = [t for t in snap.tasks if t.id.split(":", 1)[0] in _WORK_PREFIXES]
    return work or list(snap.tasks)


def _rows_window(tasks: list[TaskState], rows: int, selected: str | None) -> list[TaskState]:
    """At most ``rows`` tasks: keep order, prefer unfinished ones and the selection."""
    if len(tasks) <= rows:
        return tasks
    keep = {t.id for t in tasks if t.state in ("active", "queued", "failed")}
    if selected:
        keep.add(selected)
    chosen = [t for t in tasks if t.id in keep]
    if len(chosen) > rows:
        index = next((i for i, t in enumerate(chosen) if t.id == selected), 0)
        start = max(0, min(index - rows // 2, len(chosen) - rows))
        return chosen[start:start + rows]
    finished = [t for t in tasks if t.id not in keep]
    chosen_ids = keep | {t.id for t in finished[len(finished) - (rows - len(chosen)):]}
    return [t for t in tasks if t.id in chosen_ids]


_STATE_STYLE = {"active": "accent", "queued": "muted", "done": "muted", "reused": "muted",
                "failed": "error", "cancelled": "warn"}
_RESOLUTION_SUFFIX = re.compile(r" \((\d+p|\d+x\d+|s\d+q\d+)\)$")


def _columns(width: int, has_format: bool, has_rate: bool) -> dict[str, int]:
    """Column widths for the stream table, dropping detail as the box narrows."""
    cols = {"format": 0, "bar": 0, "pct": 6, "size": 0, "rate": 0, "left": 0, "state": 10}
    if width >= 40:
        cols["bar"] = 8
    if width >= 56:
        cols["bar"], cols["left"] = 10, 7
    if width >= 66:
        cols["size"] = 18
    if width >= 84:
        cols["bar"] = 12
        if has_rate:
            cols["rate"] = 12
    if width >= 90 and has_format:  # FORMAT before per-stream RATE (the total rate is above)
        cols["format"] = 15
        if width < 110:
            cols["rate"] = 0
    if width >= 120:
        cols["bar"] = 16
    return cols


def _short_name(task: TaskState, with_format: bool) -> str:
    """Video labels end in "(1280x720)"; drop that when the FORMAT column shows it."""
    return _RESOLUTION_SUFFIX.sub("", task.description) if with_format else task.description


def _left(task: TaskState) -> str:
    return fmt_clock(task.left) if task.running and task.left is not None else ""


def streams_body(snap: Snapshot, width: int, g: Glyphs, *, selected: str | None,
                 markers: bool, rows: int) -> list[Text]:
    tasks = visible_tasks(snap)
    if not tasks or rows < 2:
        return []
    shown = _rows_window(tasks, rows - 1, selected)
    mark = 2 if markers else 0
    cols = _columns(width, any(t.id in snap.streams for t in shown),
                    any(t.unit == "bytes" for t in shown))
    if not any(t.total for t in shown):  # nothing measurable (e.g. discovery): no empty columns
        cols.update(bar=0, size=0, left=0, rate=0)
    cols["state"] = max(cols["state"], max(cell_len(t.activity) for t in shown) + 2)
    name_room = width - mark - sum(cols.values()) - (1 if cols["bar"] else 0)
    # The "(1280x720)" suffix goes first when the FORMAT column shows it or space is short.
    names = {t.id: _short_name(t, t.id in snap.streams and (
        cols["format"] > 0 or cell_len(t.description) + 2 > name_room)) for t in shown}
    longest = max(cell_len(n) for n in names.values()) + 2
    name_width = max(8, min(longest, 30, name_room))

    def cell(value: str | Text, w: int, style: str = "", right: bool = False) -> Text:
        if w <= 0:
            return Text("")
        text = value if isinstance(value, Text) else Text(value, style=style)
        text = fit(text, max(1, w - 2), g)
        if right:
            return Text.assemble(" " * max(0, w - 2 - text.cell_len), text, "  ")
        return Text.assemble(text, " " * max(0, w - text.cell_len))

    downloads = all(t.id.startswith("download:") for t in tasks)
    head = Text.assemble(
        " " * mark, cell("STREAM" if downloads else "TASK", name_width, "label"),
        cell("FORMAT", cols["format"], "label"),
        cell("PROGRESS", cols["bar"] + 1 + cols["pct"], "label") if cols["bar"]
        else cell("", cols["pct"]),
        cell("SIZE", cols["size"], "label"), cell("RATE", cols["rate"], "label"),
        cell("LEFT", cols["left"], "label"), cell("STATE", cols["state"], "label"))
    hidden = len(tasks) - len(shown)
    if hidden > 0:
        head.append(f"+{hidden} more", style="muted")
    lines = [head]
    for task in shown:
        is_selected = markers and task.id == selected
        fmt = snap.streams.get(task.id, (g.dash, g.dash))[0]
        fraction = task.fraction if task.state != "queued" else (0.0 if task.total else None)
        if cols["bar"]:
            if fraction is not None:
                progress = Text.assemble(bar(fraction, cols["bar"], g), " ")
            elif task.state == "queued" or task.unit != "bytes":
                progress = Text(" " * (cols["bar"] + 1))
            else:  # downloading, size not reported yet
                progress = Text(fit_left("size unknown", cols["bar"], g).ljust(cols["bar"] + 1),
                                style="muted")
        else:
            progress = Text("")
        if task.state == "queued":
            size = fmt_pair(0, task.total, task.unit) if task.total else g.dash
        elif task.unit == "items" and task.total is None:
            size = ""
        else:
            size = fmt_pair(task.completed, task.total, task.unit)
        rate = fmt_rate(task.rate) if task.running and task.unit == "bytes" else ""
        lines.append(Text.assemble(
            (g.marker + " " if is_selected else "  ") if markers else "",
            cell(Text(names[task.id], style="selected" if is_selected else ""), name_width),
            cell(fmt, cols["format"], "muted"), progress,
            cell(pct(fraction), cols["pct"], "muted", right=True),
            cell(size, cols["size"]), cell(rate, cols["rate"]), cell(_left(task), cols["left"]),
            cell(task.activity, cols["state"], _STATE_STYLE.get(task.state, ""))))
    return [fit(line, width, g) for line in lines]


def focus_tasks(snap: Snapshot) -> list[TaskState]:
    """Active encode/transcribe tasks (shown in their own box)."""
    return [t for t in snap.tasks if t.running and t.id.split(":", 1)[0] in _FOCUS]


def focus_body(task: TaskState, width: int, g: Glyphs) -> list[Text]:
    lines = [fit(Text(task.description), width, g)]
    if task.left is not None:
        right = Text.assemble(("remaining ", "muted"), fmt_clock(task.left))
    else:
        right = Text.assemble(("remaining ", "muted"), (g.dash, "muted"))
    if task.fraction is not None:
        bar_width = max(8, min(48, width - 8 - right.cell_len - 2))
        lines.append(spread(Text.assemble(bar(task.fraction, bar_width, g), "  ",
                                          (pct(task.fraction, 1), "accent")), right, width, g))
    else:
        lines.append(spread(Text("length not known", style="muted"), right, width, g))
    detail: list[str] = []
    if task.unit == "seconds":
        done = fmt_clock(task.completed or 0)
        detail.append(f"{done} / {fmt_clock(task.total)} of media" if task.total
                      else f"{done} of media")
        detail.append(f"speed {task.speed:.1f}{g.times}" if task.speed else "speed measuring")
    elif task.completed is not None:
        detail.append(fmt_pair(task.completed, task.total, task.unit))
    lines.append(fit(Text(g.sep.join(detail), style="muted"), width, g))
    return lines


def pipeline_body(snap: Snapshot, width: int, g: Glyphs) -> list[Text]:
    current = snap.stage
    if current is None:
        return []
    arrow = f" {g.arrow} "
    line = Text()
    for i, stage in enumerate(STAGES):
        if i:
            line.append(arrow, style="muted")
        if stage == current:
            line.append(f"[{stage.value}]", style="stage")
        elif stage in snap.stages_seen:
            line.append(f"{stage.value} {g.check}")
        else:
            line.append(stage.value, style="muted")
    if line.cell_len <= width:
        return [line]
    index = STAGES.index(current)
    short = Text.assemble((f"{index + 1}/{len(STAGES)} ", "muted"),
                          (f"[{current.value}]", "stage"))
    if index + 1 < len(STAGES):
        short.append(f"{arrow}{STAGES[index + 1].value}", style="muted")
    return [fit(short, width, g)]


def _event_text(event: Event, g: Glyphs) -> Text:
    text = Text()
    if event.level == "warn":
        text.append("[WARN] ", style="warn")
    elif event.level == "fail":
        text.append("[FAIL] ", style="error")
    text.append(event.text, style="muted" if event.level == "stage" else "")
    if event.count > 1:
        text.append(f" {g.times}{event.count}", style="muted")
    return text


def events_body(snap: Snapshot, width: int, g: Glyphs, rows: int) -> list[Text]:
    events = list(snap.events)[-rows:] if rows > 0 else []
    lines: list[Text] = []
    for event in events:
        stamp = Text(event.wall.strftime("%H:%M:%S") + "  ", style="muted")
        body = spread(_event_text(event, g), Text(event.detail, style="muted"),
                      width - stamp.cell_len, g)
        lines.append(fit(Text.assemble(stamp, body), width, g))
    return lines


def detail_body(snap: Snapshot, width: int, g: Glyphs, selected: str | None,
                rows: int) -> list[Text]:
    task = snap.task(selected)
    lines: list[Text] = []
    if task is not None:
        started = snap.wall_at(task.started).strftime("%H:%M:%S") \
            if task.started is not None else g.dash
        ran = fmt_span(snap.duration(task)) or g.dash
        amount = fmt_pair(task.completed, task.total, task.unit)
        if task.fraction is not None:
            amount += f" ({pct(task.fraction, 1)})"
        elif task.total is None and task.state != "queued":
            amount += f"{g.sep}total unknown"
        lines += [keyed("task", task.description, width, g),
                  keyed("id", task.id, width, g),
                  keyed("state", f"{task.activity}{g.sep}started {started}{g.sep}ran {ran}",
                        width, g),
                  keyed("amount", amount, width, g)]
        if task.running and task.unit == "bytes":
            reported = fmt_rate(task.rate) or g.dash
            eta = fmt_clock(task.eta) or g.dash
            lines.append(keyed("reported", f"{reported}{g.sep}remaining {eta} (from yt-dlp)",
                               width, g))
        elif task.running and task.speed:
            lines.append(keyed("measured", f"{task.speed:.2f} {task.unit}/s{g.sep}remaining "
                               f"{fmt_clock(task.left) or g.dash}", width, g))
        fmt, codec = snap.streams.get(task.id, ("", ""))
        if fmt:
            lines.append(keyed("format", f"{fmt}{g.sep}{codec}", width, g))
        if task.message:
            lines.append(keyed("message", task.message, width, g))
    room = max(0, rows - len(lines))
    for wall, message in list(snap.debug)[-room:] if room else []:
        lines.append(fit(Text.assemble((wall.strftime("%H:%M:%S") + "  ", "muted"), message),
                         width, g))
    if not lines:
        lines.append(Text("nothing selected", style="muted"))
    return lines[:max(1, rows)]


def footer(g: Glyphs, width: int, expanded: bool) -> Text:
    keys = "↑↓" if g.arrow == "→" else "up/down"
    text = Text(f"{keys} select{g.sep}d {'hide details' if expanded else 'details'}"
                f"{g.sep}ctrl-c stop", style="muted")
    return fit(text, width, g)


# --------------------------------------------------------------------------------------
# Whole view
# --------------------------------------------------------------------------------------


@dataclass
class _Plan:
    """What to include; shrunk step by step until the frame fits the terminal height."""

    lesson: bool = True
    graph: bool = True
    network: bool = True
    pipeline: bool = True
    events: int = MAX_EVENT_ROWS
    detail: int = 12
    task_rows: int = MAX_TASK_ROWS
    boxes: bool = True


def _dec(plan: _Plan, name: str, floor: int) -> bool:
    value = getattr(plan, name)
    if value <= floor:
        return False
    setattr(plan, name, value - 1)
    return True


def _off(plan: _Plan, name: str) -> bool:
    if not getattr(plan, name):
        return False
    setattr(plan, name, False)
    return True


_SHRINK: tuple[Callable[[_Plan], bool], ...] = (
    lambda p: _dec(p, "events", 2) or _dec(p, "detail", 4),
    lambda p: _off(p, "lesson"),
    lambda p: _off(p, "graph"),
    lambda p: _dec(p, "events", 0) or _dec(p, "detail", 1),
    lambda p: _off(p, "pipeline"),
    lambda p: _off(p, "boxes"),
    lambda p: _off(p, "network"),
    lambda p: _dec(p, "task_rows", 1),
    lambda p: _dec(p, "detail", 0),
)


def render(snap: Snapshot, g: Glyphs, *, width: int, height: int, command: str,
           selected: str | None = None, expanded: bool = False, keys: bool = False,
           pipeline: bool = True) -> list[Text]:
    """The live view as lines (each at most ``width`` cells, at most ``height`` lines)."""
    outer = max(18, width - MARGIN)
    budget = max(3, height)
    task_count = len(visible_tasks(snap))

    def build(plan: _Plan) -> list[Text]:
        inner = outer - 4 if plan.boxes else outer

        def section(title: str, body: list[Text]) -> list[Text]:
            if not body:
                return []
            return box(title, body, outer, g) if plan.boxes else body

        lines = [header(snap, command, outer, g, pipeline=pipeline)]
        if plan.lesson:
            lines += section("LESSON", lesson_body(snap, inner, g))
        lines += section("TRANSFER", transfer_body(snap, inner, g, graph=plan.graph,
                                                   network=plan.network))
        lines += section("STREAMS", streams_body(snap, inner, g, selected=selected,
                                                 markers=keys,
                                                 rows=min(plan.task_rows, task_count) + 1))
        for task in focus_tasks(snap):
            lines += section(_FOCUS[task.id.split(":", 1)[0]], focus_body(task, inner, g))
        if pipeline and plan.pipeline:
            lines += section("PIPELINE", pipeline_body(snap, inner, g))
        if expanded:
            if plan.detail > 0:
                lines += section("DETAIL", detail_body(snap, inner, g, selected, plan.detail))
        elif plan.events > 0:
            lines += section("EVENTS", events_body(snap, inner, g, plan.events))
        if keys:
            lines.append(footer(g, outer, expanded))
        return lines

    plan = _Plan()
    lines = build(plan)
    for step in _SHRINK:
        while len(lines) > budget and step(plan):
            lines = build(plan)
        if len(lines) <= budget:
            break
    pad = " " * MARGIN
    out = []
    for line in lines[:budget]:
        line = Text.assemble(pad, line)
        line.rstrip()
        out.append(line)
    return out
