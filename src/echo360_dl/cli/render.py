"""Rich renderables for human output. Pure functions: models in, renderables out.

All user/platform-provided strings are wrapped in :class:`rich.text.Text` (never
interpreted as markup) and free text is passed through :func:`redact.scrub`.
Tables fold or wrap at narrow widths instead of overflowing.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from echo360_dl.cli.output import UI, fmt_bytes
from echo360_dl.cli.theme import Glyphs, status_block, status_label, wordmark
from echo360_dl.errors import QualityUnavailableError, UnavailableError
from echo360_dl.models import (
    AccessState,
    AudioTrack,
    AuthStatus,
    CaptionTrack,
    Chapter,
    ChapterModeKind,
    ChapterSet,
    DownloadRequest,
    DownloadResult,
    Layout,
    OutputFile,
    Recording,
    ResultStatus,
    SubtitleModeKind,
    TranscriptProvenance,
    VerifyReport,
    VideoSource,
)
from echo360_dl.redact import public_url, scrub
from echo360_dl.selection import language_matches, select_rendition
from echo360_dl.timecode import format_clock, format_duration


def T(value: object, style: str | None = None) -> Text:
    return Text(scrub(str(value)) if value is not None else "", style=style or "")


# --------------------------------------------------------------------------------------
# Small vocabularies
# --------------------------------------------------------------------------------------

_CODECS = {"avc1": "H.264", "avc3": "H.264", "hvc1": "HEVC", "hev1": "HEVC", "av01": "AV1",
           "vp09": "VP9", "vp9": "VP9", "mp4a": "AAC", "opus": "Opus", "ac-3": "AC-3",
           "ec-3": "E-AC-3", "flac": "FLAC"}

_VIDEO_CODECS = {"H.264", "HEVC", "AV1", "VP9"}


def codec_names(codecs: str | None) -> list[str]:
    names: list[str] = []
    for part in (codecs or "").split(","):
        key = part.strip().split(".")[0].lower()
        if not key:
            continue
        name = _CODECS.get(key, key)
        if name not in names:
            names.append(name)
    return names


def access_word(state: AccessState, detail: str | None, g: Glyphs) -> Text:
    """Access state as a word (colour is secondary)."""
    if state is AccessState.FORBIDDEN:
        return Text(f"forbidden (403 {g.dash} may exist)", style="warn")
    words = {
        AccessState.AVAILABLE: ("available", "ok"),
        AccessState.UNAUTHORIZED: ("unauthorized", "error"),
        AccessState.EXPIRED: ("expired", "warn"),
        AccessState.NOT_FOUND: ("not found", "warn"),
        AccessState.UNSUPPORTED: ("unsupported", "error"),
        AccessState.TRANSIENT: ("temporary error", "warn"),
        AccessState.UNKNOWN: ("not checked", "muted"),
    }
    word, style = words.get(state, (str(state), "muted"))
    text = Text(word, style=style)
    if detail and state is not AccessState.AVAILABLE:
        text.append(f" ({scrub(detail)})", style="muted")
    return text


def provenance_word(kind: TranscriptProvenance | str | None) -> str:
    return {
        TranscriptProvenance.PLATFORM_CAPTIONS: "platform captions",
        TranscriptProvenance.PLATFORM_TRANSCRIPT: "platform transcript",
        TranscriptProvenance.GENERATED_LOCAL: "generated",
        TranscriptProvenance.USER_FILE: "user file",
    }.get(kind, str(kind) if kind else "unknown")  # type: ignore[arg-type]


def format_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.strftime("%Y-%m-%d %H:%M") if (parsed.hour or parsed.minute) else \
        parsed.strftime("%Y-%m-%d")


def audio_label(rec: Recording, src: VideoSource) -> str:
    track = rec.audio_track(src.audio_track_id)
    if track is not None and (track.shared or len(track.used_by_sources) > 1):
        return "shared"
    if track is not None or src.has_audio:
        return "yes"
    return "no" if src.renditions else "?"


def resolutions(src: VideoSource, g: Glyphs) -> str:
    heights = sorted({r.height for r in src.renditions if r.height})
    if heights:
        return g.sep.join(f"{h}p" for h in heights)
    return g.dash if not src.renditions else f"{len(src.renditions)} variant(s)"


def video_codecs(src: VideoSource, g: Glyphs) -> str:
    names: list[str] = []
    for r in src.renditions:
        for name in codec_names(r.codecs):
            if name in _VIDEO_CODECS and name not in names:
                names.append(name)
    return ", ".join(names) or g.dash


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------


def kv_grid(rows: Iterable[tuple[str, RenderableType | str | None]]) -> Table:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="label", no_wrap=True)
    grid.add_column(overflow="fold")
    for label, value in rows:
        if value is None:
            continue
        grid.add_row(Text(label, style="label"), value if not isinstance(value, str) else T(value))
    return grid


def data_table(ui: UI, *headers: str, title: str | None = None) -> Table:
    table = Table(box=ui.glyphs.table_box, header_style="heading", title=title,
                  title_style="heading", title_justify="left", show_edge=False,
                  pad_edge=False, expand=False, safe_box=True)
    for header in headers:
        table.add_column(header, overflow="fold")
    return table


def header_line(ui: UI, subtitle: str) -> Text | None:
    """Optional wordmark, only on a terminal."""
    if ui.live:
        return wordmark(ui.glyphs, subtitle)
    return None


# --------------------------------------------------------------------------------------
# inspect
# --------------------------------------------------------------------------------------


def recording_header(rec: Recording, ui: UI) -> Table:
    g = ui.glyphs
    return kv_grid([
        ("Title", T(rec.title or "(untitled)", "heading")),
        ("Course", rec.course),
        ("Date", format_date(rec.recorded_at)),
        ("Duration", format_duration(rec.duration) if rec.duration is not None else g.dash),
        ("Lesson", rec.lesson_id),
        ("URL", T(public_url(rec.classroom_url), "path")),
    ])


NARROW = 64  # below this width tables become stacked lists


def sources_table(rec: Recording, ui: UI) -> Table:
    if ui.err.width < NARROW:
        return sources_stacked(rec, ui)
    g = ui.glyphs
    table = data_table(ui, "#", "Source", "Resolutions", "Codec", "Audio", "Access")
    for src in sorted(rec.sources, key=lambda s: s.index):
        name = T(src.display_name)
        if src.label is None:
            name.stylize("muted")
        table.add_row(T(src.index, "accent"), name, T(resolutions(src, g)),
                      T(video_codecs(src, g)), T(audio_label(rec, src)),
                      access_word(src.access, src.access_detail, g))
    return table


def sources_stacked(rec: Recording, ui: UI) -> Table:
    """Narrow-terminal variant: one block per source, values wrap under a hanging indent."""
    g = ui.glyphs
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="accent", no_wrap=True)
    grid.add_column(overflow="fold")
    for src in sorted(rec.sources, key=lambda s: s.index):
        body = Text()
        body.append(scrub(src.display_name), style="heading" if src.label else "")
        body.append(f"\n{resolutions(src, g)}{g.sep}{video_codecs(src, g)}{g.sep}"
                    f"audio {audio_label(rec, src)}", style="muted")
        body.append("\n")
        body.append_text(access_word(src.access, src.access_detail, g))
        grid.add_row(Text(f"#{src.index}"), body)
    return grid


def audio_summary(rec: Recording, g: Glyphs) -> Text:
    if not rec.audio_tracks:
        return Text("no separate audio track discovered", style="muted")
    parts: list[str] = []
    for track in rec.audio_tracks:
        parts.append(_audio_track_text(track, g))
    return T("; ".join(parts))


def _audio_track_text(track: AudioTrack, g: Glyphs) -> str:
    text = f"{len(track.renditions)} rendition(s)"
    if track.language:
        text += f"{g.sep}{track.language}"
    if track.shared or len(track.used_by_sources) > 1:
        used = ", ".join(map(str, track.used_by_sources)) or "all"
        text += f"{g.sep}shared by sources {used} (downloaded once)"
    return text


def caption_line(track: CaptionTrack, g: Glyphs) -> Text:
    timing = {True: "timed", False: "untimed", None: "timing unknown"}[track.timed]
    text = Text()
    text.append(track.language or "und", style="accent")
    text.append(f"  {track.kind.value}{g.sep}{timing}")
    if track.formats:
        text.append(f"{g.sep}{', '.join(sorted(track.formats))}", style="muted")
    if track.label:
        text.append(f"{g.sep}{scrub(track.label)}", style="muted")
    if track.access not in (AccessState.AVAILABLE, AccessState.UNKNOWN):
        text.append(g.sep)
        text.append_text(access_word(track.access, None, g))
    return text


def chapters_summary(chapters: ChapterSet | None) -> Text:
    if not chapters or not chapters.chapters:
        return Text("No native chapters", style="muted")
    text = Text(f"{len(chapters.chapters)} ({chapters.provenance.value})")
    if chapters.source_detail:
        text.append(f"  {scrub(chapters.source_detail)}", style="muted")
    return text


def notes_block(rec: Recording, g: Glyphs) -> Group | None:
    if not rec.notes:
        return None
    grid = Table.grid(padding=(0, 1))
    grid.add_column(no_wrap=True)
    grid.add_column(overflow="fold")
    for note in rec.notes:
        line = Text(scrub(note.message))
        if not note.observed:
            line.append(" (unverified)", style="warn")
        grid.add_row(Text(g.bullet, style="warn" if note.level == "warning" else "muted"), line)
    return Group(Text("Discovery notes", style="heading"), grid)


def inspect_view(rec: Recording, ui: UI) -> Group:
    g = ui.glyphs
    parts: list[RenderableType] = []
    head = header_line(ui, "inspect")
    if head is not None:
        parts.append(head)
    parts.append(recording_header(rec, ui))
    parts.append(Text(""))
    if rec.sources:
        parts.append(sources_table(rec, ui))
    else:
        parts.append(Text("No video sources discovered.", style="warn"))
    parts.append(Text(""))
    caption_rows: list[RenderableType]
    if rec.captions:
        caption_rows = [caption_line(c, g) for c in rec.captions]
    else:
        caption_rows = [Text("No platform transcript found", style="muted")]
    parts.append(kv_grid([
        ("Audio", audio_summary(rec, g)),
        ("Captions", Group(*caption_rows)),
        ("Chapters", chapters_summary(rec.chapters)),
    ]))
    notes = notes_block(rec, g)
    if notes is not None:
        parts.extend([Text(""), notes])
    return Group(*parts)


# --------------------------------------------------------------------------------------
# download plan (summary panel)
# --------------------------------------------------------------------------------------


def _source_choice(rec: Recording, index: int, req: DownloadRequest, g: Glyphs) -> Text:
    src = rec.source(index)
    if src is None:
        return T(f"Source {index} (unknown)", "warn")
    text = Text(f"{index}. ", style="accent")
    text.append(scrub(src.display_name))
    try:
        rendition = select_rendition(src, req.quality)
    except QualityUnavailableError:
        raise
    except UnavailableError:
        text.append(g.sep)
        text.append_text(access_word(src.access, src.access_detail, g))
        return text
    resolution = f"{rendition.height}p" if rendition.height else "unknown resolution"
    if rendition.resolution:
        resolution += f" ({rendition.resolution})"
    text.append(f"{g.sep}{resolution}")
    codecs = [n for n in codec_names(rendition.codecs) if n in _VIDEO_CODECS]
    if codecs:
        text.append(f"{g.sep}{', '.join(codecs)}", style="muted")
    if src.access not in (AccessState.AVAILABLE, AccessState.UNKNOWN):
        text.append(g.sep)
        text.append_text(access_word(src.access, src.access_detail, g))
    return text


def _subtitle_plan(req: DownloadRequest, g: Glyphs) -> Text:
    rec = req.recording
    mode = req.subtitles
    text = Text()
    if mode.kind is SubtitleModeKind.NONE:
        text.append("none", style="muted")
    elif mode.kind is SubtitleModeKind.AUTO:
        if rec.captions:
            text.append(", ".join(
                f"{c.language or 'und'} ({provenance_word(c.kind)}"
                f"{'' if c.timed is not False else ', untimed: sidecar only'})"
                for c in rec.captions))
        else:
            text.append("none on platform", style="muted")
    else:
        items: list[str] = []
        for lang in mode.languages:
            match = next((c for c in rec.captions if language_matches(lang, c.language)), None)
            items.append(f"{lang} ({provenance_word(match.kind)})" if match
                         else f"{lang} (not found on platform)")
        text.append(", ".join(items))
    if req.transcribe.enabled:
        when = " if missing" if req.transcribe.only_if_missing else ""
        text.append(f"\n+ generated locally{when}"
                    f" ({req.transcribe.backend}, model {req.transcribe.model}),"
                    " labelled generated", style="muted")
    if req.default_subtitle:
        text.append(f"\ndefault track: {req.default_subtitle}", style="muted")
    return text


def _chapter_plan(req: DownloadRequest, g: Glyphs) -> Text:
    mode, native = req.chapters, req.recording.chapters
    native_count = len(native.chapters) if native else 0
    if mode.kind is ChapterModeKind.FILE:
        text = T(f"from {mode.file.name if mode.file else 'file'} (user file)")
        if native_count:
            text.append(f"{g.sep}overrides {native_count} native", style="warn")
        return text
    if mode.kind is ChapterModeKind.AUTO:
        if native_count:
            return T(f"{native_count} native")
        return Text("none (no native chapters)", style="muted")
    if mode.kind is ChapterModeKind.INTERVAL:
        return T(f"every {format_duration(mode.interval)} (generated, neutral titles)")
    return Text("none", style="muted")


def _files_plan(req: DownloadRequest, g: Glyphs) -> Text:
    from echo360_dl.media import naming

    ext = req.container.value
    rec = req.recording
    lines: list[str] = []
    if req.audio_only:
        lines.append(naming.media_basename(rec, None) + ".m4a")

    def name_of(index: int) -> str:
        src = rec.source(index)
        return src.display_name if src else f"Source {index}"

    composite = req.layout is not Layout.SEPARATE and not req.audio_only
    if composite:
        from echo360_dl.selection import composite_pair

        first, second = composite_pair(req.source_indexes, req.layout_order)
        arrangement = "side by side" if req.layout is Layout.SIDE_BY_SIDE else "top-bottom"
        label = f"{name_of(first)} + {name_of(second)} ({arrangement})"
        lines.append(f"{naming.media_basename(rec, label)}.{ext}")
    if not composite or req.keep_separate:
        for index in req.source_indexes:
            lines.append(f"{naming.media_basename(rec, name_of(index))}.{ext}")
    if req.write_sidecars:
        lines.append(f"sidecars: {naming.lesson_basename(rec)}.<lang>.vtt/.srt/.txt, "
                     ".chapters.json (when present)")
    lines.append(f"existing files: {req.if_exists.value}")
    return T("\n".join(lines))


def _layout_plan(req: DownloadRequest) -> Text:
    if req.layout is Layout.SEPARATE or req.audio_only:
        return T("separate files (stream copy, original quality)")
    from echo360_dl.selection import composite_pair

    first, second = composite_pair(req.source_indexes, req.layout_order)
    where = ("left | right" if req.layout is Layout.SIDE_BY_SIDE else "top / bottom")
    text = (f"{req.layout.value}: Source {first} {where.split()[0]}, Source {second} "
            f"{where.split()[-1]} · video re-encoded ({req.encoder.value})")
    if req.keep_separate:
        text += " · separate files too"
    return T(text)


def plan_panel(req: DownloadRequest, ui: UI) -> Panel:
    g = ui.glyphs
    rec = req.recording
    if req.audio_only:
        sources: RenderableType = Text("audio only (no video)")
    else:
        sources = Group(*[_source_choice(rec, i, req, g) for i in req.source_indexes])
    track = rec.audio_track(None) if len(rec.audio_tracks) == 1 else None
    if not rec.audio_tracks:
        audio = Text("embedded with video (no separate track discovered)", style="muted")
    elif track is not None and (track.shared or len(track.used_by_sources) > 1):
        audio = T("shared track, downloaded once")
    else:
        audio = T(f"{len(rec.audio_tracks)} track(s)")
    title = rec.title or rec.lesson_id
    grid = kv_grid([
        ("Lesson", T(title, "heading")),
        ("Sources", sources),
        ("Quality", str(req.quality)),
        ("Audio", audio),
        ("Subtitles", _subtitle_plan(req, g)),
        ("Chapters", _chapter_plan(req, g)),
        ("Layout", _layout_plan(req)),
        ("Container", req.container.value),
        ("Output", T(str(req.output_dir), "path")),
        ("Files", _files_plan(req, g)),
    ])
    return Panel(grid, title=Text("Download plan", style="heading"), title_align="left",
                 box=g.panel_box, border_style="muted", expand=False, padding=(0, 1))


# --------------------------------------------------------------------------------------
# download result
# --------------------------------------------------------------------------------------


def stream_summary(verify: VerifyReport | None, g: Glyphs) -> str:
    if verify is None:
        return ""
    parts: list[str] = []
    subs: list[str] = []
    for s in verify.streams:
        if s.codec_type == "video":
            dims = f" {s.width}x{s.height}" if s.width and s.height else ""
            parts.append(f"{s.codec_name or 'video'}{dims}")
        elif s.codec_type == "audio":
            parts.append(s.codec_name or "audio")
        elif s.codec_type == "subtitle":
            subs.append(s.language or "und")
    if subs:
        parts.append("subs " + ",".join(subs))
    if verify.duration is not None:
        parts.append(format_duration(verify.duration))
    if verify.chapters:
        parts.append(f"chapters {g.check} {len(verify.chapters)} embedded")
    if not verify.ok:
        parts.append("verify failed: " + "; ".join(scrub(p) for p in verify.problems))
    return g.sep.join(parts)


def _size(out: OutputFile) -> int | None:
    if out.size is not None:
        return out.size
    try:
        return out.path.stat().st_size
    except OSError:
        return None


def outputs_table(result: DownloadResult, ui: UI) -> Table:
    g = ui.glyphs
    table = Table.grid(padding=(0, 2))
    table.add_column(style="accent", no_wrap=True)
    table.add_column(overflow="fold")
    for out in result.outputs:
        line = Text()
        line.append(str(out.path), style="path")
        size = _size(out)
        details = [fmt_bytes(size)] if size is not None else []
        if out.language:
            details.append(out.language)
        if out.provenance:
            details.append(out.provenance)
        summary = stream_summary(out.verify, g)
        if summary:
            details.append(summary)
        if out.skipped_existing:
            details.append("existing file kept")
        if details:
            line.append("\n" + g.sep.join(scrub(d) for d in details), style="muted")
        kind = out.kind.value
        table.add_row(Text(kind, style="warn" if out.verify and not out.verify.ok else "accent"),
                      line)
    return table


def result_view(result: DownloadResult, ui: UI) -> Group:
    status_words = {
        ResultStatus.OK: ("OK", "Download complete"),
        ResultStatus.PARTIAL: ("PARTIAL", "Download partly complete"),
        ResultStatus.FAILED: ("FAILED", "Download failed"),
        ResultStatus.CANCELLED: ("CANCELLED", "Download cancelled"),
    }
    label, words = status_words[result.status]
    head = status_label(label)
    count = len(result.outputs)
    head.append(f" {words}{ui.glyphs.sep}{count} output{'s' if count != 1 else ''}")
    parts: list[RenderableType] = [head]
    if result.outputs:
        parts.append(outputs_table(result, ui))
    if result.failures:
        parts.append(Text("Failures", style="heading"))
        for f in result.failures:
            message = (f"{scrub(f.step)}{ui.glyphs.sep}{scrub(f.target)} {ui.glyphs.dash} "
                       f"{scrub(f.message)}")
            parts.append(status_block("FAIL", message, scrub(f.hint) if f.hint else None))
    if result.warnings:
        parts.append(Text("Warnings", style="heading"))
        for w in result.warnings:
            parts.append(status_block("WARN", scrub(w)))
    return Group(*parts)


# --------------------------------------------------------------------------------------
# auth / chapters
# --------------------------------------------------------------------------------------


def auth_status_view(status: AuthStatus, ui: UI) -> Group:
    if status.logged_in is True:
        head = status_label("OK")
        head.append(" Logged in")
    elif status.logged_in is False:
        head = status_label("WARN")
        head.append(" Not logged in")
    else:
        head = status_label("WARN")
        head.append(" Login state unknown")
    rows = kv_grid([
        ("Host", status.host),
        ("Profile", T(str(status.profile_dir) + ("" if status.profile_exists else
                                                  " (does not exist)"), "path")),
        ("Detail", status.detail),
    ])
    parts: list[RenderableType] = [head, rows]
    if status.logged_in is not True:
        parts.append(Text("Run `echo360-dl auth login` to sign in.", style="muted"))
    return Group(*parts)


def resolve_chapter_ends(chapters: Sequence[Chapter], duration: float | None
                         ) -> list[tuple[float, float | None, str]]:
    rows: list[tuple[float, float | None, str]] = []
    ordered = list(chapters)
    for i, ch in enumerate(ordered):
        end = ch.end
        if end is None:
            end = ordered[i + 1].start if i + 1 < len(ordered) else duration
        rows.append((ch.start, end, ch.title))
    return rows


def chapters_table(rows: Sequence[tuple[float, float | None, str]], ui: UI,
                   title: str | None = None) -> Table:
    table = data_table(ui, "#", "Start", "End", "Title", title=title)
    for n, (start, end, name) in enumerate(rows, 1):
        table.add_row(T(n, "muted"), T(_clock(start)),
                      T(_clock(end) if end is not None else ui.glyphs.dash), T(name))
    return table


def _clock(seconds: float) -> str:
    try:
        return format_clock(seconds, millis=bool(seconds % 1))
    except ValueError:
        return str(seconds)
