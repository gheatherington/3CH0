"""Typer application: the ``3ch0`` commands.

The CLI talks to the implementation packages only through
:func:`threech0.services.get_services`. Output rules (see ``cli/output.py``):
JSON on stdout only, everything human on stderr, exit codes from ``errors.ExitCode``.
"""

from __future__ import annotations

import contextlib
import enum
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
from rich.text import Text

from threech0 import __version__, paths
from threech0.cli import interactive, render
from threech0.cli.doctor import doctor_view, run_checks
from threech0.cli.output import (
    CANCEL_MESSAGE,
    UI,
    UIOptions,
    cancellation,
    serialise_transcript,
)
from threech0.cli.progress import Sink, make_progress
from threech0.cli.theme import BRAND, COMMAND, status_block, status_label
from threech0.errors import (
    AuthExpiredError,
    Echo360DLError,
    ExitCode,
    LanguageUnavailableError,
    LoginRequiredError,
    SelectionRequiredError,
    TranscriptUnavailableError,
    UsageError,
)
from threech0.models import (
    ChapterModeKind,
    Container,
    ContentSession,
    DiscoveryOptions,
    DownloadRequest,
    Encoder,
    Failure,
    IfExists,
    Layout,
    Recording,
    ResultStatus,
    Stage,
    SubtitleMode,
    SubtitleModeKind,
    TranscribeOptions,
    Transcript,
    TranscriptBundle,
    TranscriptProvenance,
)
from threech0.redact import public_url, scrub
from threech0.selection import (
    composite_pair,
    language_matches,
    parse_chapters,
    parse_layout_order,
    parse_quality,
    parse_streams,
    parse_subtitles,
    resolve_streams,
    select_rendition,
    skipped_by_all,
)
from threech0.services import get_services
from threech0.timecode import parse_time

LOGIN_HINT = f"Run `{COMMAND} auth login` in a terminal to sign in, then retry."

app = typer.Typer(
    name=COMMAND,
    help=f"{BRAND}: inspect and download Echo360 lecture recordings you can already view. "
         "Not affiliated with Echo360.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
auth_app = typer.Typer(help="Manage the dedicated browser login.", no_args_is_help=True)
chapters_app = typer.Typer(help="Validate chapter files and show embedded chapters.",
                           no_args_is_help=True)
app.add_typer(auth_app, name="auth")
app.add_typer(chapters_app, name="chapters")


# --------------------------------------------------------------------------------------
# Option types (shared by every subcommand)
# --------------------------------------------------------------------------------------

OUT = "Output"
ACCESS = "Echo360 access"
TRANSCRIBE = "Local transcription"


class AuthMode(enum.StrEnum):
    BROWSER = "browser"
    REQUEST_FILE = "request-file"


class TranscribeMode(enum.StrEnum):
    OFF = "off"
    LOCAL = "local"


class Backend(enum.StrEnum):
    AUTO = "auto"
    FASTER_WHISPER = "faster-whisper"
    MLX_WHISPER = "mlx-whisper"


class TranscriptFormat(enum.StrEnum):
    VTT = "vtt"
    SRT = "srt"
    TXT = "txt"
    JSON = "json"


JsonOpt = Annotated[bool, typer.Option(
    "--json", help="Write one JSON document to stdout (never prompts).", rich_help_panel=OUT)]
QuietOpt = Annotated[bool, typer.Option(
    "--quiet", "-q", help="Only errors and the final list of outputs.", rich_help_panel=OUT)]
VerboseOpt = Annotated[bool, typer.Option(
    "--verbose", "-v", help="Debug detail (secrets redacted) and tracebacks.",
    rich_help_panel=OUT)]
NoColorOpt = Annotated[bool, typer.Option(
    "--no-color", help="Disable colour (NO_COLOR is honoured too).", rich_help_panel=OUT)]
AsciiOpt = Annotated[bool, typer.Option(
    "--ascii", help="ASCII-only symbols and box drawing.", rich_help_panel=OUT)]
NonInteractiveOpt = Annotated[bool, typer.Option(
    "--non-interactive", help="Never prompt and never open a login window.",
    rich_help_panel=OUT)]
PlainOpt = Annotated[bool, typer.Option(
    "--plain", help="No live display: plain progress lines even in a terminal (nothing is "
    "redrawn).", rich_help_panel=OUT)]

AuthOpt = Annotated[AuthMode | None, typer.Option(
    "--auth", help="How to authenticate: the dedicated browser profile (default) or a saved "
    "'Copy as cURL' request file.", rich_help_panel=ACCESS, show_default=False)]
RequestFileOpt = Annotated[Path | None, typer.Option(
    "--request-file", metavar="PATH", help="Saved 'Copy as cURL' request (parsed as data, "
    "never executed). Implies --auth request-file.", rich_help_panel=ACCESS)]
TimeoutOpt = Annotated[float, typer.Option(
    "--timeout", metavar="SECONDS", min=1, help="Upper bound for discovery.",
    rich_help_panel=ACCESS)]
ProfileDirOpt = Annotated[Path | None, typer.Option(
    "--profile-dir", metavar="DIR", help="Browser profile directory (default: the app's "
    "data dir).", rich_help_panel=ACCESS, show_default=False)]

TranscribeOpt = Annotated[TranscribeMode, typer.Option(
    "--transcribe", help="Generate a transcript on this machine when the platform has "
    "none (labelled 'generated').", rich_help_panel=TRANSCRIBE)]
BackendOpt = Annotated[Backend, typer.Option(
    "--transcribe-backend", help="Speech recognition backend.", rich_help_panel=TRANSCRIBE)]
ModelOpt = Annotated[str, typer.Option(
    "--transcribe-model", metavar="NAME", help="Speech model name (e.g. small, medium).",
    rich_help_panel=TRANSCRIBE)]
AllowModelOpt = Annotated[bool, typer.Option(
    "--allow-model-download", help="Allow downloading the speech model without asking.",
    rich_help_panel=TRANSCRIBE)]


def _opts(json_: bool, quiet: bool, verbose: bool, no_color: bool, ascii_: bool,
          non_interactive: bool, plain: bool = False) -> UIOptions:
    return UIOptions(json=json_, quiet=quiet, verbose=verbose, no_color=no_color,
                     ascii=ascii_, non_interactive=non_interactive, plain=plain)


# --------------------------------------------------------------------------------------
# Execution wrapper and shared steps
# --------------------------------------------------------------------------------------


def _add_login_hint(err: Echo360DLError) -> Echo360DLError:
    if isinstance(err, LoginRequiredError | AuthExpiredError) and \
            "auth login" not in (err.hint or ""):
        err.hint = f"{err.hint} {LOGIN_HINT}" if err.hint else LOGIN_HINT
    return err


def _execute(command: str, options: UIOptions, body: Callable[[UI], int]) -> None:
    """Run ``body`` and translate every outcome into an exit code."""
    ui = UI(command, options)
    _OPEN_SESSIONS.clear()
    try:
        code = body(ui)
    except Echo360DLError as err:
        code = ui.fail(_add_login_hint(err))
    except KeyboardInterrupt:
        code = ui.cancelled()
    except typer.Exit:
        raise
    except Exception as exc:
        code = ui.unexpected(exc)
    finally:
        _close_sessions()
    raise typer.Exit(code)


#: Sessions opened by this command; closed when the command finishes, however it ends.
_OPEN_SESSIONS: list[ContentSession] = []


def _close_sessions() -> None:
    while _OPEN_SESSIONS:
        close = getattr(_OPEN_SESSIONS.pop(), "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()


def _discovery_options(ui: UI, *, auth: AuthMode | None, request_file: Path | None,
                       timeout: float, profile_dir: Path | None,
                       save_trace: Path | None = None) -> DiscoveryOptions:
    mode = auth or (AuthMode.REQUEST_FILE if request_file else AuthMode.BROWSER)
    if mode is AuthMode.REQUEST_FILE:
        if request_file is None:
            raise UsageError("--auth request-file needs --request-file PATH.",
                             hint="Save the stream request from the browser's developer tools "
                                  "with 'Copy as cURL', then pass that file.")
        request_file = request_file.expanduser()
        if not request_file.is_file():
            raise UsageError(f"Request file not found: {request_file}")
    elif request_file is not None:
        raise UsageError("--request-file cannot be combined with --auth browser.")
    return DiscoveryOptions(
        auth_mode=mode.value,  # type: ignore[arg-type]
        request_file=request_file,
        interactive=ui.interactive,
        timeout=timeout,
        profile_dir=profile_dir.expanduser() if profile_dir else None,
        save_trace=save_trace.expanduser() if save_trace else None,
    )


def _discover(ui: UI, sink: Sink, url: str,
              options: DiscoveryOptions) -> tuple[Recording, ContentSession]:
    services = get_services()
    ref = services.parse_url(url)
    sink.stage(Stage.DISCOVER, public_url(ref.url))
    ui.debug(f"auth={options.auth_mode} interactive={options.interactive} "
             f"timeout={options.timeout:g}s lesson={ref.lesson_id}")
    recording, session = services.discover(ref, options, sink)
    _OPEN_SESSIONS.append(session)
    return recording, session


def _display_url(url: str) -> str:
    """The classroom URL without its query (signed parameters live there)."""
    try:
        return public_url(get_services().parse_url(url).url)
    except Echo360DLError:
        return public_url(url)


# --------------------------------------------------------------------------------------
# Root
# --------------------------------------------------------------------------------------


def _version(value: bool) -> None:
    if value:
        sys.stdout.write(f"{COMMAND} {__version__}\n")
        raise typer.Exit(0)


@app.callback()
def root(
    version: Annotated[bool, typer.Option(
        "--version", callback=_version, is_eager=True, help="Show the version and exit.")]
    = False,
) -> None:
    """3CH0: inspect and download Echo360 lecture recordings you can already view."""


# --------------------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------------------


@app.command()
def doctor(
    output: Annotated[Path | None, typer.Option(
        "--output", "-o", metavar="DIR", help="Output directory to check (default: "
        "Downloads).", show_default=False)] = None,
    profile_dir: ProfileDirOpt = None,
    json_: JsonOpt = False, quiet: QuietOpt = False, verbose: VerboseOpt = False,
    no_color: NoColorOpt = False, ascii_: AsciiOpt = False,
    non_interactive: NonInteractiveOpt = False, plain: PlainOpt = False,
) -> None:
    """Check Python, FFmpeg, yt-dlp, the browser, folders and transcription backends."""

    def body(ui: UI) -> int:
        checks = run_checks(
            get_services(),
            output_dir=output.expanduser() if output else paths.default_output_dir(),
            profile_dir=profile_dir.expanduser() if profile_dir else paths.browser_profile_dir())
        failed = any(c.status == "FAIL" for c in checks)
        if ui.json:
            ui.emit_json(not failed, checks=[c.to_dict() for c in checks])
        elif ui.quiet:
            for c in checks:
                if c.status == "FAIL":
                    ui.always(Text(f"[FAIL] {c.name}: {c.detail}" +
                                   (f" ({c.hint})" if c.hint else "")))
        else:
            ui.show(doctor_view(checks, ui))
        return int(ExitCode.DEPENDENCY) if failed else 0

    _execute("doctor",
             _opts(json_, quiet, verbose, no_color, ascii_, non_interactive, plain), body)


# --------------------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------------------


def _profile(profile_dir: Path | None) -> Path:
    return profile_dir.expanduser() if profile_dir else paths.browser_profile_dir()


@auth_app.command("login")
def auth_login(
    timeout: Annotated[float, typer.Option(
        "--timeout", metavar="SECONDS", min=10, help="How long to wait for you to sign in.")]
    = 300.0,
    profile_dir: ProfileDirOpt = None,
    json_: JsonOpt = False, quiet: QuietOpt = False, verbose: VerboseOpt = False,
    no_color: NoColorOpt = False, ascii_: AsciiOpt = False,
    non_interactive: NonInteractiveOpt = False, plain: PlainOpt = False,
) -> None:
    """Open the dedicated browser window and sign in to Echo360 (interactive only)."""

    def body(ui: UI) -> int:
        if not ui.interactive:
            raise UsageError("auth login opens a browser window and needs an interactive "
                             "terminal.",
                             hint="Run it in a terminal without --json or --non-interactive.")
        profile = _profile(profile_dir)
        ui.show(Text(f"Opening the dedicated browser profile at {profile}. Sign in, then "
                     "close the window or wait for it to finish.", style="muted"))
        with make_progress(ui) as sink:
            status = get_services().login(profile, start_url=None, timeout=timeout,
                                          progress=sink)
        ui.always(render.auth_status_view(status, ui))
        return 0 if status.logged_in else int(ExitCode.AUTH)

    _execute("auth login",
             _opts(json_, quiet, verbose, no_color, ascii_, non_interactive, plain), body)


@auth_app.command("status")
def auth_status(
    timeout: Annotated[float, typer.Option(
        "--timeout", metavar="SECONDS", min=1, help="Upper bound for the check.")] = 30.0,
    profile_dir: ProfileDirOpt = None,
    json_: JsonOpt = False, quiet: QuietOpt = False, verbose: VerboseOpt = False,
    no_color: NoColorOpt = False, ascii_: AsciiOpt = False,
    non_interactive: NonInteractiveOpt = False, plain: PlainOpt = False,
) -> None:
    """Report whether the saved browser login still works (exit 3 when signed out)."""

    def body(ui: UI) -> int:
        status = get_services().auth_status(_profile(profile_dir), timeout=timeout)
        code = int(ExitCode.AUTH) if status.logged_in is False else 0
        if ui.json:
            ui.emit_json(code == 0, status=status)
        else:
            ui.always(render.auth_status_view(status, ui))
        return code

    _execute("auth status",
             _opts(json_, quiet, verbose, no_color, ascii_, non_interactive, plain), body)


def _is_profile_dir(path: Path) -> bool:
    """Guard against deleting something that is not a browser profile."""
    resolved = path.resolve()
    dangerous = {Path.home().resolve(), Path("/").resolve(), Path.cwd().resolve()}
    if resolved in dangerous or resolved in Path.cwd().resolve().parents:
        return False
    # Only directories this tool created (marker file) or the default location: Chrome's
    # own "Local State"/"Preferences" files would also match a real browser profile.
    return paths.is_own_profile(resolved)


@auth_app.command("logout")
def auth_logout(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")]
    = False,
    profile_dir: ProfileDirOpt = None,
    json_: JsonOpt = False, quiet: QuietOpt = False, verbose: VerboseOpt = False,
    no_color: NoColorOpt = False, ascii_: AsciiOpt = False,
    non_interactive: NonInteractiveOpt = False, plain: PlainOpt = False,
) -> None:
    """Delete the dedicated browser profile (and the Echo360 login saved in it)."""

    def body(ui: UI) -> int:
        profile = _profile(profile_dir)
        if not profile.exists():
            if ui.json:
                ui.emit_json(True, removed=False, profile_dir=profile)
            else:
                ui.show(Text(f"No browser profile at {profile}; nothing to remove.",
                             style="muted"))
            return 0
        if not profile.is_dir() or not _is_profile_dir(profile):
            raise UsageError(f"{profile} does not look like a {COMMAND} browser profile; "
                             "refusing to delete it.")
        if not yes:
            if not ui.interactive:
                raise SelectionRequiredError("Deleting the browser profile needs confirmation.",
                                             hint="Add --yes.")
            if not interactive.confirm_logout(ui, profile):
                ui.show(Text("Kept the browser profile.", style="muted"))
                return 0
        shutil.rmtree(profile)
        if ui.json:
            ui.emit_json(True, removed=True, profile_dir=profile)
        else:
            line = status_label("OK")
            line.append(f" Removed browser profile {profile}")
            ui.always(line)
        return 0

    _execute("auth logout",
             _opts(json_, quiet, verbose, no_color, ascii_, non_interactive, plain), body)


# --------------------------------------------------------------------------------------
# inspect
# --------------------------------------------------------------------------------------


@app.command()
def inspect(
    url: Annotated[str, typer.Argument(metavar="URL", help="Echo360 classroom URL.")],
    save_trace: Annotated[Path | None, typer.Option(
        "--save-trace", metavar="FILE", help="Write a sanitised discovery trace (no secrets).")]
    = None,
    auth: AuthOpt = None, request_file: RequestFileOpt = None, timeout: TimeoutOpt = 90.0,
    profile_dir: ProfileDirOpt = None,
    json_: JsonOpt = False, quiet: QuietOpt = False, verbose: VerboseOpt = False,
    no_color: NoColorOpt = False, ascii_: AsciiOpt = False,
    non_interactive: NonInteractiveOpt = False, plain: PlainOpt = False,
) -> None:
    """Show sources, qualities, audio, captions and chapters for a lesson."""

    def body(ui: UI) -> int:
        options = _discovery_options(ui, auth=auth, request_file=request_file, timeout=timeout,
                                     profile_dir=profile_dir, save_trace=save_trace)
        with make_progress(ui) as sink:
            recording, _session = _discover(ui, sink, url, options)
        if ui.json:
            ui.emit_json(True, recording=recording)
        else:
            ui.always(render.inspect_view(recording, ui))
            # (discovery itself reports where the sanitised trace was written)
        return 0

    _execute("inspect",
             _opts(json_, quiet, verbose, no_color, ascii_, non_interactive, plain), body)


# --------------------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------------------


def _check_quality(recording: Recording, indexes: tuple[int, ...], request: DownloadRequest
                   ) -> None:
    """Fail fast (exit 4) when the requested quality cannot be met."""
    for index in indexes:
        src = recording.source(index)
        if src is not None and src.renditions:
            select_rendition(src, request.quality)


def _default_language(value: str | None) -> str | None:
    if value is None:
        return None
    mode = parse_subtitles(value)
    if mode.kind is not SubtitleModeKind.LANGUAGES or len(mode.languages) != 1:
        raise UsageError(f"Invalid --default-subtitle value {value!r}.",
                         hint="Use one language code such as 'en'.")
    return mode.languages[0]


_MEDIA_EXTENSIONS = (".mp4", ".mkv", ".m4a")


def _output_name(value: str | None) -> str | None:
    """Validate --name: a bare file name; a typed media extension is dropped (the
    container decides it)."""
    if value is None:
        return None
    from threech0.media import naming

    text = value.strip()
    if "/" in text or "\\" in text:
        raise UsageError(f"--name {value!r} must be a file name, not a path.",
                         hint="Choose the folder with --output DIR.")
    if text.lower().endswith(_MEDIA_EXTENSIONS):
        text = text[:-4]
    cleaned = naming.safe_component(text, fallback="")
    if not cleaned:
        raise UsageError(f"--name {value!r} leaves no usable file name.")
    return text


@app.command()
def download(
    url: Annotated[str, typer.Argument(metavar="URL", help="Echo360 classroom URL.")],
    streams: Annotated[str | None, typer.Option(
        "--streams", metavar="all|audio|1,2", help="Sources to download. Prompted when "
        "interactive; required otherwise.", show_default=False)] = None,
    quality: Annotated[str, typer.Option(
        "--quality", metavar="best|worst|720p", help="Video quality per source.")] = "best",
    container: Annotated[Container, typer.Option("--container", help="Output container.")]
    = Container.MP4,
    subtitles: Annotated[str, typer.Option(
        "--subtitles", metavar="auto|none|en,fr", help="Platform subtitles to embed.")]
    = "auto",
    default_subtitle: Annotated[str | None, typer.Option(
        "--default-subtitle", metavar="LANG", help="Subtitle language marked default.",
        show_default=False)] = None,
    chapters: Annotated[str, typer.Option(
        "--chapters", metavar="auto|none|interval", help="auto = native chapters only "
        "(never invented).")] = "auto",
    chapters_file: Annotated[Path | None, typer.Option(
        "--chapters-file", metavar="FILE", help="Chapter JSON (overrides native chapters).",
        show_default=False)] = None,
    chapter_interval: Annotated[str | None, typer.Option(
        "--chapter-interval", metavar="10m", help="Interval for --chapters interval "
        "(default 10m).", show_default=False)] = None,
    transcribe: TranscribeOpt = TranscribeMode.OFF,
    transcribe_backend: BackendOpt = Backend.AUTO,
    transcribe_model: ModelOpt = "small",
    allow_model_download: AllowModelOpt = False,
    output: Annotated[Path | None, typer.Option(
        "--output", "-o", metavar="DIR", help="Output directory (default: Downloads).",
        show_default=False)] = None,
    name: Annotated[str | None, typer.Option(
        "--name", "-n", metavar="NAME", help="Output file name (no extension) instead of "
        "'<course> - <date> - <title> [<id>]'. Several media files get ' - <source>' "
        "appended; sidecars use NAME too.", show_default=False)] = None,
    if_exists: Annotated[IfExists, typer.Option(
        "--if-exists", help="What to do when an output file already exists.")]
    = IfExists.SKIP,
    no_sidecars: Annotated[bool, typer.Option(
        "--no-sidecars", help="Do not write subtitle/transcript/chapter files next to the "
        "media.")] = False,
    keep_work_dir: Annotated[bool, typer.Option(
        "--keep-work-dir", help="Keep intermediate files for troubleshooting.")] = False,
    layout: Annotated[Layout | None, typer.Option(
        "--layout", help="separate = one file per source (original quality, default); "
        "side-by-side / top-bottom = both sources in one re-encoded video. Prompted when "
        "two sources are picked interactively.", show_default=False,
        rich_help_panel="Layout")] = None,
    layout_order: Annotated[str | None, typer.Option(
        "--layout-order", metavar="1,2", help="Which source goes left/top first, e.g. 2,1.",
        show_default=False, rich_help_panel="Layout")] = None,
    keep_separate: Annotated[bool, typer.Option(
        "--keep-separate", help="With a combined layout, also write each source as its own "
        "file.", rich_help_panel="Layout")] = False,
    encoder: Annotated[Encoder, typer.Option(
        "--encoder", help="Encoder for combined layouts: auto = Apple VideoToolbox "
        "(hardware) when available, else x264.", rich_help_panel="Layout")] = Encoder.AUTO,
    retries: Annotated[int, typer.Option("--retries", min=0, max=10,
                                         help="Retries per network step.")] = 3,
    concurrency: Annotated[int, typer.Option("--concurrency", min=1, max=8,
                                             help="Parallel downloads.")] = 2,
    auth: AuthOpt = None, request_file: RequestFileOpt = None, timeout: TimeoutOpt = 90.0,
    profile_dir: ProfileDirOpt = None,
    json_: JsonOpt = False, quiet: QuietOpt = False, verbose: VerboseOpt = False,
    no_color: NoColorOpt = False, ascii_: AsciiOpt = False,
    non_interactive: NonInteractiveOpt = False, plain: PlainOpt = False,
) -> None:
    """Download sources with embedded subtitles and chapters, then verify them."""

    def body(ui: UI) -> int:
        services = get_services()
        # Parse everything before touching the network (usage errors exit 2 fast).
        selection = parse_streams(streams) if streams is not None else None
        if selection is None and not ui.interactive:
            raise SelectionRequiredError("Choose which sources to download.",
                                         hint=interactive.STREAMS_HINT)
        quality_spec = parse_quality(quality)
        order = parse_layout_order(layout_order)
        subtitle_mode = parse_subtitles(subtitles)
        if chapter_interval is not None and chapters.strip().lower() != "interval":
            raise UsageError("--chapter-interval only applies with --chapters interval.")
        chapter_mode = parse_chapters(chapters, file=str(chapters_file) if chapters_file
                                      else None, interval=chapter_interval)
        if chapter_mode.kind is ChapterModeKind.FILE and chapter_mode.file and \
                not chapter_mode.file.is_file():
            raise UsageError(f"Chapters file not found: {chapter_mode.file}")
        default_lang = _default_language(default_subtitle)
        output_name = _output_name(name)
        transcribe_opts = TranscribeOptions(
            enabled=transcribe is TranscribeMode.LOCAL, backend=transcribe_backend.value,
            model=transcribe_model, allow_model_download=allow_model_download)
        out_dir = output.expanduser() if output else paths.default_output_dir()
        options = _discovery_options(ui, auth=auth, request_file=request_file, timeout=timeout,
                                     profile_dir=profile_dir)

        with make_progress(ui) as sink:
            sink.set_context(source=_display_url(url))
            recording, session = _discover(ui, sink, url, options)
            sink.set_context(title=recording.title or recording.lesson_id)
            sink.set_streams(render.stream_details(recording, ui.glyphs))
            sink.stage(Stage.SELECT)
            if selection is None:
                with sink.paused():
                    indexes, audio_only = interactive.choose_sources(ui, recording)
            else:
                audio_only = selection.audio_only
                indexes = resolve_streams(selection, recording)
                if selection.all_sources:
                    for skipped in skipped_by_all(recording):
                        sink.warn(f"Skipping {skipped.display_name}: {skipped.access.value} "
                                  "(a 403 does not prove it is absent; request it with "
                                  f"--streams {skipped.index} to try anyway).")
            chosen_layout = layout
            if chosen_layout is None:
                chosen_layout = Layout.SEPARATE
                if selection is None and not audio_only and len(indexes) == 2 \
                        and not order:
                    with sink.paused():
                        chosen_layout = interactive.choose_layout(ui, recording, indexes)
            if chosen_layout is not Layout.SEPARATE:
                if audio_only:
                    raise UsageError("--layout needs video sources; it cannot be combined "
                                     "with --streams audio.")
                composite_pair(indexes, order)  # validates before anything is downloaded
            elif order or keep_separate:
                raise UsageError("--layout-order and --keep-separate only apply with "
                                 "--layout side-by-side or --layout top-bottom.")
            request = DownloadRequest(
                recording=recording, output_dir=out_dir, source_indexes=indexes,
                audio_only=audio_only, quality=quality_spec, container=container,
                subtitles=subtitle_mode, chapters=chapter_mode, transcribe=transcribe_opts,
                if_exists=if_exists, write_sidecars=not no_sidecars,
                default_subtitle=default_lang, retries=retries, concurrency=concurrency,
                keep_work_dir=keep_work_dir, layout=chosen_layout, layout_order=order,
                keep_separate=keep_separate, encoder=encoder, name=output_name)
            _check_quality(recording, indexes, request)
            sink.set_context(target=str(out_dir))
            with sink.paused():
                ui.show(render.plan_view(request, ui))
                if selection is None and not interactive.confirm_plan(ui):
                    ui.show(Text("Nothing downloaded.", style="muted"))
                    return 0
            confirm = interactive.model_download_confirmer(
                ui, allow=allow_model_download, pause=sink.paused)
            with cancellation(ui.stderr, notify=sink.notice) as cancel:
                result = services.download(request, session, progress=sink,
                                           confirm_download=confirm, cancel=cancel)
            # Warnings the live view showed but has not printed go into the summary.
            take = getattr(sink, "take_warnings", None)
            pending = take() if callable(take) and not ui.json and not ui.quiet else []
        if ui.json:
            ui.emit_json(result.status is ResultStatus.OK, result=result)
        elif ui.quiet:
            ui.paths([o.path for o in result.outputs])
            for failure in result.failures:
                ui.always(Text(f"[FAIL] {scrub(failure.target)}: {scrub(failure.message)}"))
        else:
            ui.err.print(render.result_view(result, ui, sink.telemetry.snapshot(), pending))
        if result.status is ResultStatus.CANCELLED and not ui.json:
            ui.always(Text(CANCEL_MESSAGE if not ui.ascii else
                           CANCEL_MESSAGE.replace("—", "-"), style="warn"))
        return result.exit_code

    _execute("download",
             _opts(json_, quiet, verbose, no_color, ascii_, non_interactive, plain), body)


# --------------------------------------------------------------------------------------
# transcript
# --------------------------------------------------------------------------------------

_PROVENANCE_RANK = {
    TranscriptProvenance.PLATFORM_CAPTIONS: 0,
    TranscriptProvenance.PLATFORM_TRANSCRIPT: 1,
    TranscriptProvenance.USER_FILE: 2,
    TranscriptProvenance.GENERATED_LOCAL: 3,
}


def _fetch_failure_error(bundle: TranscriptBundle, language: str | None) -> Echo360DLError | None:
    """A platform track that exists but could not be fetched is reported as such (with
    its own kind, e.g. forbidden -> exit 3), never as "no transcript"."""
    def track_language(failure: Failure) -> str:
        return failure.target.split(":", 1)[-1]  # targets look like "captions:<lang|id>"

    relevant = [f for f in bundle.failed
                if language is None or language_matches(language, track_language(f))]
    if not relevant:
        return None
    first = relevant[0]
    by_kind = {cls.kind: cls for cls in _error_classes(Echo360DLError)}
    cls = by_kind.get(first.kind, Echo360DLError)
    return cls(first.message, hint=first.hint)


def _error_classes(base: type[Echo360DLError]) -> list[type[Echo360DLError]]:
    out = [base]
    for sub in base.__subclasses__():
        out.extend(_error_classes(sub))
    return out


def pick_transcript(bundle: TranscriptBundle, language: str | None, fmt: str,
                    transcribe_enabled: bool) -> Transcript:
    candidates = list(bundle.transcripts)
    if not candidates or (language and not any(language_matches(language, t.language)
                                               for t in candidates)):
        failure = _fetch_failure_error(bundle, language)
        if failure is not None:
            raise failure
    if language:
        candidates = [t for t in candidates if language_matches(language, t.language)]
        if not candidates:
            found = sorted({t.language or "und" for t in bundle.transcripts})
            hint = f"Available: {', '.join(found)}." if found else None
            if not transcribe_enabled:
                hint = ((hint + " ") if hint else "") + \
                    "Add --transcribe local to generate one on this machine."
            raise LanguageUnavailableError(f"No transcript in language {language!r}.",
                                           hint=hint)
    if not candidates:
        raise TranscriptUnavailableError(
            "No platform transcript found for this lesson.",
            hint=None if transcribe_enabled else
            "Add --transcribe local to generate one on this machine.")
    needs_timing = fmt in {"vtt", "srt"}
    candidates.sort(key=lambda t: (needs_timing and not t.timed,
                                   _PROVENANCE_RANK.get(t.provenance, 9)))
    return candidates[0]


def _transcript_filename(recording: Recording, transcript: Transcript, fmt: str) -> str:
    from threech0.media import naming

    # Same naming as download sidecars: "<lesson basename>.<lang>[.generated].<fmt>".
    parts = [naming.language_tag(transcript.language)]
    if transcript.generated:
        parts.append("generated")
    return naming.with_suffix(naming.lesson_basename(recording), *parts, fmt)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@app.command()
def transcript(
    url: Annotated[str, typer.Argument(metavar="URL", help="Echo360 classroom URL.")],
    language: Annotated[str | None, typer.Option(
        "--language", metavar="LANG", help="Transcript language (default: first available).",
        show_default=False)] = None,
    fmt: Annotated[TranscriptFormat, typer.Option(
        "--format", help="vtt/srt need timed cues; txt/json work for any transcript.")]
    = TranscriptFormat.VTT,
    output: Annotated[str | None, typer.Option(
        "--output", "-o", metavar="PATH|-", help="File or directory to write, or '-' for "
        "stdout (default: Downloads).", show_default=False)] = None,
    transcribe: TranscribeOpt = TranscribeMode.OFF,
    transcribe_backend: BackendOpt = Backend.AUTO,
    transcribe_model: ModelOpt = "small",
    allow_model_download: AllowModelOpt = False,
    auth: AuthOpt = None, request_file: RequestFileOpt = None, timeout: TimeoutOpt = 90.0,
    profile_dir: ProfileDirOpt = None,
    json_: JsonOpt = False, quiet: QuietOpt = False, verbose: VerboseOpt = False,
    no_color: NoColorOpt = False, ascii_: AsciiOpt = False,
    non_interactive: NonInteractiveOpt = False, plain: PlainOpt = False,
) -> None:
    """Save the platform transcript/captions, or generate one locally (--transcribe local)."""

    def body(ui: UI) -> int:
        services = get_services()
        if output == "-" and ui.json:
            raise UsageError("--output - writes to stdout, which --json also uses.",
                             hint="Give a file path with --output, or drop --json.")
        if language is not None:
            subtitle_mode = parse_subtitles(language)
            if subtitle_mode.kind is not SubtitleModeKind.LANGUAGES or \
                    len(subtitle_mode.languages) != 1:
                raise UsageError(f"Invalid --language value {language!r}.",
                                 hint="Use one language code such as 'en'.")
        else:
            subtitle_mode = SubtitleMode(SubtitleModeKind.AUTO)
        enabled = transcribe is TranscribeMode.LOCAL
        transcribe_opts = TranscribeOptions(
            enabled=enabled, backend=transcribe_backend.value, model=transcribe_model,
            language=language, allow_model_download=allow_model_download)
        options = _discovery_options(ui, auth=auth, request_file=request_file, timeout=timeout,
                                     profile_dir=profile_dir)

        with make_progress(ui) as sink, tempfile.TemporaryDirectory(
                prefix="3ch0-") as work_dir:
            recording, session = _discover(ui, sink, url, options)
            sink.stage(Stage.SUBTITLES)
            confirm = interactive.model_download_confirmer(
                ui, allow=allow_model_download, pause=sink.paused)
            with cancellation(ui.stderr, notify=sink.notice) as cancel:
                bundle = services.transcripts(
                    recording, session, subtitles=subtitle_mode, transcribe=transcribe_opts,
                    work_dir=Path(work_dir), progress=sink, confirm_download=confirm,
                    cancel=cancel)
        if cancel.is_set():
            return ui.cancelled()
        for note in bundle.notes:
            ui.debug(note)
        chosen = pick_transcript(bundle, language, fmt.value, enabled)
        text = serialise_transcript(chosen, fmt.value)
        others = sorted({t.language or "und" for t in bundle.transcripts} -
                        {chosen.language or "und"})
        if language is None and others:
            ui.warn(f"Also available: {', '.join(others)}. Use --language to choose.")

        if output == "-":
            ui.stdout.write(text)
            ui.stdout.flush()
            target: Path | None = None
        else:
            name = _transcript_filename(recording, chosen, fmt.value)
            target = Path(output).expanduser() if output else paths.default_output_dir() / name
            if target.is_dir():
                target = target / name
            _write_atomic(target, text)
        summary = {
            "path": target, "format": fmt.value, "language": chosen.language,
            "provenance": chosen.provenance, "generated": chosen.generated,
            "timed": chosen.timed, "cues": len(chosen.cues), "generator": chosen.generator,
        }
        if ui.json:
            ui.emit_json(True, transcript=summary, missing_languages=bundle.missing_languages,
                         notes=bundle.notes)
            return 0
        line = status_label("OK")
        kind = render.provenance_word(chosen.provenance)
        line.append(f" {chosen.language or 'und'} {kind} ({fmt.value}, {len(chosen.cues)} "
                    f"cues{'' if chosen.timed else ', untimed'})")
        if chosen.generated and chosen.generator:
            line.append(f" by {scrub(chosen.generator)}", style="muted")
        if target is not None:
            line.append(f" {ui.glyphs.arrow} ")
            line.append(str(target), style="path")
        if ui.quiet and target is not None:
            ui.paths([target])
        else:
            ui.show(line)
        return 0

    _execute("transcript",
             _opts(json_, quiet, verbose, no_color, ascii_, non_interactive, plain), body)


# --------------------------------------------------------------------------------------
# chapters
# --------------------------------------------------------------------------------------


def _existing_file(path: Path, what: str) -> Path:
    path = path.expanduser()
    if not path.is_file():
        raise UsageError(f"{what} not found: {path}")
    return path


@chapters_app.command("validate")
def chapters_validate(
    file: Annotated[Path, typer.Argument(metavar="FILE", help="Chapter JSON file.")],
    duration: Annotated[str | None, typer.Option(
        "--duration", metavar="1h2m", help="Media duration to validate against.",
        show_default=False)] = None,
    media: Annotated[Path | None, typer.Option(
        "--media", metavar="FILE", help="Read the duration from this media file (ffprobe).",
        show_default=False)] = None,
    json_: JsonOpt = False, quiet: QuietOpt = False, verbose: VerboseOpt = False,
    no_color: NoColorOpt = False, ascii_: AsciiOpt = False,
    non_interactive: NonInteractiveOpt = False, plain: PlainOpt = False,
) -> None:
    """Check a chapter file and print the resolved chapters (exit 8 on problems)."""

    def body(ui: UI) -> int:
        services = get_services()
        if duration is not None and media is not None:
            raise UsageError("Use either --duration or --media, not both.")
        chapter_file = _existing_file(file, "Chapter file")
        total: float | None = None
        if duration is not None:
            try:
                total = parse_time(duration)
            except ValueError as exc:
                raise UsageError(f"Invalid --duration: {exc}") from None
        elif media is not None:
            total = services.probe(_existing_file(media, "Media file")).duration
        chapter_set = services.load_chapters(chapter_file)
        problems = services.validate_chapters(chapter_set, total)
        rows = render.resolve_chapter_ends(chapter_set.chapters, total)
        if ui.json:
            ui.emit_json(not problems, file=chapter_file, duration=total,
                         provenance=chapter_set.provenance,
                         chapters=[{"start": s, "end": e, "title": t} for s, e, t in rows],
                         problems=problems)
        else:
            if rows:
                ui.show(render.chapters_table(rows, ui))
            for problem in problems:
                ui.always(status_block("FAIL", scrub(problem)))
            if not problems:
                line = status_label("OK")
                line.append(f" {len(rows)} chapter(s) valid"
                            + ("" if total is not None else " (duration not checked)"))
                ui.show(line)
        return int(ExitCode.VERIFY) if problems else 0

    _execute("chapters validate",
             _opts(json_, quiet, verbose, no_color, ascii_, non_interactive, plain), body)


@chapters_app.command("show")
def chapters_show(
    media_file: Annotated[Path, typer.Argument(metavar="MEDIA_FILE",
                                               help="Media file to read.")],
    json_: JsonOpt = False, quiet: QuietOpt = False, verbose: VerboseOpt = False,
    no_color: NoColorOpt = False, ascii_: AsciiOpt = False,
    non_interactive: NonInteractiveOpt = False, plain: PlainOpt = False,
) -> None:
    """Show the chapters embedded in a media file, as ffprobe sees them."""

    def body(ui: UI) -> int:
        report = get_services().probe(_existing_file(media_file, "Media file"))
        rows = render.resolve_chapter_ends(report.chapters, report.duration)
        if ui.json:
            ui.emit_json(True, file=media_file, duration=report.duration,
                         chapters=[{"start": s, "end": e, "title": t} for s, e, t in rows])
        elif rows:
            ui.always(render.chapters_table(rows, ui))
        else:
            ui.always(Text("No embedded chapters.", style="muted"))
        return 0

    _execute("chapters show",
             _opts(json_, quiet, verbose, no_color, ascii_, non_interactive, plain), body)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def _json_usage_error(command: str, message: str) -> None:
    """Option-parsing errors in --json mode still produce the JSON error document."""
    doc = {"schema_version": 1, "command": command, "ok": False,
           "error": {"kind": "usage", "message": scrub(message), "hint":
                     "Run with --help to see the accepted values.",
                     "exit_code": int(ExitCode.USAGE)}}
    sys.stdout.write(json.dumps(doc) + "\n")
    sys.stderr.write(f"error: {scrub(message)}\n")


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if "--json" in args and not {"-h", "--help", "--version"} & set(args):
            try:
                code = app(args=args, prog_name=COMMAND, standalone_mode=False)
            except typer.Abort:
                raise KeyboardInterrupt from None
            except Exception as exc:
                # Click-style usage errors (Typer vendors click, so match by shape).
                if getattr(exc, "exit_code", None) != 2 or not hasattr(exc, "format_message"):
                    raise
                ctx = getattr(exc, "ctx", None)
                path = getattr(ctx, "command_path", "") or ""
                command = path.removeprefix(COMMAND).strip() or COMMAND
                _json_usage_error(command, exc.format_message())
                sys.exit(int(ExitCode.USAGE))
            sys.exit(code if isinstance(code, int) else 0)
        app(args=args, prog_name=COMMAND)
    except KeyboardInterrupt:
        message = CANCEL_MESSAGE
        if "--ascii" in (argv if argv is not None else sys.argv[1:]):
            message = message.replace("—", "-")
        sys.stderr.write(message + "\n")
        sys.exit(int(ExitCode.CANCELLED))
