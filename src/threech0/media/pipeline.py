"""Download pipeline: select → download → subtitles → chapters → mux/verify → sidecars.

Behaviour summary (see also ``media.download`` and ``media.mux``):

* Video sources: the **video-only** media playlist of the selected rendition of each
  selected source is downloaded. Audio renditions are shared between sources
  (observed: ``s1_av`` and ``s2_av`` both use ``s0q*``); each distinct audio rendition
  is downloaded **once** and muxed into every output that uses it.
* Work dir ``<output_dir>/.3ch0-<short_id>/`` holds components and a manifest of
  completed ones. Reruns reuse completed components and resume partial ones. It is
  removed after the media outputs all succeed (unless ``keep_work_dir``) and kept
  after a download/mux/verify failure or cancellation, with a hint to rerun. Failures
  a rerun cannot fix (a missing caption language, a transcription error, a bad
  chapter file) do not keep it.
* Auth: a 403/401 triggers at most one ``session.refresh()`` per run (shared by all
  workers); the component is then retried. A refresh error fails the component with
  that error's kind; a 403 that persists after a refresh is ``unavailable``.
* Transcripts: AUTO fetches every platform caption/transcript track. Timed tracks are
  embedded and written as sidecars; untimed ones are written as ``.txt`` only.
  LANGUAGES with a missing language records ``Failure(kind="language-unavailable")``
  while the media still downloads, so the result is PARTIAL. Local transcription runs
  at most **once** per run, from the shared audio, and is reused by every output.
* Status: OK (no failures), PARTIAL (some outputs exist plus failures), FAILED
  (failures, no outputs) or CANCELLED. ``outputs`` lists only files that exist.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import json
import os
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from threech0.errors import (
    AuthError,
    CancelledError,
    DependencyError,
    Echo360DLError,
    LanguageUnavailableError,
    UnavailableError,
    UsageError,
    ValidationError,
    VerificationError,
)
from threech0.media import naming
from threech0.media.chapters import (
    chapters_to_json,
    choose_chapters,
    load_chapter_file,
    resolve_chapters,
    validate_chapters,
)
from threech0.media.download import (
    AuthRefresher,
    Component,
    DownloadContext,
    WorkDir,
    _RefreshExhausted,
    component_key,
    component_task_id,
    download_component,
    prefetch_size,
)
from threech0.media.ffmpeg import require_tools
from threech0.media.mux import (
    TEMP_MARK,
    CompositeSpec,
    ContainerKind,
    InputInfo,
    MuxJob,
    SubtitleInput,
    encode_composite,
    finalize,
    keep_unverified,
    measure,
    mux,
    remove_stale_temps,
    resolve_encoder,
    verify,
)
from threech0.media.subtitles import (
    language_name,
    loses_information,
    normalize_cues,
    to_srt,
    to_txt,
    to_vtt,
    track_title,
    validate_cues,
)
from threech0.models import (
    AudioTrack,
    CaptionTrack,
    ChapterModeKind,
    ChapterProvenance,
    ChapterSet,
    Container,
    ContentSession,
    DownloadRequest,
    DownloadResult,
    Failure,
    IfExists,
    Layout,
    OutputFile,
    OutputKind,
    ProgressSink,
    QualitySpec,
    Recording,
    ResultStatus,
    Stage,
    SubtitleMode,
    SubtitleModeKind,
    TranscribeOptions,
    Transcriber,
    Transcript,
    TranscriptBundle,
    TranscriptProvenance,
    VideoRendition,
)
from threech0.redact import scrub
from threech0.selection import (
    composite_pair,
    iso639_2,
    language_matches,
    select_audio,
    select_rendition,
)

TranscriberFactory = Callable[[TranscribeOptions], Transcriber]

#: Platform formats in order of preference (timed first).
FORMAT_PREFERENCE = ("vtt", "srt", "json", "txt")
_SIDECAR_KIND = {
    TranscriptProvenance.PLATFORM_CAPTIONS: None,
    TranscriptProvenance.PLATFORM_TRANSCRIPT: "transcript",
    TranscriptProvenance.GENERATED_LOCAL: "generated",
    TranscriptProvenance.USER_FILE: "user",
}


def work_dir_for(output_dir: Path, recording: Recording) -> Path:
    return Path(output_dir) / f".3ch0-{naming.safe_component(recording.short_id)}"


# --------------------------------------------------------------------------------------
# Transcripts
# --------------------------------------------------------------------------------------


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise CancelledError("Cancelled.")


def _preferred_format(track: CaptionTrack) -> str | None:
    for fmt in FORMAT_PREFERENCE:
        if fmt in track.formats:
            return fmt
    return next(iter(track.formats), None)


def _describe(track: CaptionTrack) -> str:
    kind = "captions" if track.kind is TranscriptProvenance.PLATFORM_CAPTIONS else "transcript"
    return f"{language_name(track.language)} {kind}"


def _platform_transcripts(recording: Recording, session: ContentSession, mode: SubtitleMode,
                          progress: ProgressSink, cancel: threading.Event | None,
                          refresher: AuthRefresher | None = None) -> TranscriptBundle:
    """Fetch platform tracks per ``mode``.

    A track that exists but cannot be fetched (403, expired, network) is recorded in
    ``bundle.failed`` - it is never reported as a missing language. Authorisation errors
    trigger at most one shared refresh (via ``refresher``) and one retry.
    """
    bundle = TranscriptBundle()
    if mode.kind is SubtitleModeKind.NONE:
        return bundle
    tracks = list(recording.captions)
    if not tracks:
        bundle.absent = True
        bundle.notes.append("No Echo360 captions or transcript were found for this lesson.")
    wanted = mode.languages if mode.kind is SubtitleModeKind.LANGUAGES else ()
    if wanted:
        tracks = [t for t in tracks if any(language_matches(w, t.language) for w in wanted)]
        unknown = [t for t in recording.captions if not t.language]
        if unknown and len(tracks) < len(wanted):
            bundle.notes.append(
                f"{len(unknown)} Echo360 caption track(s) have no published language, so "
                "they cannot be matched to --subtitles; use --subtitles auto to include them.")
    for track in tracks:
        _check_cancel(cancel)
        try:
            transcript = _fetch_with_refresh(session, track, refresher, progress)
        except Echo360DLError as exc:
            bundle.failed.append(Failure(
                step="subtitles", target=f"captions:{track.language or track.id}",
                kind=exc.kind, message=f"Could not fetch {_describe(track)}: "
                                       f"{scrub(exc.message)}",
                hint=scrub(exc.hint) if exc.hint else None))
            continue
        if transcript.language is None and track.language:
            transcript.language = track.language
        if not transcript.cues:
            bundle.notes.append(f"{_describe(track)} is empty; skipped.")
            continue
        if not transcript.timed:
            bundle.notes.append(f"{_describe(track)} has no timestamps; saved as text only "
                                "(timestamps are never invented).")
        progress.debug(f"Fetched {_describe(track)} ({len(transcript.cues)} cues)")
        bundle.transcripts.append(transcript)
    for lang in wanted:
        # Missing means: no platform track in that language exists at all.
        if not any(language_matches(lang, t.language) for t in tracks):
            bundle.missing_languages.append(lang)
    if (mode.kind is SubtitleModeKind.AUTO and recording.captions and not bundle.failed
            and not any(t.timed for t in bundle.transcripts)):
        bundle.notes.append("No timed Echo360 captions are available for this lesson.")
    return bundle


def _fetch_with_refresh(session: ContentSession, track: CaptionTrack,
                        refresher: AuthRefresher | None, progress: ProgressSink) -> Transcript:
    generation = refresher.current()[0] if refresher else 0
    try:
        return session.fetch_transcript(track, _preferred_format(track))
    except AuthError as exc:
        if refresher is None:
            raise
        try:
            refresher.refreshed_since(generation)
        except _RefreshExhausted:
            raise exc from None
    progress.debug(f"{_describe(track)}: retrying with refreshed authorisation")
    return session.fetch_transcript(track, _preferred_format(track))


def _cue_keys(t: Transcript) -> set[tuple[int, str]]:
    return {(round((c.start or 0.0) * 2), " ".join(c.text.lower().split()))
            for c in t.cues if c.text.strip()}


def dedupe_platform_tracks(tracks: list[Transcript], threshold: float = 0.8
                           ) -> tuple[list[Transcript], list[Transcript]]:
    """Drop a platform transcript that duplicates platform captions of the same language.

    OBSERVED (2026-11-07): one lesson exposed both a captions VTT and an automatic
    transcript JSON. They are merged only when at least ``threshold`` of the cues match in
    start time (to 0.5 s) and text; otherwise both are embedded (e.g. human vs automatic).
    """
    keep: list[Transcript] = []
    dropped: list[Transcript] = []
    captions = [t for t in tracks if t.provenance is TranscriptProvenance.PLATFORM_CAPTIONS]
    for t in tracks:
        if t.provenance is TranscriptProvenance.PLATFORM_TRANSCRIPT:
            mine = _cue_keys(t)
            twin = next((c for c in captions
                         if (c.language or "") == (t.language or "") and mine
                         and len(mine & _cue_keys(c)) / max(len(mine), len(_cue_keys(c)))
                         >= threshold), None)
            if twin is not None:
                dropped.append(t)
                if t.label and "automatic" in t.label.lower() and not twin.label:
                    # Same text as Echo360's automatic transcript: the captions are automatic
                    # too, so say so in the track title.
                    replaced = dataclasses.replace(twin, label=t.label)
                    keep = [replaced if k is twin else k for k in keep]
                    captions = [replaced if c is twin else c for c in captions]
                continue
        keep.append(t)
    return keep, dropped


def _scrubbed_raw(raw: bytes, fmt: str | None) -> bytes:
    """Platform JSON may embed signed URLs or tokens: redact them before saving. Caption
    text formats are kept byte-for-byte (their purpose is to preserve the original)."""
    if fmt != "json":
        return raw
    return scrub(raw.decode("utf-8", errors="replace")).encode("utf-8")


def _as_generated(t: Transcript) -> Transcript:
    label = t.label if t.label and "(generated)" in t.label \
        else f"{language_name(t.language)} (generated)"
    return dataclasses.replace(t, provenance=TranscriptProvenance.GENERATED_LOCAL, label=label)


def _maybe_generate(bundle: TranscriptBundle, options: TranscribeOptions,
                    factory: TranscriberFactory | None, audio_provider: Callable[[], Path],
                    progress: ProgressSink, cancel: threading.Event | None) -> Transcript | None:
    """Run local transcription once when enabled and no applicable timed platform
    transcript exists (or ``only_if_missing`` is False). Raises on errors."""
    if not options.enabled:
        return None
    platform_timed = [t for t in bundle.transcripts if t.timed and not t.generated]
    applicable = [t for t in platform_timed
                  if options.language is None or language_matches(options.language, t.language)]
    if options.only_if_missing and applicable:
        bundle.notes.append("Local transcription skipped: Echo360 captions are available.")
        return None
    if factory is None:
        raise DependencyError("Local transcription is not available here.",
                              hint="Install the optional 'transcribe' extra.")
    transcriber = factory(options)
    if cancel is not None and hasattr(transcriber, "cancel_event"):
        transcriber.cancel_event = cancel  # lets a backend stop between segments
    _check_cancel(cancel)
    audio = audio_provider()
    _check_cancel(cancel)
    # The transcriber reports its own (real) progress task; avoid a duplicate one here.
    progress.info("Transcribing the shared audio locally (audio stays on this machine).")
    result = transcriber.transcribe(audio, language=options.language, progress=progress)
    _check_cancel(cancel)
    generated = _as_generated(result)
    bundle.transcripts.append(generated)
    for lang in list(bundle.missing_languages):
        if language_matches(lang, generated.language):
            bundle.missing_languages.remove(lang)
            bundle.notes.append(f"No Echo360 captions in '{lang}'; using the locally "
                                "generated transcript instead.")
    return generated


def _audio_track_for(recording: Recording, rendition: VideoRendition | None,
                     track_id: str | None) -> AudioTrack | None:
    if track_id:
        found = recording.audio_track(track_id)
        if found:
            return found
    if rendition is not None and rendition.audio_group:
        for track in recording.audio_tracks:
            if any(r.group_id == rendition.audio_group for r in track.renditions):
                return track
    if len(recording.audio_tracks) >= 1:
        return recording.audio_tracks[0]
    return None


def _audio_component(track: AudioTrack, rendition: VideoRendition | None,
                     quality: QualitySpec) -> Component:
    audio = select_audio(track, rendition, quality)
    label = "Audio (shared)" if track.shared else "Audio"
    return Component(component_key("audio", audio.id), "audio", audio.id, audio.url, label,
                     target="audio")


def obtain_transcripts(recording: Recording, session: ContentSession, *,
                       subtitles: SubtitleMode, transcribe: TranscribeOptions, work_dir: Path,
                       progress: ProgressSink, transcriber_factory: TranscriberFactory | None,
                       cancel: threading.Event | None = None) -> TranscriptBundle:
    """Platform transcripts per ``subtitles``, plus a locally generated one when
    ``transcribe`` asks for it (audio is downloaded into ``work_dir`` if needed).
    Transcription errors propagate as :class:`Echo360DLError`."""
    progress.stage(Stage.SUBTITLES)
    refresher = AuthRefresher(session)
    bundle = _platform_transcripts(recording, session, subtitles, progress, cancel, refresher)

    def audio_provider() -> Path:
        track = _audio_track_for(recording, None, None)
        if track is None:
            raise UnavailableError("The recording has no downloadable audio to transcribe.")
        component = _audio_component(track, None, QualitySpec())
        work = WorkDir(work_dir, recording.lesson_id)
        ctx = DownloadContext(work, refresher, progress, cancel or threading.Event())
        try:
            return download_component(component, ctx)
        finally:
            work.remove_cookie_files()

    _maybe_generate(bundle, transcribe, transcriber_factory, audio_provider, progress, cancel)
    return bundle


# --------------------------------------------------------------------------------------
# Download run
# --------------------------------------------------------------------------------------


@dataclass
class _Planned:
    kind: OutputKind
    source_index: int | None
    label: str
    video: Component | None
    audio: Component | None
    final: Path
    container: ContainerKind
    audio_language: str = "und"
    videos: list[Component] = dataclasses.field(default_factory=list)  # composite inputs
    layout: str | None = None


class _Run:
    def __init__(self, request: DownloadRequest, session: ContentSession,
                 progress: ProgressSink, factory: TranscriberFactory | None,
                 cancel: threading.Event) -> None:
        self.request = request
        self.recording = request.recording
        self.session = session
        self.refresher = AuthRefresher(session)  # one bounded refresh shared by the run
        self.progress = progress
        self.factory = factory
        self.cancel = cancel
        self.outputs: list[OutputFile] = []
        self.failures: list[Failure] = []
        self.warnings: list[str] = []
        self.media_failed = False
        self.work: WorkDir | None = None
        self.planned: list[_Planned] = []
        self._sub_cache: dict[str, list[SubtitleInput]] = {}
        self._warned: set[str] = set()

    # -- bookkeeping -------------------------------------------------------------------
    @property
    def workdir(self) -> WorkDir:
        if self.work is None:
            raise RuntimeError("work directory not initialised")
        return self.work

    def warn(self, message: str) -> None:
        message = scrub(message)
        if message in self._warned:
            return
        self._warned.add(message)
        self.warnings.append(message)
        self.progress.warn(message)

    def fail(self, step: str, target: str, exc: Echo360DLError, *, media: bool) -> None:
        self.failures.append(Failure(step=step, target=target, kind=exc.kind,
                                     message=scrub(exc.message),
                                     hint=scrub(exc.hint) if exc.hint else None))
        self.progress.warn(f"{target}: {scrub(exc.message)}")
        if media:
            self.media_failed = True

    def result(self, status: ResultStatus | None = None) -> DownloadResult:
        existing = []
        for out in self.outputs:
            with contextlib.suppress(OSError):
                out.size = out.path.stat().st_size
                existing.append(out)
        if status is None:
            # Sidecars alone are not a partial success: at least one media file must exist.
            media = [o for o in existing
                     if o.kind in (OutputKind.VIDEO, OutputKind.AUDIO, OutputKind.COMPOSITE)]
            if not self.failures:
                status = ResultStatus.OK
            elif media:
                status = ResultStatus.PARTIAL
            else:
                status = ResultStatus.FAILED
        return DownloadResult(status=status, outputs=existing, failures=self.failures,
                              warnings=self.warnings, lesson_id=self.recording.lesson_id)

    # -- entry point -------------------------------------------------------------------
    def run(self) -> DownloadResult:
        try:
            return self._run()
        except (CancelledError, KeyboardInterrupt):
            self.cancel.set()
            return self._cancelled()
        finally:
            if self.work is not None:
                self.work.remove_cookie_files()

    def _cancelled(self) -> DownloadResult:
        for planned in self.planned:
            with contextlib.suppress(OSError):
                MuxJob(planned.final, planned.container, None).temp_path.unlink()
        if self.work is not None and self.work.path.exists():
            self.warnings.append(
                f"Cancelled. Partial downloads were kept in {self.work.path}; run the same "
                "command again to resume.")
        else:
            self.warnings.append("Cancelled.")
        self.progress.warn(self.warnings[-1])
        return self.result(ResultStatus.CANCELLED)

    def _run(self) -> DownloadResult:
        request = self.request
        try:
            require_tools("ffmpeg", "ffprobe", "yt-dlp")
        except DependencyError as exc:
            self.fail("setup", "tools", exc, media=False)
            return self.result()
        self.encoder_name: str | None = None
        if request.layout is not Layout.SEPARATE and not request.audio_only:
            try:
                self.encoder_name = resolve_encoder(request.encoder.value)
            except DependencyError as exc:
                self.fail("setup", "encoder", exc, media=False)
                return self.result()
        self.progress.stage(Stage.SELECT)
        planned = self._plan()
        if request.chapters.kind is ChapterModeKind.FILE and planned:
            # Fail fast: a bad chapter file is a usage problem, found before downloading.
            try:
                if request.chapters.file is None:
                    raise ValidationError("--chapters file requires a chapter file.")
                chapter_file = load_chapter_file(request.chapters.file)
                # Structure only: the metadata duration is unverified; the measured media
                # duration is checked after download.
                problems = validate_chapters(chapter_file.chapters, None)
                if problems:
                    raise ValidationError(f"Invalid chapter file ({chapter_file.source_detail}): "
                                          + "; ".join(problems))
            except Echo360DLError as exc:
                self.fail("chapters", "chapter file", exc, media=False)
                return self.result()
        if not self._prepare_transcriber():
            return self.result()
        request.output_dir.mkdir(parents=True, exist_ok=True)
        self.planned = self._apply_if_exists(planned)
        if not self.planned:
            return self.result()
        _check_cancel(self.cancel)
        self.work = WorkDir(work_dir_for(request.output_dir, self.recording),
                            self.recording.lesson_id)
        # Temp names are ".<final stem>.3ch0-tmp…": a custom --name carries no
        # "[<short_id>]", so match on its own prefix instead.
        remove_stale_temps(request.output_dir,
                           "." + naming.custom_basename(request.name) if request.name
                           else f"[{naming.lesson_tag(self.recording)}]")

        self.progress.stage(Stage.DOWNLOAD)
        components: dict[str, Component] = {}
        for p in self.planned:  # audio first: it is shared by every output
            if p.audio:
                components.setdefault(p.audio.key, p.audio)
        for p in self.planned:
            for video in ([p.video] if p.video else []) + p.videos:
                components.setdefault(video.key, video)
        paths = self._download_all(list(components.values()))
        infos = self._measure(components, paths)

        _check_cancel(self.cancel)
        bundle = self._transcripts(infos, components)
        _check_cancel(self.cancel)
        self.progress.stage(Stage.CHAPTERS)
        durations = [i.duration + (i.start_time or 0.0) for i in infos.values()
                     if i.duration is not None]
        duration = max(durations) if durations else self.recording.duration
        chapter_set = self._chapters(duration)
        _check_cancel(self.cancel)

        self.progress.stage(Stage.VERIFY)
        for p in self.planned:
            _check_cancel(self.cancel)
            self._produce(p, infos, bundle, chapter_set, duration)
        _check_cancel(self.cancel)
        if request.write_sidecars:
            self._sidecars(bundle, chapter_set)
        elif request.audio_only and any(t.timed for t in bundle.transcripts):
            self.warn("Subtitles are not embedded in .m4a files and sidecars are disabled, "
                      "so no subtitles were saved.")
        self._finish_work_dir()
        return self.result()

    def _prepare_transcriber(self) -> bool:
        """Resolve the backend and model consent *before* downloading, so a missing
        backend or model fails fast (exit 6) instead of after a long transfer. Skipped
        when platform captions will probably make transcription unnecessary."""
        options = self.request.transcribe
        if not options.enabled or self.factory is None:
            return True
        wanted = options.language
        likely_covered = options.only_if_missing and any(
            t.timed is not False and (wanted is None or language_matches(wanted, t.language))
            for t in self.recording.captions)
        if likely_covered:
            return True
        try:
            transcriber = self.factory(options)
        except Echo360DLError as exc:
            self.fail("transcribe", "generated transcript", exc, media=False)
            return False
        original = self.factory
        cached: list[Transcriber] = [transcriber]

        def factory(opts: TranscribeOptions) -> Transcriber:
            return cached[0] if opts == options else original(opts)

        self.factory = factory
        return True

    # -- planning ----------------------------------------------------------------------
    def _plan(self) -> list[_Planned]:
        request, rec = self.request, self.recording
        out_dir = request.output_dir
        if request.audio_only:
            track = _audio_track_for(rec, None, None)
            if track is None:
                self.fail("select", "audio",
                          UnavailableError("No audio track was discovered for this lesson."),
                          media=False)
                return []
            try:
                comp = _audio_component(track, None, request.quality)
            except UnavailableError as exc:
                self.fail("select", "audio", exc, media=False)
                return []
            final = out_dir / (naming.request_media_basename(request, "audio") + ".m4a")
            return [_Planned(OutputKind.AUDIO, None, "audio", None, comp, final, "m4a",
                             iso639_2(track.language))]
        indexes = request.source_indexes or tuple(s.index for s in rec.sources)
        if not indexes:
            self.fail("select", "sources",
                      UnavailableError("No video sources were discovered for this lesson."),
                      media=False)
        ext = ".mkv" if request.container is Container.MKV else ".mp4"
        container: ContainerKind = "mkv" if request.container is Container.MKV else "mp4"
        planned: list[_Planned] = []
        used_labels: set[str] = set()
        for index in indexes:
            source = rec.source(index)
            if source is None:
                self.fail("select", f"Source {index}",
                          UnavailableError(f"Source {index} was not found in this recording."),
                          media=False)
                continue
            try:
                rendition = select_rendition(source, request.quality)
            except UnavailableError as exc:
                self.fail("select", source.display_name, exc, media=False)
                continue
            video = Component(component_key("video", rendition.id), "video", rendition.id,
                              rendition.url, f"{source.display_name} video "
                                             f"({rendition.resolution or rendition.id})",
                              target=source.display_name)
            track = _audio_track_for(rec, rendition, source.audio_track_id)
            audio = None
            if track is not None:
                try:
                    audio = _audio_component(track, rendition, request.quality)
                except UnavailableError as exc:
                    self.warn(f"{source.display_name}: {exc.message} The output will have "
                              "no audio.")
            else:
                self.warn(f"{source.display_name}: no audio track was discovered; the output "
                          "will have no audio.")
            label = source.display_name
            if label.casefold() in used_labels:
                label = f"{label} ({index})"
            used_labels.add(label.casefold())
            final = out_dir / (naming.request_media_basename(request, label) + ext)
            planned.append(_Planned(OutputKind.VIDEO, index, source.display_name, video, audio,
                                    final, container,
                                    iso639_2(track.language if track else None)))
        if request.layout is Layout.SEPARATE:
            return planned
        return self._plan_composite(planned, indexes, ext, container)

    def _plan_composite(self, separate: list[_Planned], indexes: tuple[int, ...], ext: str,
                        container: ContainerKind) -> list[_Planned]:
        request = self.request
        try:
            first, second = composite_pair(indexes, request.layout_order)
        except Echo360DLError as exc:
            self.fail("select", "layout", exc, media=False)
            return []
        by_index = {p.source_index: p for p in separate}
        left, right = by_index.get(first), by_index.get(second)
        if left is None or right is None or left.video is None or right.video is None:
            return separate if request.keep_separate else []  # failure already recorded
        audio = left.audio or right.audio
        arrangement = "side by side" if request.layout is Layout.SIDE_BY_SIDE else "top-bottom"
        label = f"{left.label} + {right.label} ({arrangement})"
        final = request.output_dir / (naming.request_media_basename(request, label) + ext)
        composite = _Planned(OutputKind.COMPOSITE, None, label, None, audio, final, container,
                             left.audio_language, videos=[left.video, right.video],
                             layout=request.layout.value)
        return [composite, *(separate if request.keep_separate else [])]

    def _apply_if_exists(self, planned: list[_Planned]) -> list[_Planned]:
        kept = []
        for p in planned:
            if not p.final.exists():
                kept.append(p)
                continue
            policy = self.request.if_exists
            if policy is IfExists.SKIP:
                self.outputs.append(OutputFile(path=p.final, kind=p.kind,
                                               source_index=p.source_index,
                                               skipped_existing=True))
                self.progress.info(f"Skipping existing file {p.final.name}")
            elif policy is IfExists.ERROR:
                self.fail("output", p.label,
                          UsageError(f"{p.final.name} already exists.",
                                     hint="Use --if-exists overwrite or --if-exists skip."),
                          media=False)
            else:
                kept.append(p)
        return kept

    # -- downloads ---------------------------------------------------------------------
    def _download_all(self, components: list[Component]) -> dict[str, Path]:
        ctx = DownloadContext(self.workdir, self.refresher, self.progress,
                              self.cancel, retries=max(0, self.request.retries))
        results: dict[str, Path] = {}
        # Optional sink extra: list every component up front so a live view can show the
        # ones still waiting for a worker as "queued" (sinks without it are unaffected).
        queued = getattr(self.progress, "task_queued", None)
        if callable(queued):
            for component in components:
                queued(component_task_id(component), component.label, "bytes")
        workers = max(1, min(self.request.concurrency, len(components) or 1))
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers,
                                                     thread_name_prefix="3ch0")
        futures = {pool.submit(download_component, c, ctx): c for c in components}
        pending = set(futures)
        try:
            if callable(queued):
                # Read the playlists of waiting components now (the download reuses them),
                # so the live view knows the overall size from the start.
                for component in components[workers:]:
                    size = prefetch_size(component, ctx)
                    if size:
                        queued(component_task_id(component), component.label, "bytes", size)
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, timeout=0.2, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    self._collect(future, futures[future], results)
        except KeyboardInterrupt:
            self.cancel.set()
            raise
        finally:
            if pending:
                self.cancel.set()
            pool.shutdown(wait=True, cancel_futures=True)
        _check_cancel(self.cancel)
        return results

    def _collect(self, future: concurrent.futures.Future, component: Component,
                 results: dict[str, Path]) -> None:
        target = component.target or component.label
        try:
            results[component.key] = future.result()
        except CancelledError:
            self.cancel.set()
        except Echo360DLError as exc:
            self.fail("download", target, exc, media=True)
        except Exception as exc:  # unexpected library error: report, keep going
            self.fail("download", target, Echo360DLError(scrub(f"{type(exc).__name__}: {exc}")),
                      media=True)

    def _measure(self, components: dict[str, Component],
                 paths: dict[str, Path]) -> dict[str, InputInfo]:
        infos: dict[str, InputInfo] = {}
        for key, path in paths.items():
            comp = components[key]
            try:
                infos[key] = measure(path, comp.kind)
            except Echo360DLError as exc:
                self.fail("download", comp.label, exc, media=True)
                self.workdir.forget(comp)
                self.workdir.discard_partial(comp)
        return infos

    # -- transcripts / chapters --------------------------------------------------------
    def _transcripts(self, infos: dict[str, InputInfo],
                     components: dict[str, Component]) -> TranscriptBundle:
        self.progress.stage(Stage.SUBTITLES)
        request = self.request
        bundle = _platform_transcripts(self.recording, self.session, request.subtitles,
                                       self.progress, self.cancel, self.refresher)
        audio_key = next((k for k, c in components.items() if c.kind == "audio" and k in infos),
                         None)

        def audio_provider() -> Path:
            if audio_key is None:
                raise UnavailableError("No downloaded audio is available to transcribe.")
            return infos[audio_key].path

        try:
            _maybe_generate(bundle, request.transcribe, self.factory, audio_provider,
                            self.progress, self.cancel)
        except CancelledError:
            raise
        except Echo360DLError as exc:
            self.fail("transcribe", "generated transcript", exc, media=False)
        for note in bundle.notes:
            self.warn(note)
        for failure in bundle.failed:
            self.failures.append(failure)
            self.progress.warn(f"{failure.target}: {failure.message}")
        for lang in bundle.missing_languages:
            self.fail("subtitles", f"captions:{lang}", LanguageUnavailableError(
                f"No Echo360 captions or transcript in '{lang}' for this lesson.",
                hint="Use --subtitles auto to take whatever exists, or --transcribe to "
                     "generate one locally."), media=False)
        return bundle

    def _chapters(self, duration: float | None) -> ChapterSet | None:
        try:
            chosen, warnings = choose_chapters(self.request.chapters, self.recording.chapters,
                                               duration)
        except Echo360DLError as exc:
            self.fail("chapters", "chapters", exc, media=False)
            return None
        for warning in warnings:
            self.warn(warning)
        if chosen is None or not chosen.chapters:
            return None
        problems = validate_chapters(chosen.chapters, duration)
        if problems and chosen.provenance is ChapterProvenance.NATIVE:
            # Native chapters come from metadata we cannot fully verify: skip, don't fail.
            self.warn("Native chapters were not embedded because they are inconsistent with "
                      "the recording: " + "; ".join(problems))
            return None
        if problems:
            self.fail("chapters", f"{chosen.provenance.value} chapters",
                      ValidationError("Chapters were not embedded: " + "; ".join(problems)),
                      media=False)
            return None
        return ChapterSet(resolve_chapters(chosen.chapters, duration), chosen.provenance,
                          chosen.source_detail)

    # -- muxing ------------------------------------------------------------------------
    def _subtitle_inputs(self, bundle: TranscriptBundle, container: ContainerKind,
                         duration: float | None) -> list[SubtitleInput]:
        if container == "m4a":
            return []
        if container in self._sub_cache:
            return self._sub_cache[container]
        tracks, merged = dedupe_platform_tracks([t for t in bundle.transcripts if t.timed])
        if merged:
            self.warn("The Echo360 automatic transcript has the same text and timing as the "
                      "Echo360 captions; embedded once (its original file is kept as a "
                      ".platform sidecar).")
        default_index = None
        wanted = self.request.default_subtitle
        if wanted:
            matches = [i for i, t in enumerate(tracks) if language_matches(wanted, t.language)]
            if len(matches) == 1:
                default_index = matches[0]
            else:
                self.warn(f"Default subtitle '{wanted}' matched {len(matches)} tracks; no "
                          "subtitle track is marked default.")
        inputs: list[SubtitleInput] = []
        for i, t in enumerate(tracks):
            mkv = container == "mkv"
            cues = t.cues
            problems = validate_cues(cues, duration, allow_overlap=mkv)
            if problems:
                cues = normalize_cues(cues, duration, trim_overlaps=not mkv)
                self.warn(f"{track_title(t)}: adjusted cues for embedding "
                          f"({'; '.join(problems[:3])}{' …' if len(problems) > 3 else ''}).")
            if not cues:
                self.warn(f"{track_title(t)}: no usable cues; not embedded.")
                continue
            path = self.workdir.path / f"sub-{i}-{container}.vtt"
            path.write_text(to_vtt(dataclasses.replace(t, cues=cues), voice_tags=mkv), "utf-8")
            inputs.append(SubtitleInput(path=path, language=iso639_2(t.language),
                                        title=track_title(t), default=i == default_index))
        self._sub_cache[container] = inputs
        return inputs

    def _encode_composite(self, p: _Planned, videos: list[InputInfo]) -> InputInfo | None:
        """Step 1 of a combined layout: encode the stacked picture into the work dir
        (reused on a rerun once complete). Step 2 is the ordinary stream-copy mux."""
        spec = CompositeSpec(videos=videos, layout=p.layout or "side-by-side",
                             encoder=self.encoder_name or "libx264")
        work = self.work.path if self.work else self.request.output_dir
        keys = "-".join(v.key for v in p.videos)
        out = work / f"composite-{spec.layout}-{keys}.mp4"
        task = f"encode:{p.final.name}"
        total = spec.duration
        if not out.exists():
            self.progress.task_start(task, f"Encoding {p.label} ({spec.encoder})",
                                     total=total, unit="seconds")

            def on_progress(seconds: float) -> None:
                self.progress.task_update(task, completed=min(seconds, total or seconds),
                                          total=total)

            try:
                encode_composite(spec, out, cancel=self.cancel, on_progress=on_progress)
            except CancelledError:
                self.progress.task_done(task, ok=False, message="cancelled")
                raise
            except Echo360DLError as exc:
                self.progress.task_done(task, ok=False, message=scrub(exc.message))
                self.fail("encode", p.label, exc, media=True)
                return None
            if total:
                self.progress.task_update(task, completed=total, total=total)
            self.progress.task_done(task, ok=True)
        else:
            self.progress.info(f"Reusing the already encoded {p.label}.")
        try:
            info = measure(out, "video")
        except Echo360DLError as exc:
            self.fail("encode", p.label, exc, media=True)
            return None
        if spec.size and (info.width, info.height) != spec.size:
            self.fail("verify", p.label, VerificationError(
                f"The combined picture is {info.width}x{info.height}, expected "
                f"{spec.size[0]}x{spec.size[1]}."), media=True)
            return None
        return info

    def _produce(self, p: _Planned, infos: dict[str, InputInfo], bundle: TranscriptBundle,
                 chapter_set: ChapterSet | None, duration: float | None) -> None:
        video = infos.get(p.video.key) if p.video else None
        audio = infos.get(p.audio.key) if p.audio else None
        videos = [infos.get(v.key) for v in p.videos]
        if (p.video and video is None) or (p.audio and audio is None) or None in videos:
            return  # the component failure is already recorded
        expected_size = None
        if p.videos:
            video = self._encode_composite(p, [v for v in videos if v is not None])
            if video is None:
                return  # failure recorded
            expected_size = (video.width, video.height)
        job = MuxJob(output=p.final, container=p.container, audio=audio, video=video,
                     subtitles=self._subtitle_inputs(bundle, p.container, duration),
                     chapters=list(chapter_set.chapters) if chapter_set else [],
                     title=self.recording.title, audio_language=p.audio_language,
                     work_dir=self.work.path if self.work else None,
                     expected_size=expected_size)
        task = f"mux:{p.final.name}"
        self.progress.task_start(task, f"Writing {p.final.name}", total=None, unit="items")
        try:
            tmp = mux(job, cancel=self.cancel)
            report = verify(tmp, job)
            if not report.ok:
                kept = keep_unverified(tmp, job)
                self.progress.task_done(task, ok=False, message="verification failed")
                self.fail("verify", p.label, VerificationError(
                    f"{p.final.name} failed verification: " + "; ".join(report.problems),
                    hint=(f"The file was kept for inspection as {kept.name}." if kept
                          else "The unverified file was removed.")), media=True)
                return
            finalize(tmp, p.final)
        except CancelledError:
            self.progress.task_done(task, ok=False, message="cancelled")
            raise
        except Echo360DLError as exc:
            self.progress.task_done(task, ok=False, message=scrub(exc.message))
            self.fail("mux", p.label, exc, media=True)
            return
        except OSError as exc:
            self.progress.task_done(task, ok=False, message="could not finalise")
            self.fail("finalize", p.label, Echo360DLError(
                f"Could not move the finished file into place as {p.final.name}: "
                f"{exc.strerror or exc}"), media=True)
            return
        self.progress.task_done(task, ok=True)
        provenance = chapter_set.provenance.value if chapter_set else None
        self.outputs.append(OutputFile(path=p.final, kind=p.kind, source_index=p.source_index,
                                       provenance=provenance, verify=report,
                                       size=p.final.stat().st_size))

    # -- sidecars ----------------------------------------------------------------------
    def _write_sidecar(self, name: str, data: bytes, kind: OutputKind, language: str | None,
                       provenance: str | None) -> None:
        path = self.request.output_dir / name
        if path.exists():
            if self.request.if_exists is IfExists.SKIP:
                self.outputs.append(OutputFile(path=path, kind=kind, language=language,
                                               provenance=provenance, skipped_existing=True))
                return
            if self.request.if_exists is IfExists.ERROR:
                self.fail("output", name, UsageError(f"{name} already exists.",
                                                     hint="Use --if-exists overwrite or skip."),
                          media=False)
                return
        tmp = path.with_name(f".{path.stem}{TEMP_MARK}{path.suffix}")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)
        except OSError as exc:
            with contextlib.suppress(OSError):
                tmp.unlink()
            self.fail("output", name, Echo360DLError(f"Could not write {name}: {exc.strerror}"),
                      media=False)
            return
        except BaseException:  # KeyboardInterrupt/SIGTERM: never leave a temp behind
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        self.outputs.append(OutputFile(path=path, kind=kind, language=language,
                                       provenance=provenance, size=len(data)))

    def _sidecars(self, bundle: TranscriptBundle, chapter_set: ChapterSet | None) -> None:
        base = naming.request_basename(self.request)
        used: set[tuple[str, ...]] = set()
        kept, duplicates = dedupe_platform_tracks([t for t in bundle.transcripts if t.timed])
        for t in [*kept, *(t for t in bundle.transcripts if not t.timed)]:
            self._transcript_sidecars(base, used, t, derived=True)
        for t in duplicates:  # same text as kept captions: only the original platform file
            self._transcript_sidecars(base, used, t, derived=False)
        if chapter_set is not None:
            data = json.dumps(chapters_to_json(chapter_set), ensure_ascii=False, indent=2)
            self._write_sidecar(naming.with_suffix(base, "chapters", "json"),
                                (data + "\n").encode("utf-8"), OutputKind.CHAPTERS, None,
                                chapter_set.provenance.value)

    def _transcript_sidecars(self, base: str, used: set[tuple[str, ...]], t: Transcript, *,
                             derived: bool) -> None:
        parts: tuple[str, ...] = (naming.language_tag(t.language),)
        if _SIDECAR_KIND[t.provenance]:
            parts += (_SIDECAR_KIND[t.provenance],)  # type: ignore[operator]
        unique, n = parts, 2
        while unique in used:
            unique, n = (*parts, str(n)), n + 1
        used.add(unique)
        prov = t.provenance.value
        if t.timed and derived:
            self._write_sidecar(naming.with_suffix(base, *unique, "vtt"),
                                to_vtt(t).encode("utf-8"), OutputKind.SUBTITLE,
                                t.language, prov)
            self._write_sidecar(naming.with_suffix(base, *unique, "srt"),
                                to_srt(t).encode("utf-8"), OutputKind.SUBTITLE,
                                t.language, prov)
        if derived:
            self._write_sidecar(naming.with_suffix(base, *unique, "txt"),
                                to_txt(t).encode("utf-8"), OutputKind.TRANSCRIPT,
                                t.language, prov)
        if (not t.generated and t.raw and t.raw_format
                and (loses_information(t.raw, t.raw_format) or not derived)):
            raw_kind = (OutputKind.SUBTITLE if t.raw_format in ("vtt", "srt")
                        else OutputKind.TRANSCRIPT)
            self._write_sidecar(naming.with_suffix(base, *unique, "platform", t.raw_format),
                                _scrubbed_raw(t.raw, t.raw_format), raw_kind,
                                t.language, prov)

    # -- cleanup -----------------------------------------------------------------------
    def _finish_work_dir(self) -> None:
        if self.work is None:
            return
        self.work.remove_cookie_files()
        if self.media_failed:
            self.warn(f"Partial downloads were kept in {self.work.path}; run the same command "
                      "again to retry (completed parts are reused).")
        elif not self.request.keep_work_dir:
            shutil.rmtree(self.work.path, ignore_errors=True)


def run_download(request: DownloadRequest, session: ContentSession, *, progress: ProgressSink,
                 transcriber_factory: TranscriberFactory | None = None,
                 cancel: threading.Event | None = None) -> DownloadResult:
    """Download, mux, verify and finalise the requested outputs (see module docstring).

    Never raises for per-output problems: they become :class:`Failure` entries.
    KeyboardInterrupt and ``cancel`` both yield status CANCELLED.
    """
    run = _Run(request, session, progress, transcriber_factory, cancel or threading.Event())
    return run.run()
