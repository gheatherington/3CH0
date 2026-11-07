"""Facade between the CLI and the implementation packages.

The CLI only calls functions on a :class:`Services` instance. Tests replace it via
:func:`set_services`. Real implementations are imported lazily so ``--help`` and
``doctor`` stay fast and work when optional pieces are missing.

Exact function names below are the contract each area implements:

* ``echo360_dl.echo360.urls.parse_classroom_url(url) -> ClassroomRef``
* ``echo360_dl.echo360.discovery.discover(ref, options, progress) -> (Recording, ContentSession)``
* ``echo360_dl.echo360.browser.login(profile_dir, *, start_url, timeout, progress) -> AuthStatus``
* ``echo360_dl.echo360.browser.check_status(profile_dir, *, timeout) -> AuthStatus``
* ``echo360_dl.echo360.browser.browser_installed() -> tuple[bool, str]``
* ``echo360_dl.media.pipeline.run_download(request, session, *, progress, transcriber_factory,
  cancel) -> DownloadResult``
* ``echo360_dl.media.pipeline.obtain_transcripts(recording, session, *, subtitles, transcribe,
  work_dir, progress, transcriber_factory, cancel) -> TranscriptBundle``
* ``echo360_dl.media.chapters.load_chapter_file(path) -> ChapterSet``
* ``echo360_dl.media.chapters.validate_chapters(chapters, duration) -> list[str]``
* ``echo360_dl.media.ffmpeg.probe_report(path) -> VerifyReport``
* ``echo360_dl.media.ffmpeg.tool_versions() -> dict[str, str | None]``
* ``echo360_dl.transcribe.get_transcriber(options, *, confirm_download) -> Transcriber``
* ``echo360_dl.transcribe.available_backends() -> list[BackendInfo]``
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from echo360_dl.models import (
    AuthStatus,
    ChapterSet,
    ClassroomRef,
    ContentSession,
    DiscoveryOptions,
    DownloadRequest,
    DownloadResult,
    ProgressSink,
    Recording,
    SubtitleMode,
    TranscribeOptions,
    Transcriber,
    VerifyReport,
)

ConfirmFn = Callable[[str], bool]


class Services:
    """Real implementation. Every method may raise :class:`Echo360DLError`."""

    # -- discovery / auth --------------------------------------------------------------
    def parse_url(self, url: str) -> ClassroomRef:
        from echo360_dl.echo360.urls import parse_classroom_url

        return parse_classroom_url(url)

    def discover(self, ref: ClassroomRef, options: DiscoveryOptions,
                 progress: ProgressSink) -> tuple[Recording, ContentSession]:
        from echo360_dl.echo360.discovery import discover

        return discover(ref, options, progress)

    def login(self, profile_dir: Path, *, start_url: str | None, timeout: float,
              progress: ProgressSink) -> AuthStatus:
        from echo360_dl.echo360.browser import login

        return login(profile_dir, start_url=start_url, timeout=timeout, progress=progress)

    def auth_status(self, profile_dir: Path, *, timeout: float = 30.0) -> AuthStatus:
        from echo360_dl.echo360.browser import check_status

        return check_status(profile_dir, timeout=timeout)

    # -- media -------------------------------------------------------------------------
    def download(self, request: DownloadRequest, session: ContentSession, *,
                 progress: ProgressSink, confirm_download: ConfirmFn | None,
                 cancel: threading.Event | None = None) -> DownloadResult:
        from echo360_dl.media.pipeline import run_download

        return run_download(request, session, progress=progress,
                            transcriber_factory=self._transcriber_factory(confirm_download),
                            cancel=cancel)

    def transcripts(self, recording: Recording, session: ContentSession, *,
                    subtitles: SubtitleMode, transcribe: TranscribeOptions, work_dir: Path,
                    progress: ProgressSink, confirm_download: ConfirmFn | None,
                    cancel: threading.Event | None = None) -> Any:
        from echo360_dl.media.pipeline import obtain_transcripts

        return obtain_transcripts(recording, session, subtitles=subtitles, transcribe=transcribe,
                                  work_dir=work_dir, progress=progress,
                                  transcriber_factory=self._transcriber_factory(confirm_download),
                                  cancel=cancel)

    def load_chapters(self, path: Path) -> ChapterSet:
        from echo360_dl.media.chapters import load_chapter_file

        return load_chapter_file(path)

    def validate_chapters(self, chapters: ChapterSet, duration: float | None) -> list[str]:
        from echo360_dl.media.chapters import validate_chapters

        return validate_chapters(chapters.chapters, duration)

    def probe(self, path: Path) -> VerifyReport:
        from echo360_dl.media.ffmpeg import probe_report

        return probe_report(path)

    # -- environment -------------------------------------------------------------------
    def tool_versions(self) -> dict[str, str | None]:
        from echo360_dl.media.ffmpeg import tool_versions

        return tool_versions()

    def browser_installed(self) -> tuple[bool, str]:
        from echo360_dl.echo360.browser import browser_installed

        return browser_installed()

    def transcription_backends(self) -> list[Any]:
        from echo360_dl.transcribe import available_backends

        return available_backends()

    @staticmethod
    def _transcriber_factory(confirm: ConfirmFn | None
                             ) -> Callable[[TranscribeOptions], Transcriber]:
        def factory(options: TranscribeOptions) -> Transcriber:
            from echo360_dl.transcribe import get_transcriber

            return get_transcriber(options, confirm_download=confirm)

        return factory


_services: Services | None = None


def get_services() -> Services:
    global _services
    if _services is None:
        _services = Services()
    return _services


def set_services(services: Services | None) -> None:
    """Install a replacement (tests). ``None`` restores the real implementation."""
    global _services
    _services = services
