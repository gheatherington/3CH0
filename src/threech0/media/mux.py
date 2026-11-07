"""Muxing (stream copy), verification and atomic finalisation.

Timing decision (evidence-based)
--------------------------------
Live Echo360 components (2026-11-06): video ``start_time=0.066016``, audio
``start_time=0.000000``. ``0.066016`` is 1014/15360, i.e. two frames at 30 fps, the
composition delay of an H.264 stream with B-frames in fMP4 without an edit list. Our
fixtures packaged with ffmpeg's HLS fMP4 muxer reproduce exactly that value.

Video and audio share one HLS media timeline in the player: audio sample 0 plays at
t=0 and the first video frame is *presented* at t=0.066. Captions and chapters from
the platform are relative to that same timeline (the player's ``currentTime``).

* ffmpeg's default behaviour normalises *each input file* to start at 0, which would
  shift the video 66 ms earlier than the audio (measured: output video start 0.000).
  We therefore mux with ``-copyts`` so all streams keep their original timestamps
  (measured: MP4 keeps 0.066016 via an edit list; MKV keeps 0.066 at 1 ms precision).
* Subtitle cues and chapters are written unshifted on that timeline (t=0 = audio
  start); no offset is added or removed. The first subtitle packet in the output is
  at exactly the cue start time.
* Verification compares the measured video-minus-audio start offset of the output with
  that of the downloaded inputs (tolerance :data:`SYNC_TOLERANCE`).

Containers
----------
* MP4: subtitles as ``mov_text`` (speakers rendered as ``Name: text``; overlapping
  cues trimmed since samples cannot overlap). Track titles are written both as
  ``title`` (udta ``name``) and ``handler_name``. MP4 has no separate "default"
  subtitle flag: the mov muxer marks the first track of a type as *enabled* when none
  is default, which ffprobe reports as ``default=1``; at most one subtitle is ever
  default.
* MKV: subtitles are the WebVTT text stream-copied (keeps ``<v Speaker>`` spans and
  overlaps); ``default`` is exactly what was requested.
* M4A (audio only): audio + chapters. Subtitles are *not* embedded in M4A because
  audio players generally ignore text tracks there; they are written as sidecars.
* Chapters come from an ffmetadata input (``-map_chapters``/``-map_metadata``), which
  also carries the global title.

Finalisation and verification failures
--------------------------------------
Output is written to a hidden temporary name in the output directory, verified with
ffprobe, then moved into place with ``os.replace`` (atomic on one filesystem). An
interrupted or failed finalisation removes the temporary file; the final name never
holds a truncated file. When verification fails the file is kept for inspection as
``<name>.UNVERIFIED.<ext>`` and reported as a ``verification`` failure; the final
name is not created.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from threech0.media.chapters import to_ffmetadata
from threech0.media.ffmpeg import count_packets, ffprobe_json, report_from_probe, run_ffmpeg
from threech0.models import Chapter, VerifyReport

ContainerKind = Literal["mp4", "mkv", "m4a"]

#: Allowed difference (s) between output and input video-audio start offsets.
SYNC_TOLERANCE = 0.002
CHAPTER_TOLERANCE = 0.001 + 1e-6
DURATION_TOLERANCE = 1.0
PACKET_TOLERANCE = 1
TEMP_MARK = ".3ch0-tmp"
UNVERIFIED_MARK = ".UNVERIFIED"


@dataclass
class InputInfo:
    """Measured properties of a downloaded component."""

    path: Path
    codec: str | None = None
    width: int | None = None
    height: int | None = None
    packets: int | None = None
    start_time: float | None = None
    duration: float | None = None


def measure(path: Path, kind: Literal["video", "audio"]) -> InputInfo:
    data = ffprobe_json(path)
    report = report_from_probe(data)
    stream = next((s for s in report.streams if s.codec_type == kind), None)
    if stream is None:
        from threech0.errors import VerificationError

        raise VerificationError(f"{path.name}: no {kind} stream in the downloaded file.")
    info = InputInfo(path=path, codec=stream.codec_name, width=stream.width,
                     height=stream.height, start_time=stream.start_time,
                     duration=stream.duration or report.duration)
    if kind == "video":
        info.packets = count_packets(path, "v:0")
    return info


@dataclass
class SubtitleInput:
    path: Path  # WebVTT file prepared for this container
    language: str  # ISO 639-2 ("eng", "und")
    title: str
    default: bool = False


@dataclass
class MuxJob:
    output: Path  # final path
    container: ContainerKind
    audio: InputInfo | None
    video: InputInfo | None = None
    subtitles: list[SubtitleInput] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)  # resolved (ends set)
    title: str | None = None
    audio_language: str = "und"
    work_dir: Path | None = None
    #: For a composite intermediate: the exact picture size the output must have.
    expected_size: tuple[int, int] | None = None

    @property
    def temp_path(self) -> Path:
        return self.output.with_name(f".{self.output.stem}{TEMP_MARK}{self.output.suffix}")

    @property
    def unverified_path(self) -> Path:
        return self.output.with_name(f"{self.output.stem}{UNVERIFIED_MARK}"
                                     f"{self.output.suffix}")

    @property
    def expected_duration(self) -> float | None:
        values = [i.duration + (i.start_time or 0.0) for i in (self.video, self.audio)
                  if i is not None and i.duration is not None]
        return max(values) if values else None


def _even(value: float) -> int:
    """Nearest even integer >= 2 (H.264 4:2:0 needs even dimensions)."""
    return max(2, round(value / 2) * 2)


@dataclass
class CompositeSpec:
    """Two sources stacked into one picture. ``videos`` are in display order
    (first = left/top); ``encoder`` is a resolved ffmpeg encoder name."""

    videos: list[InputInfo]
    layout: str  # "side-by-side" | "top-bottom"
    encoder: str

    @property
    def size(self) -> tuple[int, int] | None:
        if any(not v.width or not v.height for v in self.videos):
            return None
        tiles = composite_tile_sizes(self)
        if self.layout == "side-by-side":
            return sum(w for w, _h in tiles), tiles[0][1]
        return tiles[0][0], sum(h for _w, h in tiles)

    @property
    def duration(self) -> float | None:
        ends = [v.duration + (v.start_time or 0.0) for v in self.videos
                if v.duration is not None]
        return max(ends) if ends else None


def composite_tile_sizes(spec: CompositeSpec) -> list[tuple[int, int]]:
    """Per-source scaled size: common height (side by side) or width (top/bottom), the
    larger source's, each source keeping its aspect ratio (within one pixel)."""
    videos = [(v.width or 2, v.height or 2) for v in spec.videos]
    if spec.layout == "side-by-side":
        height = _even(max(h for _w, h in videos))
        return [(_even(w * height / h), height) for w, h in videos]
    width = _even(max(w for w, _h in videos))
    return [(width, _even(h * width / w)) for w, h in videos]


_ENCODERS: set[str] | None = None


def available_encoders() -> set[str]:
    global _ENCODERS
    if _ENCODERS is None:
        import subprocess

        try:
            out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True,
                                 text=True, timeout=30, check=False).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        _ENCODERS = {line.split()[1] for line in out.splitlines()
                     if len(line.split()) > 1 and line.lstrip().startswith("V")}
    return _ENCODERS


def resolve_encoder(choice: str) -> str:
    """ffmpeg encoder for a composite: VideoToolbox (Apple hardware) when available for
    ``auto``, otherwise libx264. Raises DependencyError if the explicit choice is missing."""
    from threech0.errors import DependencyError

    have = available_encoders()
    if choice in ("auto", "videotoolbox") and "h264_videotoolbox" in have:
        return "h264_videotoolbox"
    if choice == "videotoolbox":
        raise DependencyError("The VideoToolbox H.264 encoder is not available in this ffmpeg.",
                              hint="Use --encoder x264 (software).")
    if "libx264" in have:
        return "libx264"
    raise DependencyError("No H.264 encoder (libx264 or VideoToolbox) is available in ffmpeg.",
                          hint="Install an ffmpeg build with libx264: brew install ffmpeg")


#: Bitrate for VideoToolbox composites: inputs' combined video bitrate x this factor,
#: within [floor, cap]. Lecture video is mostly static, so the inputs' own (low) bitrates
#: are the right reference; a generous factor keeps handwriting/slide text sharp.
VT_BITRATE_FACTOR = 2.0
VT_MIN_BPS = 0.012  # bits per pixel per frame
#: Absolute floor. Live (2026-11-07): 360p top-bottom at 150 kb/s left handwriting soft and
#: the camera half blocky; 720p at ~590 kb/s was crisp.
VT_MIN_BITRATE = 300_000
VT_MAX_BITRATE = 6_000_000


def _input_bitrate(info: InputInfo) -> float | None:
    try:
        size = info.path.stat().st_size
    except OSError:
        return None
    return size * 8 / info.duration if info.duration else None


def composite_bitrate(spec: CompositeSpec, fps: float = 30.0) -> int:
    rates = [_input_bitrate(v) for v in spec.videos]
    size = spec.size or (1280, 720)
    floor = max(VT_MIN_BITRATE, VT_MIN_BPS * size[0] * size[1] * fps)
    target = VT_BITRATE_FACTOR * sum(r for r in rates if r) if all(rates) else floor * 2
    return int(min(VT_MAX_BITRATE, max(floor, target)))


def _encoder_args(spec: CompositeSpec) -> list[str]:
    if spec.encoder == "h264_videotoolbox":
        # VideoToolbox's constant-quality mode produced ~10x the inputs' size on lecture
        # video (live, 2026-11-07); an explicit bitrate derived from the inputs does not.
        rate = composite_bitrate(spec)
        return ["-c:v", "h264_videotoolbox", "-b:v", str(rate), "-maxrate", str(rate * 2),
                "-bufsize", str(rate * 4), "-allow_sw", "1"]
    # x264 CRF adapts to static content by itself.
    return ["-c:v", "libx264", "-crf", "23", "-preset", "veryfast"]


def composite_filter(spec: CompositeSpec) -> str:
    """Scale both sources to a common height (side by side) or width (top/bottom), keep
    each one's aspect ratio, then stack. Explicit even sizes (not scale=-2, whose rounding
    is hard to predict) make the output size deterministic and verifiable."""
    sizes = composite_tile_sizes(spec)
    stack = "hstack" if spec.layout == "side-by-side" else "vstack"
    parts = [f"[{i}:v:0]scale={w}:{h},setsar=1[v{i}]" for i, (w, h) in enumerate(sizes)]
    inputs = "".join(f"[v{i}]" for i in range(len(sizes)))
    parts.append(f"{inputs}{stack}=inputs={len(sizes)}:shortest=0,format=yuv420p[vout]")
    return ";".join(parts)


#: Echo360's own video tracks use a 1/15360 time base (observed, 2026-11-06).
DEFAULT_TIMESCALE = 15360


def _source_timescale(spec: CompositeSpec) -> int:
    """Keep the source's video track timescale. OBSERVED (ffmpeg 9.0, 54-min lecture):
    after a re-encode at timescale 90000/30000/1000, the stream-copy mux that adds chapters
    silently lost the QuickTime chapter track ("Referenced QT chapter track not found");
    at the source's 15360 it is kept, exactly as for separate files."""
    try:
        streams = ffprobe_json(spec.videos[0].path).get("streams", [])
        base = next(s["time_base"] for s in streams if s.get("codec_type") == "video")
        denominator = int(str(base).split("/", 1)[1])
        return denominator if 1000 <= denominator <= 1_000_000 else DEFAULT_TIMESCALE
    except Exception:
        return DEFAULT_TIMESCALE


def encode_composite(spec: CompositeSpec, out_path: Path, *,
                     cancel: threading.Event | None = None,
                     on_progress: Callable[[float], None] | None = None) -> Path:
    """Step 1 of a combined layout: encode the stacked picture alone (no audio, subtitles
    or chapters) to ``out_path``. Step 2 is an ordinary stream-copy :func:`mux` of this file
    with the audio, subtitles and chapters - the path verified live for separate files.

    ``-copyts`` keeps absolute timestamps, so the picture stays in sync with the audio.
    Written to a temporary name and renamed only when ffmpeg succeeds.
    """
    tmp = out_path.with_name(f".{out_path.stem}{TEMP_MARK}{out_path.suffix}")
    args = ["-y", "-copyts"]
    for video in spec.videos:
        args += ["-i", str(video.path)]
    args += ["-filter_complex", composite_filter(spec), "-map", "[vout]", "-an", "-sn",
             "-dn", "-map_metadata", "-1", "-map_chapters", "-1",
             *_encoder_args(spec), "-pix_fmt", "yuv420p", "-tag:v", "avc1",
             "-video_track_timescale", str(_source_timescale(spec)), "-f", "mp4", str(tmp)]
    try:
        run_ffmpeg(args, cancel=cancel, on_progress=on_progress)
        os.replace(tmp, out_path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    return out_path


def build_command(job: MuxJob, out_path: Path, ffmeta: Path) -> list[str]:
    args: list[str] = ["-y", "-copyts"]
    maps: list[str] = []
    n = 0
    if job.video is not None:
        args += ["-i", str(job.video.path)]
        maps += ["-map", f"{n}:v:0"]
        n += 1
    if job.audio is not None:
        args += ["-i", str(job.audio.path)]
        maps += ["-map", f"{n}:a:0"]
        n += 1
    subs = job.subtitles if job.container != "m4a" else []
    for sub in subs:
        args += ["-i", str(sub.path)]
        maps += ["-map", f"{n}:s:0"]
        n += 1
    args += ["-i", str(ffmeta)]
    meta_index = n
    args += maps
    args += ["-map_metadata", str(meta_index),
             "-map_chapters", str(meta_index) if job.chapters else "-1"]
    if job.video is not None:
        args += ["-c:v", "copy", "-disposition:v:0", "default"]
    if job.audio is not None:
        args += ["-c:a", "copy", "-disposition:a:0", "default",
                 "-metadata:s:a:0", f"language={job.audio_language}"]
    if subs:
        args += ["-c:s", "copy" if job.container == "mkv" else "mov_text"]
    for i, sub in enumerate(subs):
        args += [f"-metadata:s:s:{i}", f"language={sub.language}"]
        if job.container == "mkv":
            args += [f"-metadata:s:s:{i}", f"title={sub.title}"]
        else:
            # MP4 track names live in handler_name; a "title" tag is written as a udta
            # 'name' box that ffmpeg's own reader then warns about ("UDTA parsing failed").
            args += [f"-metadata:s:s:{i}", f"handler_name={sub.title}"]
        args += [f"-disposition:s:{i}", "default" if sub.default else "0"]
    if job.container == "mkv":
        args += ["-f", "matroska"]
    else:
        args += ["-movflags", "+faststart", "-f", "mp4"]
    args.append(str(out_path))
    return args


def mux(job: MuxJob, *, cancel: threading.Event | None = None,
        on_progress: Callable[[float], None] | None = None) -> Path:
    """Write the muxed file to :attr:`MuxJob.temp_path` and return it. On any error
    or interruption the temporary file is removed before the exception propagates."""
    ffmeta_dir = job.work_dir or job.output.parent
    ffmeta = ffmeta_dir / f"{job.output.stem[:60]}.ffmeta.txt"
    ffmeta.write_text(to_ffmetadata(job.chapters, title=job.title), "utf-8")
    tmp = job.temp_path
    try:
        run_ffmpeg(build_command(job, tmp, ffmeta), cancel=cancel, on_progress=on_progress)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    finally:
        with contextlib.suppress(OSError):
            ffmeta.unlink()
    return tmp


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------


def verify(path: Path, job: MuxJob) -> VerifyReport:
    """Check stream counts/languages/dispositions, stream-copied video, chapters,
    duration and A/V start offset of ``path`` against ``job``'s inputs."""
    try:
        data = ffprobe_json(path)
    except Exception as exc:  # VerificationError / DependencyError
        return VerifyReport(ok=False, problems=[getattr(exc, "message", str(exc))])
    report = report_from_probe(data)
    problems = report.problems
    by_type: dict[str, list] = {"video": [], "audio": [], "subtitle": []}
    for s in report.streams:
        if s.codec_type in by_type:
            by_type[s.codec_type].append(s)
    want_subs = job.subtitles if job.container != "m4a" else []
    expected = {"video": 1 if job.video else 0, "audio": 1 if job.audio else 0,
                "subtitle": len(want_subs)}
    for kind, count in expected.items():
        if len(by_type[kind]) != count:
            problems.append(f"expected {count} {kind} stream(s), found {len(by_type[kind])}")
    langs = [s.language or "und" for s in by_type["subtitle"]]
    if len(langs) == len(want_subs) and langs != [s.language for s in want_subs]:
        problems.append(f"subtitle languages {langs} != expected "
                        f"{[s.language for s in want_subs]}")
    if sum(1 for s in by_type["subtitle"] if s.default) > 1:
        problems.append("more than one subtitle track is marked default")
    if job.container == "mkv":
        wanted_defaults = [s.default for s in want_subs]
        got = [s.default for s in by_type["subtitle"]]
        if len(got) == len(wanted_defaults) and got != wanted_defaults:
            problems.append(f"subtitle default flags {got} != requested {wanted_defaults}")
    if job.expected_size and by_type["video"]:
        out_v = by_type["video"][0]
        if (out_v.width, out_v.height) != job.expected_size:
            problems.append(f"picture is {out_v.width}x{out_v.height}, expected "
                            f"{job.expected_size[0]}x{job.expected_size[1]}")
    if job.video and by_type["video"]:
        out_v = by_type["video"][0]
        ref = job.video
        if (out_v.codec_name, out_v.width, out_v.height) != (ref.codec, ref.width, ref.height):
            problems.append(f"video changed: {out_v.codec_name} {out_v.width}x{out_v.height} "
                            f"vs input {ref.codec} {ref.width}x{ref.height} (not a stream copy)")
        if ref.packets is not None:
            packets = count_packets(path, "v:0")
            if packets is None or abs(packets - ref.packets) > PACKET_TOLERANCE:
                problems.append(f"video frame count {packets} differs from input "
                                f"{ref.packets} (not a stream copy)")
    if job.audio and by_type["audio"] and by_type["audio"][0].codec_name != job.audio.codec:
        problems.append("audio codec changed (not a stream copy)")
    v_ref = job.video.start_time if job.video else None
    if job.video and job.audio and by_type["video"] and by_type["audio"]:
        v_in, a_in = v_ref, job.audio.start_time
        v_out, a_out = by_type["video"][0].start_time, by_type["audio"][0].start_time
        if None not in (v_in, a_in, v_out, a_out):
            drift = (v_out - a_out) - (v_in - a_in)  # type: ignore[operator]
            if abs(drift) > SYNC_TOLERANCE:
                problems.append(f"audio/video start offset changed by {drift * 1000:.1f} ms")
    problems += _chapter_problems(report.chapters, job.chapters)
    if job.chapters and job.container in ("mp4", "m4a") and not any(
            s.codec_type == "data" for s in report.streams):
        # ffprobe can still list chapters from the Nero 'chpl' atom, but Apple players need
        # the QuickTime chapter track; ffmpeg can drop it silently (seen on 2026-11-07).
        problems.append("the QuickTime chapter track is missing (Apple players would not "
                        "show the chapters)")
    expected_duration = job.expected_duration
    if expected_duration is not None and report.duration is not None \
            and abs(report.duration - expected_duration) > DURATION_TOLERANCE:
        problems.append(f"duration {report.duration:.3f}s differs from the inputs "
                        f"({expected_duration:.3f}s)")
    report.ok = not problems
    return report


def _chapter_problems(found: list[Chapter], wanted: list[Chapter]) -> list[str]:
    if len(found) != len(wanted):
        return [f"expected {len(wanted)} chapter(s), found {len(found)}"]
    problems = []
    for n, (got, want) in enumerate(zip(found, wanted, strict=True), start=1):
        if got.title != want.title:
            problems.append(f"chapter {n}: title {got.title!r} != {want.title!r}")
        want_start = round(want.start * 1000) / 1000
        if abs(got.start - want_start) > CHAPTER_TOLERANCE:
            problems.append(f"chapter {n}: starts at {got.start:.3f}s, expected "
                            f"{want_start:.3f}s")
        if want.end is not None and got.end is not None:
            want_end = round(want.end * 1000) / 1000
            # The last chapter's end may be derived from the file duration by the
            # container, so it gets the duration tolerance; all others are exact.
            tolerance = DURATION_TOLERANCE if n == len(wanted) else CHAPTER_TOLERANCE
            if abs(got.end - want_end) > tolerance:
                problems.append(f"chapter {n}: ends at {got.end:.3f}s, expected "
                                f"{want_end:.3f}s")
    return problems


# --------------------------------------------------------------------------------------
# Finalisation
# --------------------------------------------------------------------------------------


def _atomic_replace(src: Path, dst: Path) -> None:
    os.replace(src, dst)


def finalize(tmp: Path, final: Path) -> Path:
    """Atomically move a verified temporary file to its final name. If that fails the
    temporary file is removed and the exception propagates; ``final`` is untouched."""
    try:
        _atomic_replace(tmp, final)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    return final


def keep_unverified(tmp: Path, job: MuxJob) -> Path | None:
    try:
        os.replace(tmp, job.unverified_path)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink()
        return None
    return job.unverified_path


#: A temp file untouched for this long cannot belong to a live mux (ffmpeg writes
#: continuously), so it is treated as left over from a killed run.
STALE_TEMP_AGE = 120.0


def remove_stale_temps(output_dir: Path, lesson_tag: str | None = None, *,
                       now: float | None = None) -> None:
    """Delete temporary files left by a hard-killed earlier run *of this lesson*.

    Only names containing ``lesson_tag`` (the ``[<short_id>]`` part of every output name)
    and not modified for :data:`STALE_TEMP_AGE` seconds are removed, so a concurrent run
    writing into the same folder is never disturbed.
    """
    now = time.time() if now is None else now
    for path in output_dir.glob(f".*{TEMP_MARK}*"):
        if lesson_tag and lesson_tag not in path.name:
            continue
        with contextlib.suppress(OSError):
            if now - path.stat().st_mtime >= STALE_TEMP_AGE:
                path.unlink()
