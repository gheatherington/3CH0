# echo360-dl

A personal command-line tool that inspects and downloads **Echo360 recordings you can already
watch**. It can also embed selectable subtitles and chapter markers, and generate a local
transcript when the platform has none.

> Not an official Echo360 product and not affiliated with Echo360. It uses your own login,
> keeps credentials on your machine, and does not get around DRM or access controls.
> Use it in line with your institution's policies.

Verified with **echo360.ca** classroom URLs on macOS (Apple Silicon), Python 3.12/3.13, FFmpeg
9.0, yt-dlp 2026.10.26 and Playwright Chromium. Other Echo360 regions and operating systems
are untested and not claimed.

## Contents
- [Install](#install)
- [Quick start](#quick-start)
- [Commands](#commands)
- [Selecting sources and quality](#selecting-sources-and-quality)
- [Combined layouts (both sources in one video)](#combined-layouts-both-sources-in-one-video)
- [Subtitles and transcripts](#subtitles-and-transcripts)
- [Chapters](#chapters)
- [Local transcription](#local-transcription)
- [Output files](#output-files)
- [Scripting: JSON, exit codes, non-interactive mode](#scripting-json-exit-codes-non-interactive-mode)
- [Authentication](#authentication)
- [Troubleshooting](#troubleshooting)
- [What has been verified](#what-has-been-verified)
- [Limitations](#limitations)
- [Development](#development)

## Install

Requirements: [uv](https://docs.astral.sh/uv/), FFmpeg (provides `ffprobe`), and Playwright's
Chromium.

```zsh
brew install ffmpeg uv
cd /path/to/3CH0                       # this project folder
uv tool install .                      # installs the `echo360-dl` command (isolated env)
                                       # after updating the code: uv tool install --reinstall .
"$(uv tool dir)/echo360-dl/bin/python" -m playwright install chromium   # one-time (~150 MB)
echo360-dl doctor                      # checks everything
```

Optional local transcription (see [Local transcription](#local-transcription)):

```zsh
uv tool install --force '.[transcribe]'       # faster-whisper (CPU)
uv tool install --force '.[transcribe-mlx]'   # mlx-whisper (Apple GPU), Apple Silicon only
```

The original script still works as a thin wrapper: `python3 download_echo360.py URL` runs
`echo360-dl download URL --streams all` (see [Compatibility](#compatibility)).

## Quick start

```zsh
echo360-dl auth login          # sign in once in the dedicated browser window (SSO/MFA by you)
echo360-dl inspect 'https://echo360.ca/lesson/…/classroom'
echo360-dl download 'https://echo360.ca/lesson/…/classroom'      # interactive selection
```

A download goes through six stages: **discover → select → download → subtitles → chapters
→ verify**. Interactive mode shows the sources it found, asks which ones you want (one,
several, all, or audio only), shows a short plan, and starts after one confirmation.

## Commands

| Command | Purpose |
| --- | --- |
| `echo360-dl doctor` | Checks Python, ffmpeg/ffprobe, yt-dlp, Chromium, the browser profile, the output folder and transcription backends. |
| `echo360-dl auth login` | Opens the dedicated browser so you can sign in. Interactive only. |
| `echo360-dl auth status` | Checks whether the saved session still works (exit 3 if not). |
| `echo360-dl auth logout [--yes]` | Deletes the dedicated browser profile. |
| `echo360-dl inspect URL` | Lists video sources, resolutions, audio, captions/transcripts, chapters and discovery notes. |
| `echo360-dl download URL` | Downloads the selected sources with embedded subtitles and chapters, then verifies them. |
| `echo360-dl transcript URL` | Exports the platform transcript (or a locally generated one) as VTT/SRT/TXT/JSON. |
| `echo360-dl chapters validate FILE` | Validates a chapter JSON file, optionally against `--duration` or `--media`. |
| `echo360-dl chapters show MEDIA` | Lists the chapters actually embedded in a media file (from ffprobe). |

Every command accepts `--json`, `--quiet/-q`, `--verbose/-v`, `--no-color`, `--ascii` and
`--non-interactive`, and `-h` for help. Put these options **after** the subcommand
(`inspect URL --json`). Commands that contact Echo360 also take `--auth`, `--request-file`,
`--timeout` and `--profile-dir`.

### Common workflows

```zsh
# Machine-readable inspection
echo360-dl inspect 'CLASSROOM_URL' --json

# Everything, best quality, platform captions and native chapters if present
echo360-dl download 'CLASSROOM_URL' --streams all --quality best \
  --subtitles auto --chapters auto --output ~/Downloads

# Source 2 only, at most 720p, English captions, chapters from your own file, no prompts
echo360-dl download 'CLASSROOM_URL' --streams 2 --quality 720p \
  --subtitles en --chapters-file chapters.json --non-interactive

# Export the platform transcript as WebVTT
echo360-dl transcript 'CLASSROOM_URL' --language en --format vtt

# No platform transcript? Generate one locally once (shared audio), embed it in every source
echo360-dl download 'CLASSROOM_URL' --streams all --transcribe local --non-interactive \
  --allow-model-download

# Audio only, with chapters every 10 minutes
echo360-dl download 'CLASSROOM_URL' --streams audio --chapters interval --chapter-interval 10m

# MKV container (keeps WebVTT styling and speaker tags)
echo360-dl download 'CLASSROOM_URL' --streams all --container mkv
```

Sample outputs (sanitized, fictional identifiers):
[`examples/inspect.sample.json`](examples/inspect.sample.json) and
[`examples/download-result.sample.json`](examples/download-result.sample.json).

## Selecting sources and quality

- **Sources** are numbered as Echo360 numbers them (`s1`, `s2`, …). A source gets a name such
  as "camera" or "screen" only when lesson metadata reliably says so. Otherwise it is shown as
  *Source 1*, *Source 2*. Source 1 is not assumed to be the camera.
- `--streams all` means every source that is available or not yet checked. A source whose
  playlist returned **403** is skipped with a warning, because a 403 means *forbidden or
  absent*, not *absent*.
  - A source the player itself requested but that then returned 403 can still be requested by
    number (`--streams 2`), and the download is attempted.
  - A source number that was only guessed by name and refused (shown in `inspect` notes as
    "could not access") fails with exit 3 and a "403 — may exist" message, never with "not
    found".
- `--streams audio` downloads audio only (`.m4a`).
- **Quality**: `best` is the highest resolution, `worst` the lowest. `720p` (or `720`) means
  the **best rendition whose height is at most 720**. If every rendition is taller, the
  command fails with exit 4 and lists the available heights.
- By default each selected source becomes **its own file** with the audio attached. The
  recording's shared audio track is downloaded **once** and reused.
- Video and audio are **stream-copied**, so the original quality is kept with no re-encoding.
  The one exception is a combined layout (below), which you have to ask for.
- The whole recording is downloaded, even when the classroom URL contains a `timestamp`
  parameter.

## Combined layouts (both sources in one video)

To watch the professor's screen and the camera together in one video:

```zsh
echo360-dl download 'CLASSROOM_URL' --streams all --layout side-by-side   # left | right
echo360-dl download 'CLASSROOM_URL' --streams all --layout top-bottom     # top / bottom
echo360-dl download 'CLASSROOM_URL' --streams 1,2 --layout side-by-side --layout-order 2,1
```

- `--layout separate` (the default) keeps one file per source. `side-by-side` puts the first
  source on the left and the second on the right; `top-bottom` puts the first on top.
- `--layout-order 2,1` swaps which source comes first. Echo360 doesn't say which source is
  the camera and which is the screen, so check one frame and swap if needed.
- In interactive mode, picking two sources asks for the layout once (Enter = separate).
- A combined layout needs exactly two video sources. With `--streams audio`, or a single
  source, it fails with exit 2 before anything is downloaded.
- How the combined video is built:
  - Both pictures are scaled to the same height (side by side) or width (top/bottom),
    keeping their aspect ratios. With two 720p sources, side-by-side is 2560×720 and
    top-bottom is 1280×1440.
  - **The video is re-encoded**: pictures can't be placed next to each other without
    re-encoding. `--encoder auto` (the default) uses Apple's hardware H.264 encoder
    (VideoToolbox) when ffmpeg has it, and falls back to libx264. Encoding shows real
    progress against the recording's length.
  - Size: with VideoToolbox, the video bitrate is set to about twice the two sources'
    combined bitrate. Lecture video is mostly static, so that is small, but text stays sharp.
    libx264 uses constant quality (CRF 23).
  - The combined picture is encoded first, into the work folder, and is reused if you run
    the same command again. Audio is then stream-copied and the timestamps are kept, so the
    audio stays in sync, and subtitles and chapters are embedded the same way as for
    separate files, including the QuickTime chapter track Apple players need.
  - ffprobe checks the final picture size, sync, duration, subtitles and chapters.
- `--keep-separate` also writes each source as its own original-quality file.

## Subtitles and transcripts

The tool keeps three different things apart, and labels each one in the UI, in the JSON and in
the track title:

| Kind | Meaning |
| --- | --- |
| `platform-captions` | Timed captions published by Echo360 |
| `platform-transcript` | An Echo360 transcript. It may have no timestamps. |
| `generated-local` | Produced on your machine by speech recognition, always labelled *generated* |

- `--subtitles auto` embeds every timed platform caption track it finds. When there are none,
  it says so and the video still downloads.
- A caption track that exists but **cannot be fetched** (403, expired session, network) is
  retried once with refreshed authorisation. If that fails, it is reported as its own failure
  (for example `forbidden`), never as "language not available", and the run is partial
  (exit 5).
- Captions seen only in the player's traffic get their language from the WebVTT `Language:`
  header, a `lang=` URL parameter, or a recognised file name such as `en.vtt`. Otherwise the
  language is shown as unknown, and `--subtitles en` cannot match that track (use `auto`).
- `--subtitles en,fr` asks for specific languages. If one is missing, the media is still
  produced, the missing language is reported as a failure, and the exit code is **5
  (partial)**.
- `--subtitles none` embeds nothing.
- Subtitles are **selectable tracks, never burned in**. MP4 uses `mov_text`; MKV uses WebVTT,
  which keeps speaker `<v>` tags. Each track gets an ISO 639-2 language tag and a title such as
  "English — Echo360 captions" or "English — generated (local)".
- No subtitle track is marked default unless you pass `--default-subtitle LANG`. (In MP4,
  ffmpeg flags the first subtitle track as enabled when none is chosen. Never more than one
  track is default.)
- Timestamps are **never invented**. A transcript without timing can be exported only as
  TXT/JSON; asking for VTT/SRT fails with a clear message.
- Sidecar files: `.vtt`, `.srt`, `.txt`, and the original platform file (`.platform.<ext>`)
  whenever converting it would lose styling or information. Caption files are kept
  byte-for-byte; a platform JSON file has signed URLs and tokens redacted before it is saved.
  `--no-sidecars` turns them off.
- `transcript --format json` always uses echo360-dl's own schema (version 1, with
  provenance), never the platform's raw JSON.

## Chapters

Chapters are written **inside the media container** (FFmpeg chapter metadata) and then read
back with ffprobe. A JSON sidecar on its own never counts as "embedded".

| Mode | Behaviour |
| --- | --- |
| `--chapters auto` (default) | Native lesson chapters when Echo360 metadata provides them. Nothing is ever made up. |
| `--chapters-file FILE` | Your chapter file. **It takes precedence over native chapters** (with a warning). |
| `--chapters interval [--chapter-interval 10m]` | Generated chapters at fixed intervals with neutral titles ("Part 1", …), labelled `generated-interval` |
| `--chapters none` | No chapters |

Chapter file schema, version 1 (see [`examples/chapters.sample.json`](examples/chapters.sample.json)):

```json
{
  "version": 1,
  "chapters": [
    {"start": "00:00:00", "title": "Introduction"},
    {"start": "00:08:30", "title": "Worked example"},
    {"start": "00:21:15", "end": "00:40:00", "title": "Diagonalisation"}
  ]
}
```

- `start`/`end` accept `HH:MM:SS(.mmm)`, `MM:SS`, or a number of seconds.
- When `end` is omitted, it is the next chapter's start, or the media duration for the last
  chapter.
- Validation requires:
  - version 1
  - non-negative times
  - strictly increasing starts
  - `end > start`
  - `end` no later than the next start
  - everything within the media duration
  - non-empty titles

  A chapter file's structure is checked **before** downloading. The duration check runs after
  downloading, against the measured media duration, because the duration in lesson metadata
  is not verified.
- Native chapters that don't fit the recording are skipped with a warning; they don't fail
  the run.

  Titles may contain any Unicode as well as `= ; # \`; these are escaped correctly.

```zsh
echo360-dl chapters validate chapters.json --media 'lecture.mp4'
echo360-dl chapters show 'lecture.mp4'
```

## Local transcription

This is optional and offline, and your audio is never uploaded. Generation happens **only
when you ask** (`--transcribe local`) and, by default, only when there is no platform
transcript. The shared audio is transcribed **once** and the result is embedded in every
selected source.

| Backend | Install (from the project folder) | Hardware |
| --- | --- | --- |
| `faster-whisper` | `uv tool install --force '.[transcribe]'` | **CPU** (int8). CTranslate2 has no Apple GPU backend. |
| `mlx-whisper` | `uv tool install --force '.[transcribe-mlx]'` | Apple GPU via MLX/Metal (Apple Silicon only; heavier install that pulls PyTorch) |

- `--transcribe-backend auto` picks mlx-whisper if it is installed on Apple Silicon, otherwise
  faster-whisper.
- `--transcribe-model` accepts `tiny`, `base`, `small` (default), `medium`, `large-v3` or
  `turbo`, plus `.en` variants.
- **Models are downloaded only with your consent.** Interactive runs ask first and show the
  approximate size (tiny ~75 MB, small ~480 MB, large-v3 ~3.1 GB). Non-interactive runs need
  `--allow-model-download`; otherwise they exit 6 with a hint. This is checked **before
  anything is downloaded**, unless platform captions are expected to make transcription
  unnecessary.
- Ctrl-C stops faster-whisper between segments. mlx-whisper has no way to interrupt, so it
  finishes the current file first.
- Region codes such as `en-CA` are reduced to `en` for the speech model.
- Models are stored in `~/Library/Caches/echo360-dl/models/`.
- Measured here: faster-whisper `tiny` on CPU transcribed a 54-minute lecture in about 2
  minutes on Apple Silicon. Larger models are slower but more accurate.

## Output files

```
<course> - <YYYY-MM-DD> - <title> [<id>] - Source 1.mp4
<course> - <YYYY-MM-DD> - <title> [<id>] - Source 2.mp4
<course> - <YYYY-MM-DD> - <title> [<id>] - audio.m4a           (--streams audio)
<course> - <YYYY-MM-DD> - <title> [<id>].en.vtt / .en.srt / .en.txt
<course> - <YYYY-MM-DD> - <title> [<id>].en.generated.vtt      (local transcription)
<course> - <YYYY-MM-DD> - <title> [<id>].chapters.json
```

- Missing metadata falls back to `Echo360 lesson [<id>]`.
- `<id>` is a stable 10-character hash of the full lesson id. Lessons in the same course share
  a lesson-id prefix, so a hash avoids name collisions.
- Unicode is kept (NFC). Path separators, control characters and reserved characters are
  removed, and names are capped at 180 bytes.

How a download is finalised:
- **Atomic writes**: each file is muxed under a temporary name, verified with ffprobe, then
  renamed. A file that fails verification is kept as `….UNVERIFIED.mp4` and never takes the
  final name.
- **Existing files**: `--if-exists skip` (default; reported as skipped), `overwrite`, or
  `error`.
- **Resume**: downloads run in `<output>/.echo360-dl-<id>/`. If a run is cancelled (Ctrl-C) or
  fails, that folder is kept, and running the same command again reuses finished parts and
  resumes partial ones. It is deleted after a fully successful run (`--keep-work-dir` keeps it).
- **Retries and concurrency**: `--retries` (default 3) and `--concurrency` (default 2) are
  bounded.

## Scripting: JSON, exit codes, non-interactive mode

- `--json` writes **one JSON document to stdout**, with `schema_version`, `command` and `ok`.
  Progress and errors go to **stderr**. The JSON contains no ANSI codes, no cookies, no auth
  headers and no URL query strings. A `--json` run never prompts.
- `--non-interactive` never prompts and **never opens a login window**. A missing choice (for
  example `--streams`) exits 2 and names the flag to add. A missing or expired login exits 3
  and tells you to run `echo360-dl auth login`.
- Interactive mode is used only when stdin and stderr are terminals and neither `--json` nor
  `--non-interactive` is given.
- `--quiet` prints only errors and the final list of outputs. `--verbose` adds debug detail
  and tracebacks, with secrets still redacted.
- `NO_COLOR`/`--no-color`, redirected output (plain lines, no animation), narrow terminals
  (stacked layout below 64 columns) and `--ascii` are all supported. Status labels are always
  words (`[OK]`, `[WARN]`, `[FAIL]`), never colour alone.

| Exit code | Meaning |
| --- | --- |
| 0 | Success (including "nothing to do": files skipped because they already exist) |
| 1 | Unexpected internal error |
| 2 | Usage error, or a required choice is missing in non-interactive mode |
| 3 | Authentication: login required, session expired, or access refused (`unauthorized` 401, `forbidden` 403 that persisted after a refresh) |
| 4 | Requested content unavailable (source, quality, language, transcript) |
| 5 | Partial success: at least one media file was produced, and something else failed (see `failures`). Sidecars alone don't count: if every media file failed, the run fails with that failure's code. |
| 6 | Missing dependency (ffmpeg, Chromium, optional backend, model not downloaded) |
| 7 | Network failure that persisted after the bounded retries |
| 8 | Verification/validation failed (ffprobe check, invalid chapter file) |
| 130 | Cancelled (Ctrl-C). Run the command again to resume. |

## Authentication

Signing in to the classroom page and getting permission to fetch the video are **separate
steps**. Echo360 serves video from `content.echo360.ca` behind short-lived **CloudFront signed
cookies** that are scoped to a single recording. That is why copying only the stream URL gives
HTTP 403.

**Normal path: the dedicated browser profile.**
- `echo360-dl auth login` opens a separate Chromium profile kept in
  `~/Library/Application Support/echo360-dl/browser-profile` (mode 0700). It never touches
  your everyday browser. You complete SSO, MFA and pop-ups yourself; credentials are never
  typed or stored by the tool.
- Later commands reuse that profile **headlessly**. They load the classroom page, watch the
  player's own network traffic, keep only responses that belong to *this* lesson (preloaded
  data for other lessons is discarded), and then fetch the playlists with the browser's
  cookies. Cookies keep their domain, path, secure flag and expiry, and are never sent to other
  hosts or across redirects.
- **When discovery finishes**: it needs at least one playlist for this lesson, plus 3 seconds
  with no new relevant traffic, plus every source listed in metadata accounted for. It stops
  after `--timeout` (default 90 s) in any case. If nothing plays within 8 s, it clicks the play
  control once and records that in the notes.
- **Expired authorisation**: when a download gets a 403 part-way through, the tool refreshes
  the authorisation once (bounded) and retries. If the 403 persists, the failure kind is
  `forbidden` (exit 3), since a 403 cannot show whether the content exists. A 401 is
  `unauthorized` (exit 3).
- **Signed in or not**: `auth login` and `auth status` report "signed in" only when the page
  is not a sign-in page **and** a current Echo360 session cookie is present. `ECHO_JWT` was
  observed in a signed-in request; `PLAY_SESSION` is a guess. Otherwise the state is
  reported as unknown, not as signed in.
- **Cookies for downloads**: yt-dlp receives only the cookies that match the URL being
  downloaded, through a 0600 file in a private temporary folder (never in the output folder,
  which may be synced to iCloud). The file is deleted when the run ends, including on
  Ctrl-C, SIGTERM or a closed terminal; files left by a killed process are removed on the
  next run.
- **Background traffic**: repeated or unrelated JSON requests (heartbeats, analytics) don't
  hold discovery open.
- **Browser profile safety**: echo360-dl uses, and `auth logout` deletes, only directories it
  created (marked with `.echo360-dl-profile`) or the default location. Pointing
  `--profile-dir` at your everyday Chrome profile is refused.

**Troubleshooting path: a saved request.** In the browser's developer tools, right-click the
`s1_av.m3u8` request → *Copy as cURL*, save it to a file, then run:

```zsh
echo360-dl inspect 'CLASSROOM_URL' --request-file ~/path/to/request.txt
```

- The file is **parsed as data and never executed**. Only cookies and a few harmless headers
  are used, and the cookies are kept host-only.
- This mode cannot see lesson metadata, so it gets no titles, transcripts or chapters. It finds
  sources by checking sibling playlist names, and labels that clearly.
- It cannot refresh authorisation, which lasts only a few hours.
- The file contains your session secrets. Keep it private and delete it when you are done.
  (`.gitignore` excludes `*request*.txt`.)

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Login required` (exit 3) | `echo360-dl auth login`, then retry. `auth status` checks the session. |
| Source shown as `forbidden (403 — may exist)` | Normal for sources that don't exist. If the source plays in the browser, sign in again and retry with `--streams N`. |
| `No platform transcript found` | The lesson has none that discovery could see. Use `--transcribe local`. |
| `Speech model … is not downloaded yet` (exit 6) | Add `--allow-model-download`, or run interactively and confirm. |
| `faster-whisper` crashes with `metadata_errors` | PyAV ≥ 17 is incompatible with faster-whisper 1.2.1. The `transcribe` extra pins `av<17`; reinstall with `uv tool install --force '.[transcribe]'`. |
| Playwright says the browser is missing | `"$(uv tool dir)/echo360-dl/bin/python" -m playwright install chromium` (`doctor` prints the exact command) |
| `CERTIFICATE_VERIFY_FAILED` | Use the installed tool (it ships certifi). python.org Python without "Install Certificates" breaks `urllib`. |
| Dev checkout: `ModuleNotFoundError: echo360_dl` from `uv run` | `~/Documents` is synced by iCloud, which keeps marking `.venv` *hidden*, and Python 3.13 skips hidden `.pth` files. Run `chflags -R nohidden .venv`, or keep the dev environment outside iCloud: `export UV_PROJECT_ENVIRONMENT=~/.venvs/echo360-dl`. The installed tool (`uv tool install .`) is not affected. |
| `auth status` says sign-in "could not be confirmed" | The page loaded, but no Echo360 session cookie was found. Run `echo360-dl auth login` again. If it keeps happening, send the `--save-trace` output so the real cookie name can be checked. |
| `… is not an echo360-dl browser profile` | `--profile-dir` points at a non-empty folder echo360-dl didn't create. Use a new folder, or omit the option. |
| Inspect shows no title/chapters | Metadata shapes are not yet verified on real lessons (see below). Run `inspect URL --save-trace trace.json` and look at the sanitized trace (URLs without queries, JSON key structure only, no values). |
| Interrupted download | Run the same command again. It resumes from `<output>/.echo360-dl-<id>/`. |

## What has been verified

**Live-tested** against one real echo360.ca recording, using an imported request (not the browser):
- `inspect` found 2 sources (360p/720p) sharing one audio track. Sources 3 and 4 were reported
  as "403, may exist", not as missing.
- `download --streams all --quality 360p --container mkv --transcribe local --chapters-file …`
  produced two MKVs.
  - ffprobe showed the h264 video copied, keeping its real 0.066 s start offset, and AAC audio
    starting at 0.
  - Each file has one WebVTT track tagged `eng`, titled "English — generated (local)", not
    default.
  - All 4 chapters are embedded with exact boundaries, and the last one ends at the real
    duration.
  - The shared audio was downloaded once.
  - A scan of every output and log found none of the session's secret values.
- yt-dlp handled Echo360's byte-range fMP4 HLS with host-scoped cookies.

**Fixture-tested** (about 400 automated tests, including real ffmpeg/ffprobe runs on
synthetic media served by a local HLS server that requires cookies):
- one- and two-source downloads, quality and source selection, audio only
- 403 on one source, expired authorisation with exactly one refresh
- MP4/MKV subtitle and chapter embedding with languages, titles, dispositions and escaping
- stream copy (frame counts unchanged), A/V sync with a non-zero video start
- cancellation and resume, interrupted finalisation, if-exists handling
- Unicode and malformed subtitles, untimed transcripts, missing languages
- non-interactive behaviour, JSON free of secrets and styling, narrow and no-colour output
- cURL import that never executes anything

**Live-tested through the browser** (2026-11-07, one lesson):
- `auth login` and headless discovery through the dedicated profile, including Echo360's
  `login.echo360.ca` → institution → auth0 sign-in redirects.
- The player requests `s1_v.m3u8` (a video-only master for source 1) as well as
  `s2_av.m3u8`, plus `s0q*` for the shared audio. Discovery maps these to sources 1 and 2.
  This was fixed after the first live run, which reported a phantom "Source 3" and a
  "Source 0".
- Transcript JSON: `/api/ui/echoplayer/lessons/{lesson}/medias/{media}/transcript` returns
  `data.contentJSON.cues[]` (`startMs`, `endMs`, `speaker`, `content`) and
  `data.isAutomated`. Echo360's own automatic transcripts are titled "Echo360 automatic
  captions".
- Captions: `https://captions.echo360.ca/<institution>/captions-<media>-<timestamp>.vtt`.

- Sign-in persistence: after `auth login`, both `auth status` and a headless `inspect` stay
  signed in (session cookie `ECHO_JWT` present). echo360-dl saves and restores the
  echo360.ca cookies itself (`echo360-dl-session.json`, 0600, inside the 0700 profile),
  because Chromium does not keep session cookies between launches.
- This lesson exposes both a captions VTT and an automatic-transcript JSON with the same
  text. When cue timing and text match (≥ 80 %), only one subtitle track is embedded; both
  are kept as sidecar files.

- Lesson details come from the player data embedded in the classroom page
  (`Echo["echoPlayerV2FullApp"]`):
  - the title
  - the course
  - the recording start time, from `lesson.timing.start` and the lesson's time zone, for
    example `2026-09-14T10:28:00-04:00`
  - the duration
  - the list of sources and their playlists (`video.playableMedias`)

  Its `startTimeMillis` is the *playback resume position* and is never used to trim.
- A full download through the browser session (360p, both sources) verified with ffprobe:
  - h264 and AAC stream-copied
  - one subtitle track, "Echo360 automatic captions"
  - chapters every 10 minutes, ending at the real duration
  - file names carry the correct date
  - no secrets in any output, and no cookie files left behind

**Not available from Echo360 (for the lessons seen so far):**
- Camera/screen labels: the player data has only `sourceIndex` and `trackType`, so sources
  are shown as "Source 1", "Source 2".
- Caption language: neither the captions, the transcript nor the player data names it, so
  tracks are tagged `und`. (`locale` is the website's interface language, not the caption
  language.)
- Native chapters: `chapters` was `null` on the lesson tested, so reading platform
  chapters is implemented but has not been exercised live.

## Limitations

- Only `echo360.ca` is supported; other regional domains are rejected until verified.
- Echo360 doesn't say which source is the camera and which is the screen, or what
  language the captions are in (see above), so expect "Source N" and `und` subtitle
  tracks. Use `--default-subtitle` only with a known language.
- Request-file mode has no metadata and cannot refresh authorisation.
- Local transcription quality depends on the model. Small models may produce long cues over
  silence (observed: a 5-minute first cue from `tiny`). Use `small` or larger for lectures.
- MP4 subtitle tracks cannot carry WebVTT styling or voice tags (speakers become
  `Speaker: text`). Use `--container mkv`, or keep the `.platform.vtt` sidecar.
- Audio-only `.m4a` files get chapters, but subtitles only as sidecar files.
- Resuming relies on a workaround for a yt-dlp quirk with byte-range fMP4 playlists. If a
  resumed file has the wrong size, that part is re-downloaded once from scratch.
- Combined layouts are re-encoded, so they are not bit-identical to the original video;
  use the default separate files when original quality matters. Only two sources can be
  combined, in a 50/50 arrangement (no picture-in-picture).
- DRM-protected media is out of scope.

## Compatibility

`download_echo360.py` is now a thin wrapper:

| Old | New |
| --- | --- |
| `download_echo360.py URL` | `echo360-dl download URL --streams all` |
| `--list-formats` | `echo360-dl inspect URL` |
| `--output-dir DIR` | `--output DIR` |
| `--request-file FILE` | `--request-file FILE` (no longer read implicitly from `~/Downloads/work/`) |
| `--browser-profile DIR` | `--profile-dir DIR` (default moved to the app data dir) |
| `--fresh` | default (the browser session is used unless `--request-file` is given) |

The original script is kept for reference in `docs/dev/legacy/download_echo360.original.py`.

## Development

```zsh
uv sync --group dev --extra transcribe
uv run --group dev pytest -q          # ~460 tests, ~35 s, needs ffmpeg
uv run --group dev ruff check src tests
uv run --group dev pytest -m live     # opt-in, needs ECHO360_DL_LIVE_URL and a signed-in profile
PYTHONPATH=src uv run python -m echo360_dl --help   # run the working copy, not the installed tool
```

After changing the code, run `uv tool install --reinstall .` to update the installed command.

**Working with Claude Code** (`CLAUDE.md`, `.claude/`):
- `CLAUDE.md` holds the project rules: secrets, the observed-vs-hypothesis rule for Echo360
  data, layering, commands.
- `/live-test <what to check>` runs a live check against Echo360 through a testing agent. The
  agent is headless and non-interactive, verifies outputs with ffprobe, scans for leaked
  secrets and deletes downloaded lecture files.
- Hooks in `.claude/settings.json`:
  - one blocks agent tool calls that name the saved request, the browser profile or the
    session-cookie file;
  - one runs `ruff check` on every Python file Claude edits.

Architecture: [`docs/architecture.md`](docs/architecture.md). Engineering notes and evidence:
[`docs/dev/BRIEFING.md`](docs/dev/BRIEFING.md).
