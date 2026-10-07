# transcriptEX

> CLI tool to export transcripts from YouTube channels, transcribe local media files and YouTube URLs, capture realtime audio, audit transcript collections, and search across transcripts — all without an official YouTube API key.

**Version:** `0.1.0` | **Python:** 3.10+ | **Package name:** `transcriptex`

---

## Table of Contents

1. [What this tool does](#what-this-tool-does)
2. [Architecture / Module Overview](#architecture--module-overview)
3. [Installation](#installation)
4. [Commands](#commands)
   - [run — Export YouTube channel transcripts](#1-run--export-youtube-channel-transcripts)
   - [transcribe — Transcribe files or video URLs](#2-transcribe--transcribe-files-or-video-urls)
   - [asr — Batch ASR fallback for missing subtitles](#3-asr--batch-asr-fallback-for-missing-subtitles)
   - [realtime — Live system audio transcription](#4-realtime--live-system-audio-transcription)
   - [audit — Audit an existing channel output folder](#5-audit--audit-an-existing-channel-output-folder)
   - [search — Search transcripts](#6-search--search-transcripts)
5. [Output Layout](#output-layout)
6. [Notes & Tips](#notes--tips)

---

## What This Tool Does

- Accepts a YouTube channel URL and exports transcripts for every video, organized by playlist folders.
- Indexes videos across channel tabs (`videos`, `shorts`, `streams`).
- Fetches YouTube's native subtitle tracks (manual + auto-generated) — **no audio downloaded** in the main flow.
- Falls back to local GPU transcription (`faster-whisper`) via the `asr` command for videos with no subtitles.
- **Transcribes local audio/video files or YouTube URLs** via the `transcribe` command (SRT, VTT, TXT, JSON output).
- **Captures and transcribes live system audio** through PulseAudio/PipeWire on Linux or WASAPI Loopback on Windows.
- Includes an `audit` command to detect missing, orphaned, or duplicate transcript files.
- Includes a `search` command to find and export transcripts by keyword, phrase, regex, video ID, or playlist.
- Writes detailed metadata reports in a `_meta` folder per channel.

---

## Architecture / Module Overview

```
transcriptex/
├── __init__.py          # Package version (0.1.0)
├── __main__.py          # Enables: python -m transcriptex
├── cli.py               # Argument parser & command dispatcher (8 subcommands)
├── models.py            # Shared dataclasses: VideoRecord, PlaylistRecord, ChannelInfo, SubtitleTrack, TranscriptResult
├── url_utils.py         # YouTube URL normalization, video/playlist ID extraction
├── fs_utils.py          # Filename sanitization, transcript filename builder, directory helpers
├── youtube_client.py    # YouTubeClient class: channel discovery, playlist/video indexing, audio download via yt-dlp
├── subtitle_parser.py   # Subtitle parsers: VTT/SRT, JSON3, TTML/XML/SRV
├── transcriber.py       # ChannelTranscriber class: full channel export pipeline + CSV/JSON reports
├── asr_engine.py        # ASREngine class: lazy-loading wrapper around faster-whisper WhisperModel
├── asr.py               # Batch ASR pipeline: reads video_index.csv, downloads audio, transcribes, updates CSV
├── transcribe_single.py # SingleTranscriber class: transcribes one local file or YouTube URL
├── subtitle_segments.py # Timed subtitle cues for URL exports
├── live_decoder.py      # Pause-aware bounded Linux recognition with provisional previews
├── realtime_linux.py    # PipeWire/PulseAudio capture and full-recording finalization
└── realtime.py          # Realtime configuration and Windows WASAPI backend
```

**Key design points:**
- Heavy inference and capture imports (`faster_whisper`, `numpy`, `pyaudiowpatch`) are lazy — only loaded when actually used. PyTorch and sounddevice are not required.
- `ASREngine` handles CUDA availability detection and falls back to `cpu/int8` automatically.
- The `asr` pipeline saves `video_index.csv` incrementally after each video, making it crash-safe and resumable.
- `ChannelTranscriber` uses `YouTubeClient` for all network calls, keeping concerns separated.

---

## Installation

### Requirements

- Python 3.10+
- `yt-dlp >= 2026.2.4`
- `ffmpeg` on PATH for the channel ASR extraction pipeline and media conversion. Single-file ASR decodes through PyAV; single-video URLs keep compressed audio.
- Linux realtime: PulseAudio client utilities (`parec`, `pactl`), also supported on PipeWire.

```bash
pip install -r requirements.txt
```

### Optional — ASR / Realtime (GPU strongly recommended)

```bash
# faster-whisper for ASR and single file transcription
pip install faster-whisper

# faster-whisper uses CTranslate2 for GPU acceleration; PyTorch is not its inference engine.
# Configure the compatible CUDA/cuDNN runtime for CTranslate2 on your platform.

# Realtime audio capture (Windows only)
pip install pyaudiowpatch numpy
```

### Install as CLI tool

```bash
# Editable install (recommended for development)
pip install -e .

# Or install from source
pip install .
```

After installation, the `transcriptex` command is available globally.

Alternatively, run without installing:

```bash
python -m transcriptex <command> [args]
```

### Codex integration

The reusable Codex skill is in `integrations/codex/transcriptex/`, including
its small and large transparent PNG icons and `agents/openai.yaml` metadata.
Install the CLI first, then copy the skill from this checkout:

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills/transcriptex"
cp -R integrations/codex/transcriptex/. "${CODEX_HOME:-$HOME/.codex}/skills/transcriptex/"
```

Invoke `$transcriptex` in Codex. If its icon is cached, reopen Codex to reload
the skill metadata. FFmpeg remains available for media preparation and editing.
Generated transcripts, scratch audio, logs and installation metadata are kept
locally and ignored by Git.

---

## Commands

### 1. `run` — Export YouTube Channel Transcripts

Downloads subtitle tracks for every video in a YouTube channel and saves them as `.txt` files organized by playlist.

```bash
transcriptex run "https://www.youtube.com/@channel_handle"
# or equivalently (backward-compat shortcut):
transcriptex "https://www.youtube.com/@channel_handle"
```

#### Options

| Flag | Default | Description |
|---|---|---|
| `-o`, `--output` | `./output` | Root output directory |
| `--lang-priority` | `es,en` | Comma-separated subtitle language preference order |
| `--tabs` | `videos,shorts,streams` | Channel tabs to index |
| `--retries` | `2` | Retry attempts for each video |
| `--timeout` | `30` | Network timeout in seconds |
| `--sleep` | `0.0` | Sleep between videos (seconds) |
| `--force` | — | Regenerate transcripts even if the file already exists |
| `--verbose` | — | Enable yt-dlp verbose output |

#### Example

```bash
transcriptex run "https://www.youtube.com/@mychannel" \
  --output ./exports \
  --lang-priority es,en \
  --retries 3 \
  --force
```

**Exit codes:** `0` = success, `1` = fatal error, `2` = completed with failures.

---

### 2. `transcribe` — Transcribe Files or Video URLs

Transcribes **local audio/video files** (MP4, MKV, MP3, WAV, etc.) with local `faster-whisper`. Video URLs prefer existing manual/automatic subtitles and fall back to local ASR when unavailable. One invocation can accept several inputs and reuse one ASR model.

```bash
transcriptex transcribe <input_path> [more_inputs ...] [options]
```

When given a local file, the file is read directly — **no download occurs**.  
For a URL, `--source auto` tries native subtitles first, without loading a model or downloading audio. If subtitles are unavailable, yt-dlp downloads compressed audio temporarily and PyAV decodes it directly. `--source asr` forces recognition; `--source subtitles` fails instead of using ASR. The JSON report includes source, language, timing and the output path.

#### Options

| Flag | Default | Description |
|---|---|---|
| `-o`, `--output` | *(same folder as input)* | Output directory. For local files: same folder as the input. For URLs: current dir |
| `--format` | `txt` | Output format (see below) |
| `--model` | `large-v3-turbo` | Whisper model size |
| `--device` | `auto` | `auto` (GPU if available, else CPU), `cuda`, or `cpu` |
| `--compute-type` | `auto` | `auto` (float16 for GPU, int8 for CPU), or explicit: `float16`, `int8_float16`, `int8`, `float32` |
| `--language` | *(auto-detect)* | Language code, e.g. `es`, `en`. Empty = auto-detect |
| `--beam-size` | `1` | Beam search width; `5` explores more candidates at greater cost, without guaranteeing better quality |
| `--source` | `auto` | URL subtitles first; `subtitles` requires captions, `asr` forces recognition |
| `--lang-priority` | `--language` or `es,en` | Subtitle language preferences |
| `--batch-size` | `0` | Auto: 8 on CUDA / 1 on CPU; reduced on GPU OOM |
| `--no-vad` | — | Disable silence filtering and batching |
| `--json` | — | A single JSON report on stdout; diagnostics on stderr |
| `--quiet` | — | Suppress the human summary |
| `--verbose` | — | Enable verbose output |

#### Formats

| `--format` | Output | Description |
|---|---|---|
| **`txt`** | `.txt` | **Default.** Clean plain text. Segments joined with spaces. Blank line inserted on silences > 3s |
| `srt` | `.srt` | Numbered subtitles with timestamps (`00:00:00,000 --> 00:00:00,000`) |
| `vtt` | `.vtt` | WebVTT subtitles, compatible with HTML5 `<track>` |
| `json` | `.json` | Full segment array: `[{"start": 0.0, "end": 2.5, "text": "..."}]` |

#### Examples

```bash
# Minimal — plain text next to the video (default behavior)
transcriptex transcribe "C:\Users\USER\Videos\TEST\2026-02-23 17-27-21.mp4"
# → saves: C:\Users\USER\Videos\TEST\2026-02-23 17-27-21.txt

# Force language (faster than auto-detect)
transcriptex transcribe "C:\Users\USER\Videos\TEST\2026-02-23 17-27-21.mp4" --language es

# Get timestamped subtitles (SRT)
transcriptex transcribe "C:\Users\USER\Videos\TEST\2026-02-23 17-27-21.mp4" --format srt

# Get WebVTT (for HTML5 video players)
transcriptex transcribe "C:\Users\USER\Videos\TEST\2026-02-23 17-27-21.mp4" --format vtt

# Get full segment data with timestamps as JSON
transcriptex transcribe "C:\Users\USER\Videos\TEST\2026-02-23 17-27-21.mp4" --format json

# Force CPU (no GPU needed)
transcriptex transcribe "C:\Users\USER\Videos\TEST\2026-02-23 17-27-21.mp4" --device cpu

# Transcribe a YouTube URL — saves in current directory
transcriptex transcribe "https://www.youtube.com/watch?v=VIDEOID" --language es
```

**Output file name:** For local files, the output file uses the same base name as the input (e.g., `2026-02-23 17-27-21.srt`). For YouTube URLs, it uses `transcript_<timestamp>.<format>`.

---

#### Automation and multiple files

```bash
transcriptex transcribe video.mp4 recording.wav --language es --format json --json -o ./transcripts
transcriptex transcribe "https://www.youtube.com/watch?v=VIDEOID" --source subtitles --json -o ./transcripts
transcriptex transcribe "https://www.youtube.com/watch?v=VIDEOID" --source asr --language es --json -o ./transcripts
```

`--format json` writes the segment array; `--json` prints the command report.
The report names the output file. Multiple inputs return `report.results`; exit 2
means some inputs failed, while successful outputs remain available. Identical
input arguments are processed once. Outputs are written atomically, and duplicate
local stems in a shared output directory receive a path-derived suffix.
`--batch-size 1` selects sequential ASR. Weights load lazily only if ASR is needed.

### 3. `asr` — Batch ASR Fallback for Missing Subtitles

After running `transcriptex run`, videos with no YouTube subtitles are marked `no_subtitles` in `video_index.csv`. This command processes those videos using local Whisper.

```bash
transcriptex asr "output/My Channel"
```

**Features:**
- Reads `_meta/video_index.csv` automatically.
- Downloads audio temporarily (deleted after transcription).
- Updates `video_index.csv` incrementally after each video — crash-safe and resumable.
- Displays a progress bar with ETA.

#### Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `large-v3-turbo` | Whisper model |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--compute-type` | `float16` | Quantization type |
| `--language` | `es` | Language code |
| `--beam-size` | `1` | Beam search size |
| `--batch-size` | `16` | Audio chunk batch size |
| `--force` | — | Re-transcribe videos that already have a file |
| `--limit N` | `0` (no limit) | Process at most N videos per run |
| `--sleep` | `0.0` | Seconds to wait between videos |
| `--include-failed` | — | Also retry videos marked as `failed` (not just `no_subtitles`) |
| `--verbose` | — | Show detailed yt-dlp download logs |

#### Example

```bash
transcriptex asr "output/My Channel" \
  --model large-v3-turbo \
  --device cuda \
  --compute-type float16 \
  --language es \
  --beam-size 5 \
  --limit 50
```

#### Current channel state and resume path

As of 2026-06-01, the `Jesús G. Maestro` channel is in this state:

- 1,988 videos indexed.
- 798 transcripts already existing on disk.
- 94 videos transcribed from YouTube subtitles during the last `run` pass.
- 1,050 videos still marked `no_subtitles` and ready for ASR fallback.
- 46 videos marked `failed` because YouTube returned private/locked metadata.

To resume the paused ASR batch, run the same command again:

```bash
transcriptex asr "output/Jesús G. Maestro"
```

The command is resumable: it reads `_meta/video_index.csv`, skips videos that already have a transcript file, and only processes rows still marked `no_subtitles` unless you add `--include-failed`.

To retry the private/locked videos as well, use cookies:

```bash
transcriptex asr "output/Jesús G. Maestro" --include-failed --cookies cookies.txt
```

---

### 4. `realtime` — Live System Audio Transcription

Linux uses **PulseAudio/PipeWire**; Windows uses **WASAPI** and requires
`pyaudiowpatch`. List devices before selecting a source:

```bash
transcriptex realtime --list-devices
# Linux:
transcriptex realtime --language es --output meeting.txt
```

Press **Ctrl+C** to stop. Linux drains queued audio and confirms the current
utterance, then generates `meeting.final.txt` from the full recording. Audio is
spooled locally into temporary WAV files and removed on completion; add
`--record-audio` to retain separate source WAVs, or `--no-finalize` to skip the
full-recording pass. An abrupt kill cannot perform finalization; the live TXT
already written remains usable. New transcript/audio files request mode 0600 on filesystems that support POSIX permissions.

`[Parcial]` console lines are previews and can change. The live TXT contains only
confirmed segments. Linux commits at pauses instead of cutting every three
seconds. Long utterances use overlapping bounded windows and word timestamps to
avoid repeating the overlap. Source labels are **not speaker diarization**.

| Flag | Default | Description |
|---|---|---|
| `--model` | `large-v3-turbo` | Local Whisper model |
| `--device` | `auto` | CUDA if available, otherwise CPU |
| `--compute-type` | `auto` | float16 on CUDA / int8 on CPU |
| `--language` | auto | Language code |
| `--beam-size` | `1` | Live decoding beam |
| `--chunk-duration` | `3.0` | Linux preview interval / Windows buffer window |
| `--source` | `@DEFAULT_MONITOR@` | Linux audio source |
| `--microphone` | empty | Optional second Linux input source |
| `--monitor-stream` | `-1` | Linux sink-input index to isolate meeting playback |
| `--output` | empty | Required Linux TXT path, appended and flushed |
| `--max-segment-duration` | `15.0` | Maximum Linux segment before overlapping cut; at most 30 s |
| `--silence-duration` | `0.6` | Pause needed to confirm speech |
| `--silence-threshold` | `0.002` | RMS pause threshold |
| `--overlap-duration` | `1.0` | Context retained at a forced cut |
| `--queue-seconds` | `30.0` | Bounded queued audio per source; overflow is an explicit error |
| `--no-preview` | off | Lower GPU usage; only show confirmed segments |
| `--no-finalize` | off | Skip full-recording final pass |
| `--record-audio` | off | Retain source WAVs beside the transcript |
| `--process` | empty | Windows-only application capture by process name |

Source names and monitor stream/sink relationships are validated before loading a model. Capture properties disable PipeWire fallback and reconnection to another source; the runtime checks the capture route every second and fails on mismatches or stalled PCM. Normal Linux sources retry briefly after EOF only while the same source exists. An isolated meeting stream that
ends fails explicitly: find its new index and restart; capture never broadens to
all system audio automatically. Capture errors are recorded and cause a failing
exit status. The available audio may still receive a final transcription. Longer context does not guarantee that the final pass will be more accurate than every live segment.

---

### 5. `audit` — Audit an Existing Channel Output Folder

Compares the `video_index.csv` against actual transcript files on disk and reports:
- Videos indexed but missing a transcript file.
- Transcript files on disk not in the index (orphans).
- Duplicate transcript IDs.

```bash
transcriptex audit "output/My Channel"
```

#### Options

| Flag | Default | Description |
|---|---|---|
| `--max-rows` | `10` | Max missing rows to print in the console summary |
| `--no-write-files` | — | Skip writing audit files to `_meta` |
| `--rename-legacy` | — | Rename old `VIDEOID.txt` files to `Title [VIDEOID].txt` |

#### Audit output files (written to `_meta/`)

| File | Contents |
|---|---|
| `audit_report.json` | Full audit summary in JSON |
| `missing_transcripts.csv` | Videos indexed without a transcript file |
| `orphan_transcripts.csv` | Transcript files not in the index |
| `duplicate_transcripts.csv` | Videos with multiple transcript files found |

---

### 6. `search` — Search Transcripts

Search by keyword, phrase, regex, video ID, title, or playlist. Results are printed to the console and optionally exported.

```bash
transcriptex search "output/My Channel" "search query"
```

#### Options

| Flag | Default | Description |
|---|---|---|
| `--fields` | `title,playlist,video_id,transcript` | Fields to search across |
| `--status` | *(all)* | Filter by status: `ok,existing,no_subtitles,no_text,failed` |
| `--top` | `25` | Maximum results to display |
| `--match-all` | — | All query tokens must match (AND logic) |
| `--regex` | — | Treat query as a regular expression |
| `--case-sensitive` | — | Case-sensitive search |
| `--snippet-len` | `140` | Character length of transcript match snippets |
| `--save-csv` | — | Write top results CSV to `_meta/` |
| `--export-dir` | — | Directory to copy matched transcript files |
| `--export-top N` | `0` | Export top N results |
| `--export-all` | — | Export all shown results |
| `--export-ids` | — | Export specific video IDs (comma-separated) |
| `--export-indices` | — | Export by result index shown in console (1-based) |
| `--zip` | — | Create a ZIP archive of exported files |

#### Examples

```bash
# Simple keyword search
transcriptex search "output/My Channel" "inteligencia artificial"

# Search only in title and transcript, require all words
transcriptex search "output/My Channel" "reforma energética" \
  --fields title,transcript \
  --match-all \
  --top 30

# Export top 10 matches to a folder
transcriptex search "output/My Channel" "cervantes" \
  --export-top 10 \
  --export-dir ./results

# Export specific result indices and ZIP them
transcriptex search "output/My Channel" "quijote" \
  --top 20 \
  --export-indices 1,3,7 \
  --zip
```

---

## Output Layout

```
output/
  Channel_Name/
    Playlist_A [PLxxxx]/
      Video title [VIDEOID].txt
      Another video [VIDEOID].txt
    Playlist_B [PLyyyy]/
      Video title [VIDEOID].txt
    Sin_Playlist/
      Video without playlist [VIDEOID].txt
    _meta/
      playlists.csv          ← All playlists with folder paths
      video_index.csv        ← All videos with status, language, path
      run_report.json        ← Summary of the last `run` execution
      audit_report.json      ← Summary of last `audit` execution
      missing_transcripts.csv
      orphan_transcripts.csv
      duplicate_transcripts.csv
```

Each `.txt` transcript file includes a metadata header:

```
Title: Video Title
Video ID: xxxxxxxxxxx
Video URL: https://www.youtube.com/watch?v=xxxxxxxxxxx
Channel: Channel Name
Subtitle language: es
Subtitle source: auto   ← "auto", "manual", or "asr"
Generated at: 2026-02-24T10:00:00+00:00

-----

[transcript text...]
```

---

## Notes & Tips

- **No API key required.** The tool uses `yt-dlp` to extract metadata and subtitle URLs anonymously.
- **The `run` command never downloads audio.** Only subtitle track bytes are fetched. Use `asr` or `transcribe` for audio-based transcription.
- **ffmpeg remains useful** for the channel `asr` extraction pipeline and conversion. Single-file ASR uses PyAV; caption retrieval needs neither GPU nor audio extraction.
- **ASR is resumable.** Re-running `transcriptex asr` continues from `_meta/video_index.csv` and skips transcript files that already exist on disk.
- **Private videos need auth.** Rows marked `failed` usually need `--include-failed` plus `--cookies` or browser-exported cookies.
- **CUDA fallback:** If CUDA is unavailable, `ASREngine` automatically switches to `cpu` with `int8` quantization.
- **Windows environments:** Set `HF_HUB_DISABLE_SYMLINKS=1` (already applied in `asr.py`) to avoid permission errors when downloading Whisper model weights as a non-admin user.
- **Restricted videos** (private, members-only, region-blocked) will fail gracefully and be logged in `video_index.csv`.
- **Incremental runs:** Re-running `transcriptex run` skips videos that already have a transcript file. Use `--force` to regenerate all.
- **Realtime supports Linux and Windows.** macOS capture is not implemented.

## Linux realtime (PulseAudio / PipeWire)

`transcriptex realtime` supports Linux through `parec` and `pactl`, without WASAPI dependencies. Use `--list-devices` to list sources and `pactl list sink-inputs` to identify the active meeting stream.

```bash
transcriptex realtime --model large-v3-turbo --language es --chunk-duration 3 --source HEADSET.monitor --monitor-stream MEET_INDEX --microphone MICROPHONE_SOURCE --output meeting.txt
```

Replace the source names and numeric stream index with current values. `--microphone` is optional; `--monitor-stream` isolates the chosen playback stream. Without a stream index, the selected source contains all its playback audio. `--process` remains a Windows-only option. Text is appended as UTF-8 and flushed after each segment; timestamps mark recognized segment starts and labels identify audio sources, not individual speakers. Ctrl+C stops capture, drains pending frames and generates a final full-recording transcript by default; the last capture frame can be incomplete. If the meeting recreates its audio stream, stop and select the new sink-input index. Use `--record-audio` if you also want to retain source audio for later correction.
