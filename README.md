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
   - [transcribe — Transcribe a single file or YouTube URL](#2-transcribe--transcribe-a-single-file-or-youtube-url)
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
- **Transcribes any single local audio/video file or YouTube URL** via the `transcribe` command (SRT, VTT, TXT, JSON output).
- **Captures and transcribes live system audio** in real-time via WASAPI Loopback (`realtime` command, Windows only).
- Includes an `audit` command to detect missing, orphaned, or duplicate transcript files.
- Includes a `search` command to find and export transcripts by keyword, phrase, regex, video ID, or playlist.
- Writes detailed metadata reports in a `_meta` folder per channel.

---

## Architecture / Module Overview

```
transcriptex/
├── __init__.py          # Package version (0.1.0)
├── __main__.py          # Enables: python -m transcriptex
├── cli.py               # Argument parser & command dispatcher (6 subcommands)
├── models.py            # Shared dataclasses: VideoRecord, PlaylistRecord, ChannelInfo, SubtitleTrack, TranscriptResult
├── url_utils.py         # YouTube URL normalization, video/playlist ID extraction
├── fs_utils.py          # Filename sanitization, transcript filename builder, directory helpers
├── youtube_client.py    # YouTubeClient class: channel discovery, playlist/video indexing, audio download via yt-dlp
├── subtitle_parser.py   # Subtitle parsers: VTT/SRT, JSON3, TTML/XML/SRV
├── transcriber.py       # ChannelTranscriber class: full channel export pipeline + CSV/JSON reports
├── asr_engine.py        # ASREngine class: lazy-loading wrapper around faster-whisper WhisperModel
├── asr.py               # Batch ASR pipeline: reads video_index.csv, downloads audio, transcribes, updates CSV
├── transcribe_single.py # SingleTranscriber class: transcribes one local file or YouTube URL
└── realtime.py          # RealtimeTranscriber class: WASAPI loopback capture + live Whisper transcription
```

**Key design points:**
- All heavy imports (`faster_whisper`, `torch`, `sounddevice`, `pyaudiowpatch`) are lazy — only loaded when actually used.
- `ASREngine` handles CUDA availability detection and falls back to `cpu/int8` automatically.
- The `asr` pipeline saves `video_index.csv` incrementally after each video, making it crash-safe and resumable.
- `ChannelTranscriber` uses `YouTubeClient` for all network calls, keeping concerns separated.

---

## Installation

### Requirements

- Python 3.10+
- `yt-dlp >= 2026.2.4`
- `ffmpeg` on PATH (required for audio extraction — download from [ffmpeg.org](https://ffmpeg.org/))

```bash
pip install -r requirements.txt
```

### Optional — ASR / Realtime (GPU strongly recommended)

```bash
# faster-whisper for ASR and single file transcription
pip install faster-whisper

# CUDA-enabled PyTorch for GPU acceleration (RTX recommended)
pip install torch --index-url https://download.pytorch.org/whl/cu121

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

### 2. `transcribe` — Transcribe a Single File or YouTube URL

Transcribes **any local audio/video file** (MP4, MKV, MP3, WAV, etc.) or a **YouTube URL** using local `faster-whisper`.

```bash
transcriptex transcribe <input_path> [options]
```

When given a local file, the file is read directly — **no download occurs**.  
When given a YouTube URL, audio is temporarily downloaded via yt-dlp, then deleted after transcription.

#### Options

| Flag | Default | Description |
|---|---|---|
| `-o`, `--output` | *(same folder as input)* | Output directory. For local files: same folder as the input. For URLs: current dir |
| `--format` | `txt` | Output format (see below) |
| `--model` | `large-v3-turbo` | Whisper model size |
| `--device` | `auto` | `auto` (GPU if available, else CPU), `cuda`, or `cpu` |
| `--compute-type` | `auto` | `auto` (float16 for GPU, int8 for CPU), or explicit: `float16`, `int8_float16`, `int8`, `float32` |
| `--language` | *(auto-detect)* | Language code, e.g. `es`, `en`. Empty = auto-detect |
| `--beam-size` | `5` | Beam search width (higher = better quality, slower) |
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
| `--beam-size` | `5` | Beam search size |
| `--batch-size` | `8` | Audio chunk batch size |
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

---

### 4. `realtime` — Live System Audio Transcription

Captures audio playing on your PC (browsers, media players, meetings, etc.) via **WASAPI Loopback** and transcribes it live to the console.

> **Windows only.** Requires `pyaudiowpatch` and `numpy`.

```bash
transcriptex realtime
```

Press **Ctrl+C** to stop.

#### Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `medium` | Whisper model (smaller = lower latency) |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--compute-type` | `float16` | Quantization type |
| `--language` | *(auto-detect)* | Language code. Empty = auto-detect |
| `--beam-size` | `1` | Beam size (1 = fastest for live use) |
| `--chunk-duration` | `3.0` | Audio buffer window duration in seconds |
| `--process` | *(empty)* | Captures audio only from a specific process by name (e.g. `zen.exe`). Requires Windows 10/11 Application Loopback. |

#### Example

```bash
transcriptex realtime --model medium --language es --chunk-duration 3.0
```

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
- **ffmpeg is required** for the `asr` and `transcribe` commands (audio extraction). Install it and add it to your PATH.
- **CUDA fallback:** If CUDA is unavailable, `ASREngine` automatically switches to `cpu` with `int8` quantization.
- **Windows environments:** Set `HF_HUB_DISABLE_SYMLINKS=1` (already applied in `asr.py`) to avoid permission errors when downloading Whisper model weights as a non-admin user.
- **Restricted videos** (private, members-only, region-blocked) will fail gracefully and be logged in `video_index.csv`.
- **Incremental runs:** Re-running `transcriptex run` skips videos that already have a transcript file. Use `--force` to regenerate all.
- **Realtime is Windows-only** (WASAPI Loopback via `pyaudiowpatch`). It is not available on Linux/macOS.
